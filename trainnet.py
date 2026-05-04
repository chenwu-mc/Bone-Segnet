import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import sys
import random
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from mkunet_network import MK_UNet_ViewBranch
from metrics import dice_coef


# ===================== 随机种子 =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===================== 损失函数 =====================
class FocalTverskyLoss(torch.nn.Module):
    """
    更偏向减少漏检（FN）的 Focal Tversky
    alpha: FP 权重
    beta: FN 权重
    gamma: 聚焦难样本
    """
    def __init__(self, alpha=0.25, beta=0.75, gamma=1.0, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        tp = (probs_flat * targets_flat).sum(dim=1)
        fp = ((1.0 - targets_flat) * probs_flat).sum(dim=1)
        fn = (targets_flat * (1.0 - probs_flat)).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        loss = torch.pow((1.0 - tversky), self.gamma)
        return loss.mean()


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice
        return loss.mean()


class CombinedLoss(torch.nn.Module):
    """
    总损失：
    1) Focal Tversky：重点压 FN（漏检）
    2) Dice：保持整体区域重叠稳定
    3) Weighted BCE：增强前景像素监督
    """
    def __init__(
        self,
        weight_ft=0.60,
        weight_dice=0.25,
        weight_bce=0.15,
        alpha=0.25,
        beta=0.75,
        gamma=1.0,
        pos_weight=3.0
    ):
        super().__init__()
        self.ft_loss = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.dice_loss = DiceLoss()
        self.bce_loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
        )

        self.w_ft = weight_ft
        self.w_dice = weight_dice
        self.w_bce = weight_bce

    def forward(self, logits, targets):
        device = logits.device
        if self.bce_loss.pos_weight.device != device:
            self.bce_loss.pos_weight = self.bce_loss.pos_weight.to(device)

        loss_ft = self.ft_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        loss_bce = self.bce_loss(logits, targets)

        total_loss = (
            self.w_ft * loss_ft +
            self.w_dice * loss_dice +
            self.w_bce * loss_bce
        )
        return total_loss


# ===================== 数据集 =====================
class BoneViewDataset(Dataset):
    def __init__(self, samples, img_ext=".tif", mask_ext=".tif", transform=None):
        self.samples = samples
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        img_id = sample["img_id"]
        img_path = os.path.join(sample["img_dir"], img_id + self.img_ext)
        mask_path = os.path.join(sample["mask_dir"], img_id + self.mask_ext)
        view = sample["view"]

        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask.float(), torch.tensor(view, dtype=torch.long), img_id


# ===================== 工具函数 =====================
def get_img_ids(img_dir, img_ext=".tif"):
    img_ids = []
    for f in os.listdir(img_dir):
        if f.lower().endswith(img_ext.lower()):
            img_ids.append(os.path.splitext(f)[0])
    img_ids.sort()
    return img_ids


def build_samples(img_dir, mask_dir, view, img_ext=".tif"):
    img_ids = get_img_ids(img_dir, img_ext)
    samples = []
    for img_id in img_ids:
        img_path = os.path.join(img_dir, img_id + img_ext)
        mask_path = os.path.join(mask_dir, img_id + img_ext)
        if os.path.exists(img_path) and os.path.exists(mask_path):
            samples.append({
                "img_id": img_id,
                "img_dir": img_dir,
                "mask_dir": mask_dir,
                "view": view
            })
    return samples


def batch_dice_score(logits, masks, threshold=0.4):
    """
    Dice 评估阈值改成 0.4，与推理阶段保持一致
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    dices = []
    for i in range(preds.shape[0]):
        d = dice_coef(preds[i], masks[i])
        if isinstance(d, torch.Tensor):
            d = d.item()
        if np.isnan(d) or np.isinf(d):
            d = 0.0
        dices.append(float(d))
    return float(np.mean(dices)) if dices else 0.0


# ===================== 主函数 =====================
def main():
    set_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 16
    num_workers = 0
    epochs = 200
    initial_lr = 5e-4
    patience = 30
    min_delta = 1e-4
    use_amp = torch.cuda.is_available()

    # 评估阈值，与 predictresnet.py 保持一致
    eval_threshold = 0.4

    # ===================== 路径 =====================
    train_root = r"D:\DATA\train"
    test_root = r"D:\DATA\test"

    train_anterior_img_dir = os.path.join(train_root, "01_image_anterior")
    train_anterior_mask_dir = os.path.join(train_root, "01_image_anterior_labels")
    train_posterior_img_dir = os.path.join(train_root, "02_image_all_posterior")
    train_posterior_mask_dir = os.path.join(train_root, "02_image_all_posterior_labels")

    val_anterior_img_dir = os.path.join(test_root, "01_image_anterior")
    val_anterior_mask_dir = os.path.join(test_root, "01_image_anterior_labels")
    val_posterior_img_dir = os.path.join(test_root, "02_image_all_posterior")
    val_posterior_mask_dir = os.path.join(test_root, "02_image_all_posterior_labels")

    # ===================== 数据增强 =====================
    data_transform = {
        "train": A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT),
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.15,
                rotate_limit=12,
                p=0.4,
                border_mode=cv2.BORDER_REFLECT
            ),
            A.Normalize(mean=[0.485], std=[0.229], max_pixel_value=255.0),
            ToTensorV2()
        ]),
        "val": A.Compose([
            A.Normalize(mean=[0.485], std=[0.229], max_pixel_value=255.0),
            ToTensorV2()
        ])
    }

    # ===================== 样本构建 =====================
    train_samples = []
    train_samples += build_samples(train_anterior_img_dir, train_anterior_mask_dir, view=0)
    train_samples += build_samples(train_posterior_img_dir, train_posterior_mask_dir, view=1)

    val_samples = []
    val_samples += build_samples(val_anterior_img_dir, val_anterior_mask_dir, view=0)
    val_samples += build_samples(val_posterior_img_dir, val_posterior_mask_dir, view=1)

    train_dataset = BoneViewDataset(train_samples, transform=data_transform["train"])
    val_dataset = BoneViewDataset(val_samples, transform=data_transform["val"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Eval threshold: {eval_threshold}")

    # ===================== 网络 =====================
    net = MK_UNet_ViewBranch(
        num_classes=1,
        in_channels=3,
        view_embed_dim=32,
        view_scale=0.1
    ).to(device)

    # ===================== 损失函数 =====================
    loss_function = CombinedLoss(
        weight_ft=0.60,
        weight_dice=0.25,
        weight_bce=0.15,
        alpha=0.25,
        beta=0.75,
        gamma=1.0,
        pos_weight=3.0
    ).to(device)

    # ===================== 优化器 =====================
    optimizer = optim.AdamW(
        net.parameters(),
        lr=initial_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-3
    )

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    scaler = GradScaler(enabled=use_amp)

    save_path = r".\viewbranch_best.pth"
    save_last_path = r".\viewbranch_last.pth"
    log_path = r".\viewbranch_results.txt"

    best_dice = 0.0
    patience_counter = 0

    train_loss_list = []
    val_loss_list = []
    train_dice_list = []
    val_dice_list = []

    # ===================== 训练 =====================
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        running_dice = 0.0
        train_steps = 0

        train_bar = tqdm(train_loader, file=sys.stdout, desc=f"Train Epoch [{epoch+1}/{epochs}]")
        for images, masks, views, _ in train_bar:
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            masks = masks.to(device, dtype=torch.float32, non_blocking=True)
            views = views.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                outputs = net(images, views)
                loss = loss_function(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_dice += batch_dice_score(outputs.detach(), masks, threshold=eval_threshold)
            train_steps += 1

            train_bar.set_description(
                f"Train Epoch [{epoch+1}/{epochs}] "
                f"loss:{loss.item():.4f}"
            )

        train_loss = running_loss / train_steps if train_steps > 0 else 0.0
        train_dice = running_dice / train_steps if train_steps > 0 else 0.0

        train_loss_list.append(train_loss)
        train_dice_list.append(train_dice)

        # ===================== 验证 =====================
        net.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_steps = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout, desc=f"Val Epoch [{epoch+1}/{epochs}]")
            for images, masks, views, _ in val_bar:
                images = images.to(device, dtype=torch.float32, non_blocking=True)
                masks = masks.to(device, dtype=torch.float32, non_blocking=True)
                views = views.to(device, non_blocking=True)

                with autocast(enabled=use_amp):
                    outputs = net(images, views)
                    loss = loss_function(outputs, masks)

                val_loss += loss.item()
                val_dice += batch_dice_score(outputs, masks, threshold=eval_threshold)
                val_steps += 1

        val_loss = val_loss / val_steps if val_steps > 0 else 0.0
        val_dice = val_dice / val_steps if val_steps > 0 else 0.0

        val_loss_list.append(val_loss)
        val_dice_list.append(val_dice)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}: "
            f"lr={current_lr:.6f} | "
            f"Train Loss={train_loss:.4f}, Train Dice={train_dice:.4f} | "
            f"Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}"
        )

        # 保存最优模型
        if val_dice > best_dice + min_delta:
            best_dice = val_dice
            torch.save(net.state_dict(), save_path)
            patience_counter = 0
            print(f"Saved best model with val dice: {best_dice:.4f}")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")

        torch.save(net.state_dict(), save_last_path)

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

        scheduler.step()

    # ===================== 保存训练日志 =====================
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Epoch\tTrainLoss\tValLoss\tTrainDice\tValDice\n")
        for i in range(len(train_loss_list)):
            f.write(
                f"{i+1}\t"
                f"{train_loss_list[i]:.6f}\t"
                f"{val_loss_list[i]:.6f}\t"
                f"{train_dice_list[i]:.6f}\t"
                f"{val_dice_list[i]:.6f}\n"
            )

    print("Training finished.")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Best model saved to: {save_path}")
    print(f"Last model saved to: {save_last_path}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    main()