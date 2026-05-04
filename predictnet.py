import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import sys
import json
import warnings
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from mkunet_network import MK_UNet_ViewBranch
import metrics

warnings.filterwarnings("ignore")


def get_img_ids(img_dir, img_ext=".tif"):
    if not os.path.exists(img_dir):
        return []
    img_ids = []
    for file in os.listdir(img_dir):
        if file.lower().endswith(img_ext.lower()):
            img_ids.append(os.path.splitext(file)[0])
    img_ids.sort()
    return img_ids


def safe_float(val):
    if isinstance(val, torch.Tensor):
        val = val.item()
    try:
        val = float(val)
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return val
    except Exception:
        return 0.0


def save_mask(pred_mask, img_id, view, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    mask_np = pred_mask.squeeze().detach().cpu().numpy()
    mask_np = (mask_np * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_np)
    save_path = os.path.join(save_dir, f"{img_id}_view{view}_mask.tif")
    mask_img.save(save_path)


class BoneViewTestDataset(Dataset):
    """
    samples:
        {
            "img_id": xxx,
            "img_dir": xxx,
            "mask_dir": xxx or "",
            "view": 0 or 1
        }
    """
    def __init__(self, samples, img_ext=".tif", mask_ext=".tif", transform=None, has_mask=True):
        self.samples = samples
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform
        self.has_mask = has_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        img_id = sample["img_id"]
        img_path = os.path.join(sample["img_dir"], img_id + self.img_ext)
        view = sample["view"]

        image = np.array(Image.open(img_path).convert("L"))

        if self.has_mask:
            mask_path = os.path.join(sample["mask_dir"], img_id + self.mask_ext)
            mask = np.array(Image.open(mask_path).convert("L"))
            mask = (mask > 0).astype(np.float32)

            if self.transform is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            return image, mask.float(), torch.tensor(view, dtype=torch.long), img_id
        else:
            if self.transform is not None:
                augmented = self.transform(image=image)
                image = augmented["image"]

            return image, torch.tensor(view, dtype=torch.long), img_id


def build_samples(img_dir, mask_dir, view, img_ext=".tif", has_mask=True):
    samples = []
    img_ids = get_img_ids(img_dir, img_ext)

    for img_id in img_ids:
        img_path = os.path.join(img_dir, img_id + img_ext)
        if not os.path.exists(img_path):
            continue

        if has_mask:
            mask_path = os.path.join(mask_dir, img_id + img_ext)
            if not os.path.exists(mask_path):
                continue

        samples.append({
            "img_id": img_id,
            "img_dir": img_dir,
            "mask_dir": mask_dir,
            "view": view
        })

    return samples


def main():
    # ===================== 1. 基础配置 =====================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device for testing")

    num_classes = 1
    img_ext = ".tif"
    mask_ext = ".tif"
    batch_size = 1
    num_workers = 0

    # 小病灶漏检时建议使用 0.40
    threshold = 0.40

    # ===================== 2. 路径配置 =====================
    test_root = r"D:\chenwu\segmentation\segcode_zyf_transformer_change4_NEW\test\20260409"

    test_anterior_img_dir = os.path.join(test_root, "01_image_anterior")
    test_anterior_mask_dir = os.path.join(test_root, "01_image_anterior_labels")
    test_posterior_img_dir = os.path.join(test_root, "02_image_all_posterior")
    test_posterior_mask_dir = os.path.join(test_root, "02_image_all_posterior_labels")

    model_path = r".\mkunet_viewbranch_change4_best.pth"

    pred_mask_save_dir = os.path.join(
        test_root,
        f"pred_viewbranch_masks_thr{str(threshold).replace('.', '')}"
    )

    # ===================== 3. 路径检查 =====================
    if not os.path.exists(test_anterior_img_dir):
        print(f"错误：前位测试图像路径不存在 - {test_anterior_img_dir}")
        return

    if not os.path.exists(test_posterior_img_dir):
        print(f"警告：后位测试图像路径不存在 - {test_posterior_img_dir}")
        print("将仅使用前位图像进行预测")
        posterior_exists = False
    else:
        posterior_exists = True

    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 - {model_path}")
        return

    anterior_has_mask = os.path.exists(test_anterior_mask_dir)
    posterior_has_mask = posterior_exists and os.path.exists(test_posterior_mask_dir)

    anterior_img_num = len(get_img_ids(test_anterior_img_dir, img_ext))
    posterior_img_num = len(get_img_ids(test_posterior_img_dir, img_ext)) if posterior_exists else 0

    print(f"Threshold: {threshold}")
    print(f"Anterior images: {anterior_img_num}")
    print(f"Posterior images: {posterior_img_num}")

    # 只有存在图像的视角都具备标签时，才计算指标
    if anterior_img_num > 0 and posterior_img_num > 0:
        calculate_metrics = anterior_has_mask and posterior_has_mask
    elif anterior_img_num > 0 and posterior_img_num == 0:
        calculate_metrics = anterior_has_mask
    else:
        calculate_metrics = False

    if calculate_metrics:
        print("测试标签路径存在，将计算分割指标")
    else:
        print("测试标签路径不完整或不存在，仅进行预测并保存mask")

    # ===================== 4. 数据预处理 =====================
    test_transform = A.Compose([
        A.Normalize(mean=[0.485], std=[0.229], max_pixel_value=255.0),
        ToTensorV2()
    ])

    # ===================== 5. 构建测试样本 =====================
    test_samples = []

    if anterior_img_num > 0:
        test_samples += build_samples(
            test_anterior_img_dir,
            test_anterior_mask_dir,
            view=0,
            img_ext=img_ext,
            has_mask=calculate_metrics
        )

    if posterior_exists and posterior_img_num > 0:
        test_samples += build_samples(
            test_posterior_img_dir,
            test_posterior_mask_dir,
            view=1,
            img_ext=img_ext,
            has_mask=calculate_metrics
        )

    if len(test_samples) == 0 and not calculate_metrics:
        anterior_ids = get_img_ids(test_anterior_img_dir, img_ext)
        for img_id in anterior_ids:
            test_samples.append({
                "img_id": img_id,
                "img_dir": test_anterior_img_dir,
                "mask_dir": "",
                "view": 0
            })

        if posterior_exists:
            posterior_ids = get_img_ids(test_posterior_img_dir, img_ext)
            for img_id in posterior_ids:
                test_samples.append({
                    "img_id": img_id,
                    "img_dir": test_posterior_img_dir,
                    "mask_dir": "",
                    "view": 1
                })

    if len(test_samples) == 0:
        print("错误：未找到任何测试样本")
        return

    test_dataset = BoneViewTestDataset(
        samples=test_samples,
        img_ext=img_ext,
        mask_ext=mask_ext,
        transform=test_transform,
        has_mask=calculate_metrics
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_num = len(test_dataset)
    print(f"Total test samples: {test_num}")

    # ===================== 6. 加载模型 =====================
    net = MK_UNet_ViewBranch(
        num_classes=num_classes,
        in_channels=3,
        view_embed_dim=32,
        view_scale=0.1
    ).to(device)

    try:
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict)
        net.eval()
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"错误：加载模型失败 - {e}")
        return

    # ===================== 7. 推理与指标计算 =====================
    dice_list, iou_list = [], []
    precision_list, recall_list = [], []
    sensitivity_list, specificity_list = [], []

    per_sample_results = []

    test_bar = tqdm(test_loader, file=sys.stdout, desc="Testing")

    with torch.no_grad():
        for step, data in enumerate(test_bar):
            if calculate_metrics:
                images, masks, views, img_ids = data
                images = images.to(device, dtype=torch.float32, non_blocking=True)
                masks = masks.to(device, dtype=torch.float32, non_blocking=True)
                masks = (masks > 0.5).float()
                views = views.to(device, non_blocking=True)

                outputs = net(images, views)
                preds = torch.sigmoid(outputs)
                preds_binary = (preds > threshold).float()

                for i in range(images.shape[0]):
                    img_id = img_ids[i]
                    view_i = int(views[i].item())

                    save_mask(preds_binary[i], img_id, view_i, pred_mask_save_dir)

                    dice = safe_float(metrics.dice_coef(preds_binary[i], masks[i]))
                    iou_val = safe_float(metrics.iou_score(preds_binary[i], masks[i]))
                    prec = safe_float(metrics.precision(preds_binary[i], masks[i]))
                    rec = safe_float(metrics.recall(preds_binary[i], masks[i]))
                    sens = safe_float(metrics.sensitivity(preds_binary[i], masks[i]))
                    spec = safe_float(metrics.specificity(preds_binary[i], masks[i]))

                    dice_list.append(dice)
                    iou_list.append(iou_val)
                    precision_list.append(prec)
                    recall_list.append(rec)
                    sensitivity_list.append(sens)
                    specificity_list.append(spec)

                    per_sample_results.append({
                        "img_id": img_id,
                        "view": view_i,
                        "dice": dice,
                        "iou": iou_val,
                        "precision": prec,
                        "recall": rec,
                        "sensitivity": sens,
                        "specificity": spec
                    })

                if len(dice_list) > 0:
                    test_bar.set_description(
                        f"Testing [{step + 1}/{len(test_loader)}] "
                        f"Dice:{dice_list[-1]:.4f} IoU:{iou_list[-1]:.4f}"
                    )

            else:
                images, views, img_ids = data
                images = images.to(device, dtype=torch.float32, non_blocking=True)
                views = views.to(device, non_blocking=True)

                outputs = net(images, views)
                preds = torch.sigmoid(outputs)
                preds_binary = (preds > threshold).float()

                for i in range(images.shape[0]):
                    img_id = img_ids[i]
                    view_i = int(views[i].item())

                    save_mask(preds_binary[i], img_id, view_i, pred_mask_save_dir)

                    per_sample_results.append({
                        "img_id": img_id,
                        "view": view_i
                    })

                test_bar.set_description(
                    f"Testing [{step + 1}/{len(test_loader)}] Saved {img_ids[0]}"
                )

    # ===================== 8. 输出结果 =====================
    if calculate_metrics and len(dice_list) > 0:
        avg_dice = float(np.nanmean(dice_list))
        avg_iou = float(np.nanmean(iou_list))
        avg_prec = float(np.nanmean(precision_list))
        avg_rec = float(np.nanmean(recall_list))
        avg_sens = float(np.nanmean(sensitivity_list))
        avg_spec = float(np.nanmean(specificity_list))

        print("\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)
        print(f"Threshold:                   {threshold:.2f}")
        print(f"Average Dice Coefficient:    {avg_dice:.4f}")
        print(f"Average IoU (Jaccard):       {avg_iou:.4f}")
        print(f"Average Precision:           {avg_prec:.4f}")
        print(f"Average Recall:              {avg_rec:.4f}")
        print(f"Average Sensitivity:         {avg_sens:.4f}")
        print(f"Average Specificity:         {avg_spec:.4f}")
        print("=" * 60)

        results = {
            "model_info": {
                "model_path": model_path,
                "backbone": "MK_UNet_ViewBranch",
                "num_classes": num_classes,
                "view_embed_dim": 32,
                "view_scale": 0.1,
                "threshold": threshold
            },
            "data_info": {
                "test_root": test_root,
                "test_samples": test_num,
                "anterior_images": anterior_img_num,
                "posterior_images": posterior_img_num,
                "pred_mask_save_dir": pred_mask_save_dir
            },
            "average_metrics": {
                "dice": avg_dice,
                "iou": avg_iou,
                "precision": avg_prec,
                "recall": avg_rec,
                "sensitivity": avg_sens,
                "specificity": avg_spec
            },
            "per_sample_metrics": per_sample_results
        }
    else:
        results = {
            "model_info": {
                "model_path": model_path,
                "backbone": "MK_UNet_ViewBranch",
                "num_classes": num_classes,
                "view_embed_dim": 32,
                "view_scale": 0.1,
                "threshold": threshold
            },
            "data_info": {
                "test_root": test_root,
                "test_samples": test_num,
                "anterior_images": anterior_img_num,
                "posterior_images": posterior_img_num,
                "pred_mask_save_dir": pred_mask_save_dir
            },
            "message": "No ground truth masks, only prediction masks saved",
            "per_sample_results": per_sample_results
        }

        print("\n" + "=" * 60)
        print("Only prediction masks saved (no ground truth for metric calculation)")
        print("=" * 60)

    # ===================== 9. 保存JSON结果 =====================
    save_json_path = f"mkunet_viewbranch_change3_test_metrics_thr{str(threshold).replace('.', '')}.json"
    try:
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\n测试结果已保存到 {save_json_path}")
    except Exception as e:
        print(f"警告：保存指标文件失败 - {e}")

    print(f"\nAll predicted masks saved to {pred_mask_save_dir}")


if __name__ == "__main__":
    main()