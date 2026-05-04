import os
import cv2
import numpy as np
import torch
import torch.utils.data
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image  # 核心修复：新增PIL.Image导入


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """单类别分割数据集（适配torchvision.transforms）"""
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes  # 单类别设为1
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # 1. 读取图片（PIL格式，兼容torchvision）
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"图片不存在：{img_path}")
        # 自动判断并处理通道：灰度图→转3通道RGB；彩色图→BGR转RGB
        if len(img.shape) == 2:  # 灰度图（H,W）
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 转为3通道RGB（适配模型输入）
        else:  # 彩色图（H,W,3）
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB
        # 2. 读取单类别掩码
        mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"掩码不存在：{mask_path}")

        # 3. 应用torchvision变换（仅对图片，掩码手动同步尺寸）
        if self.transform is not None:
            augmented = self.transform(image=img,mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        mask = mask.unsqueeze(0).float() / 255.0

        return img, mask, img_id