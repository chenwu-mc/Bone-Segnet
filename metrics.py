import numpy as np
import torch
import cv2
from sklearn.metrics import roc_auc_score


# ===================== 6种核心分割指标（移除内部sigmoid，主程序已处理） =====================
def iou_score(output, target, smooth=1e-7):
    """计算IoU交并比（兼容张量/NumPy，主程序已做sigmoid，仅内部二值化）"""
    # 张量转NumPy
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    # 仅二值化，不做sigmoid
    output_ = output > 0.5
    target_ = target > 0.5

    TP = (output_ * target_).sum()
    FP = (output_ * ~target_).sum()
    FN = (~output_ * target_).sum()
    return (TP + smooth) / (TP + FN + FP + smooth)


def dice_coef(output, target, smooth=1e-7):
    """计算Dice系数（兼容张量/NumPy，主程序已做sigmoid，仅内部二值化）"""
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    intersection = (output_ * target_).sum()
    return (2. * intersection + smooth) / (output_.sum() + target_.sum() + smooth)


def precision(output, target, smooth=1e-7):
    """精确率 TP/(TP+FP)（主程序已做sigmoid）"""
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    TP = (output_ * target_).sum()
    FP = (output_ * ~target_).sum()
    return (TP + smooth) / (TP + FP + smooth)


def recall(output, target, smooth=1e-7):
    """召回率 TP/(TP+FN)（主程序已做sigmoid）"""
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    TP = (output_ * target_).sum()
    FN = (~output_ * target_).sum()
    return (TP + smooth) / (TP + FN + smooth)


sensitivity = recall  # 敏感性=召回率，别名


def specificity(output, target, smooth=1e-7):
    """特异度 TN/(TN+FP)（主程序已做sigmoid）"""
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    TN = (~output_ * ~target_).sum()
    FP = (output_ * ~target_).sum()
    return (TN + smooth) / (TN + FP + smooth)


# ===================== 实例级Dice（保留，适配修复后逻辑） =====================
def instance_dice_coef(pred_mask, gt_mask):
    """实例级Dice系数（对每个小病灶单独计算后平均）"""
    smooth = 1e-6
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.data.cpu().numpy()
    if torch.is_tensor(gt_mask):
        gt_mask = gt_mask.data.cpu().numpy()

    # 压缩维度为单通道
    if len(pred_mask.shape) > 2:
        pred_mask = pred_mask[0] if pred_mask.shape[0] in [1, 3] else cv2.cvtColor(pred_mask, cv2.COLOR_RGB2GRAY)
    if len(gt_mask.shape) > 2:
        gt_mask = gt_mask[0] if gt_mask.shape[0] in [1, 3] else cv2.cvtColor(gt_mask, cv2.COLOR_RGB2GRAY)

    # 二值化（统一0/255）
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    gt_mask = (gt_mask > 0.5).astype(np.uint8) * 255

    # 提取真实病灶轮廓
    gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(gt_contours) == 0:
        return 1.0 if np.sum(pred_mask) == 0 else 0.0

    dice_list = []
    for contour in gt_contours:
        x, y, w, h = cv2.boundingRect(contour)
        y1, y2 = max(0, y), min(gt_mask.shape[0], y + h)
        x1, x2 = max(0, x), min(gt_mask.shape[1], x + w)
        if y2 <= y1 or x2 <= x1:
            continue

        # 提取单病灶并二值化
        gt_instance = np.zeros_like(gt_mask)
        cv2.drawContours(gt_instance, [contour], 0, 255, -1)
        gt_bin = (gt_instance[y1:y2, x1:x2] > 127).astype(np.float32)
        pred_bin = (pred_mask[y1:y2, x1:x2] > 127).astype(np.float32)

        # 计算单实例Dice
        intersection = np.sum(pred_bin * gt_bin)
        dice = (2 * intersection + smooth) / (np.sum(pred_bin) + np.sum(gt_bin) + smooth)
        dice_list.append(dice)

    return np.mean(dice_list) if dice_list else 0.0