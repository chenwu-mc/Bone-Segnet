import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# 纯 PyTorch 实现 Lovasz Hinge Loss（无需外部库）
# ============================================

def lovasz_grad(gt_sorted):
    """计算 Lovasz 扩展的梯度 w.r.t sorted errors"""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # 处理 1-pixel 情况
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    二进制 Lovasz hinge loss（展平版本）
      logits: [P] Variable, 预测logits (between -infty and +infty)
      labels: [P] Tensor, 二进制真实标签 (0 or 1)
    """
    if len(labels) == 0:
        return logits.sum() * 0.

    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    展平预测在batch中（二进制情况）
    移除等于'ignore'的标签
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    二进制 Lovasz hinge loss
      logits: [B, H, W] 或 [B, 1, H, W] Variable, 每个像素的logits
      labels: [B, H, W] 或 [B, 1, H, W] Tensor, 二进制真实掩码 (0 or 1)
      per_image: 是否每张图像单独计算，而不是整个batch
      ignore: 忽略的类别id（void class）
    """
    # 处理可能的4维输入 [B, 1, H, W] -> [B, H, WeightedLovaszLoss]
    if logits.dim() == 4:
        logits = logits.squeeze(1)
    if labels.dim() == 4:
        labels = labels.squeeze(1)

    if per_image:
        losses = []
        for log, lab in zip(logits, labels):
            loss = lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
            losses.append(loss)
        losses = torch.stack(losses)
        return losses.mean()
    else:
        return lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))


# ============================================
# 原有损失函数定义（保持不变）
# ============================================

__all__ = [
    'BCEDiceLoss', 'LovaszHingeLoss',
    'WeightedBCEDiceLoss', 'FocalDiceLoss', 'WeightedLovaszLoss',
    'CombinedSmallObjectLoss'
]


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


class WeightedBCEDiceLoss(nn.Module):
    """
    加权BCEDiceLoss - 为小目标像素分配更高权重，提升小目标梯度贡献
    适用于单类别小目标分割，通过pos_weight放大目标像素的BCE损失权重
    """

    def __init__(self, pos_weight=10.0, smooth=1e-5):
        super().__init__()
        self.pos_weight = pos_weight  # 目标像素权重（小目标建议设为5-20）
        self.smooth = smooth

    def forward(self, input, target):
        # 1. 加权BCE损失：对目标像素赋予更高权重
        # 创建权重矩阵：目标区域=pos_weight，背景区域=1
        weight = torch.ones_like(target)
        weight[target == 1] = self.pos_weight

        # 计算加权BCE loss
        bce = F.binary_cross_entropy_with_logits(input, target, weight=weight)

        # 2. Dice损失（保持原有逻辑，保证对重叠区域的关注）
        input_sigmoid = torch.sigmoid(input)
        num = target.size(0)
        input_flat = input_sigmoid.view(num, -1)
        target_flat = target.view(num, -1)

        intersection = (input_flat * target_flat).sum(1)
        dice = (2. * intersection + self.smooth) / (input_flat.sum(1) + target_flat.sum(1) + self.smooth)
        dice_loss = 1 - dice.sum() / num

        # 3. 混合损失（平衡BCE和Dice）
        return 0.5 * bce + dice_loss


class FocalDiceLoss(nn.Module):
    """
    FocalDiceLoss - 抑制背景易分类样本的梯度，聚焦小目标难分类区域
    gamma越大，对易分类样本的抑制越强（小目标建议gamma=2）
    """

    def __init__(self, gamma=2.0, alpha=1.0, smooth=1e-5):
        super().__init__()
        self.gamma = gamma  # 聚焦系数，小目标推荐2.0
        self.alpha = alpha  # 平衡系数，可微调
        self.smooth = smooth

    def forward(self, input, target):
        # 1. Focal Loss计算
        input_sigmoid = torch.sigmoid(input)
        # 计算pt：预测为正/负的概率
        pt = input_sigmoid * target + (1 - input_sigmoid) * (1 - target)
        # 聚焦权重：难分类样本（pt小）权重高，易分类样本（pt大）权重低
        focal_weight = (1 - pt) ** self.gamma
        # 加权BCE loss
        bce = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        focal_loss = (self.alpha * focal_weight * bce).mean()

        # 2. Dice损失（保证对小目标重叠区域的关注）
        num = target.size(0)
        input_flat = input_sigmoid.view(num, -1)
        target_flat = target.view(num, -1)

        intersection = (input_flat * target_flat).sum(1)
        dice = (2. * intersection + self.smooth) / (input_flat.sum(1) + target_flat.sum(1) + self.smooth)
        dice_loss = 1 - dice.sum() / num

        # 3. 混合损失
        return 0.5 * focal_loss + dice_loss


class WeightedLovaszLoss(nn.Module):
    """
    加权LovaszHingeLoss - 适配小目标分割的Lovasz损失
    Lovasz损失对类别不平衡鲁棒，加权后进一步提升小目标关注度
    """

    def __init__(self, pos_weight=5.0):
        super().__init__()
        self.pos_weight = pos_weight  # 小目标权重，建议5-10

    def forward(self, input, target):
        # 调整维度（适配lovasz_hinge输入要求）
        input = input.squeeze(1)
        target = target.squeeze(1)

        # 计算加权Lovasz损失
        # 方式1：直接加权（简单有效）
        loss = lovasz_hinge(input, target, per_image=True)

        # 方式2：对目标区域的误差赋予更高权重（进阶版，需确保lovasz_hinge支持weight参数）
        # weight = torch.ones_like(target)
        # weight[target == 1] = self.pos_weight
        # loss = lovasz_hinge(input, target, per_image=True, weight=weight)

        return self.pos_weight * loss if loss > 0.1 else loss


class CombinedSmallObjectLoss(nn.Module):
    """
    小目标分割综合损失：融合 BCE-Dice + Focal-Dice + Lovasz
    无需改造原有损失类，直接实例化传入即可
    """

    def __init__(self,
                 bce_dice_weight=0.4,
                 focal_dice_weight=0.4,
                 lovasz_weight=0.2,
                 pos_weight=8.0,  # WeightedBCEDiceLoss用
                 gamma=2.0,  # FocalDiceLoss用
                 lovasz_pos_weight=5.0):  # WeightedLovaszLoss用

        super().__init__()

        # 实例化你已有的损失函数（保持原样，不做任何修改）
        self.bce_dice = WeightedBCEDiceLoss(pos_weight=pos_weight)
        self.focal_dice = FocalDiceLoss(gamma=gamma, alpha=1.0)

        # Lovasz检查：现在一定可用，因为我们已经内置了实现
        self.lovasz = WeightedLovaszLoss(pos_weight=lovasz_pos_weight)
        self.use_lovasz = True

        self.weights = {
            'bce_dice': bce_dice_weight,
            'focal_dice': focal_dice_weight,
            'lovasz': lovasz_weight
        }

        # 监控用
        self.loss_history = {'bce_dice': [], 'focal_dice': [], 'lovasz': []}

    def forward(self, pred, target):
        # 计算各分量
        loss_bd = self.bce_dice(pred, target)
        loss_fd = self.focal_dice(pred, target)

        # 加权求和
        total_loss = self.weights['bce_dice'] * loss_bd + \
                     self.weights['focal_dice'] * loss_fd

        # Lovasz计算
        if self.use_lovasz and self.weights['lovasz'] > 0:
            loss_lov = self.lovasz(pred, target)
            total_loss += self.weights['lovasz'] * loss_lov
            self.loss_history['lovasz'].append(loss_lov.item())

        # 记录用于分析
        self.loss_history['bce_dice'].append(loss_bd.item())
        self.loss_history['focal_dice'].append(loss_fd.item())

        return total_loss

    def get_metrics(self):
        """返回各损失分量的近期平均值，用于tensorboard记录"""
        return {k: sum(v[-20:]) / len(v[-20:]) if v else 0
                for k, v in self.loss_history.items()}