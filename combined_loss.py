from monai.losses import DiceLoss, FocalLoss
import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, weights=None, dice_weight=0.5, focal_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(
            include_background=True,
            to_onehot_y=False,
            softmax=True,
            squared_pred=False,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )
        self.focal_loss = FocalLoss(
            include_background=True, to_onehot_y=False, gamma=2.0
        )
        self.weights = weights
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, y_pred, y_true):
        dice_loss = self.dice_loss(y_pred, y_true)
        focal_loss = self.focal_loss(y_pred, y_true)

        if self.weights is not None:
            weights = self.weights.to(dice_loss.device)
            dice_loss = dice_loss * weights
            focal_loss = focal_loss * weights

        total_loss = (
            self.dice_weight * dice_loss.mean() + self.focal_weight * focal_loss.mean()
        )
        return total_loss


class CombinedLossDCE(nn.Module):
    """
    Weighted Dice + Weighted Cross Entropy loss.
    """

    def __init__(self, dice_weight=0.5, ce_weight=0.5, class_weights=None):
        super(CombinedLossDCE, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        self.dice_loss = DiceLoss(
            include_background=True,
            to_onehot_y=False,   # targets already one-hot from the train loop
            softmax=True,
            squared_pred=False,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )

        # Store class weights as a buffer so .to(device) works automatically
        if class_weights is not None:
            self.register_buffer(
                "class_weights", torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None

    def forward(self, y_pred, y_true):
        # Dice component — receives one-hot targets directly
        dice = self.dice_loss(y_pred, y_true)

        # CE component — nn.CrossEntropyLoss expects class-index targets (B, H, W)
        y_true_idx = torch.argmax(y_true, dim=1).long()
        ce_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(y_pred.device) if self.class_weights is not None else None
        )
        ce = ce_fn(y_pred, y_true_idx)

        return self.dice_weight * dice + self.ce_weight * ce