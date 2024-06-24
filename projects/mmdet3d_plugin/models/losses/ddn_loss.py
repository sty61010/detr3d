from typing import List
import torch
import torch.nn as nn

from mmdet3d.models.builder import LOSSES
from .balancer import Balancer
from .focalloss import FocalLoss
# based on:
# https://github.com/TRAILab/CaDDN/blob/master/pcdet/models/backbones_3d/ffe/ddn_loss/ddn_loss.py


@LOSSES.register_module()
class DDNLoss(nn.Module):
    """Depth Distribution Network Loss from https://arxiv.org/abs/2103.01100.
    Args:
        alpha [float]: Alpha value for Focal Loss
        gamma [float]: Gamma value for Focal Loss
        fg_weight [float]: Foreground loss weight
        bg_weight [float]: Background loss weight
        downsample_factor [int]: Depth map downsample factor
        loss_weight [float]: weight for loss_ddn
    """

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 fg_weight: float = 13,
                 bg_weight: float = 1,
                 downsample_factor: int = 1,
                 loss_weight: float = 1.0,
                 ):

        super().__init__()
        self.balancer = Balancer(
            downsample_factor=downsample_factor,
            fg_weight=fg_weight,
            bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        self.loss_func = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")
        self.loss_weight = loss_weight

    def forward(self,
                depth_logits: torch.Tensor,
                depth_target: torch.Tensor,
                gt_bboxes_2d: List[List[torch.Tensor]]) -> torch.Tensor:
        """Gets depth_map loss
        Args:
            depth_logits(torch.Tensor): [B, N, D+1, H, W]: Predicted depth logits
            depth_target(torch.Tensor): [B, N, H, W]
                Thr ground truth depth maps with shape
                [batch, num_cameras, depth_map_H, depth_map_W]
            gt_bboxes_2d: A list of list of tensor containing 2D ground truth bboxes(x, y, w, h)
                for each sample and each camera. Each tensor has shape [N_i, 4].
                Below is the brief explanation of a single batch:
                [B, N, ....]
                [[gt_bboxes_tensor_camera_0, gt_bboxes_tensor_camera_1, ..., gt_bboxes_tensor_camera_n](sample 0),
                    [gt_bboxes_tensor_camera_0, gt_bboxes_tensor_camera_1, ..., gt_bboxes_tensor_camera_n](sample 1),
                    ...].
        Returns:
            loss [torch.Tensor(1)]: Depth classification network loss
                [B*N, H, W]
        """
        B, N, D, H, W = depth_logits.shape
        # depth_logits(torch.Tensor): [B, N, D+1, H, W] -> [B*N, D+1, H, W]
        depth_logits = depth_logits.flatten(0, 1)
        # depth_target(torch.Tensor): [B, N, H, W] -> [B*N, H, W]
        depth_target = depth_target.flatten(0, 1)
        # Compute loss
        loss = self.loss_func(depth_logits, depth_target)
        # loss = loss.view(B, N, H, W)
        # print(f'loss: {loss.shape}')
        # Compute foreground/background balancing
        loss = self.balancer(loss=loss, gt_boxes2d=gt_bboxes_2d)

        loss = self.loss_weight * loss

        return loss
