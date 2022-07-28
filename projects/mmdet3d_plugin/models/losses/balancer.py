

import torch
import torch.nn as nn

# based on
# https://github.com/TRAILab/CaDDN/blob/master/pcdet/models/backbones_3d/ffe/ddn_loss/balancer.py


class Balancer(nn.Module):
    def __init__(self, fg_weight, bg_weight, downsample_factor=1):
        """
        Initialize fixed foreground/background loss balancer
        Args:
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.downsample_factor = downsample_factor

    def forward(self, loss, gt_boxes2d):
        """
        Forward pass
        Args:
            loss [torch.Tensor(BN, H, W)]: Pixel-wise loss
                [B*N, H, W]
            gt_bboxes_2d: A list of list of tensor containing 2D ground truth bboxes(x, y, w, h)
                for each sample and each camera. Each tensor has shape [N_i, 4].
                Below is the brief explanation of a single batch:
                [B, N, ....]
                [[gt_bboxes_tensor_camera_0, gt_bboxes_tensor_camera_1, ..., gt_bboxes_tensor_camera_n](sample 0),
                    [gt_bboxes_tensor_camera_0, gt_bboxes_tensor_camera_1, ..., gt_bboxes_tensor_camera_n](sample 1),
                    ...]
        Returns:
            loss [torch.Tensor(1)]: Total loss after foreground/background balancing
            tb_dict [dict[float]]: All losses to log in tensorboard
        """
        # Compute masks
        fg_mask = compute_fg_mask(gt_boxes2d=gt_boxes2d,
                                  shape=loss.shape,
                                  downsample_factor=self.downsample_factor,
                                  device=loss.device)
        bg_mask = ~fg_mask

        # Compute balancing weights
        weights = self.fg_weight * fg_mask + self.bg_weight * bg_mask
        num_pixels = fg_mask.sum() + bg_mask.sum()

        # Compute losses
        loss *= weights
        fg_loss = loss[fg_mask].sum() / num_pixels
        bg_loss = loss[bg_mask].sum() / num_pixels

        # Get total loss
        loss = fg_loss + bg_loss
        return loss


def compute_fg_mask(gt_boxes2d, shape, downsample_factor=1, device=torch.device("cpu")):
    """
    Compute foreground mask for images
    Args:
        gt_bboxes_2d: A list of list of tensor containing 2D ground truth bboxes(x, y, w, h)
            for each sample and each camera. Each tensor has shape [N_i, 4].
            Below is the brief explanation of a single batch:
            [B, N, ....]
            [[gt_bboxes_tensor_camera_0, gt_bboxes_tensor_camera_1, ..., gt_bboxes_tensor_camera_n](sample 0),
                [gt_bboxes_tensor_camera_0, gt_bboxes_tensor_camera_1, ..., gt_bboxes_tensor_camera_n](sample 1),
                ...]
        shape [torch.Size or tuple]: Foreground mask desired shape of
            loss [torch.Tensor(BN, H, W)]: Pixel-wise loss [B*N, H, W]
        downsample_factor [int]: Downsample factor for image
        device [torch.device]: Foreground mask desired device
    Returns:
        fg_mask [torch.Tensor(shape)]: Foreground mask
    """
    fg_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    # print(f'fg_mask: {fg_mask.shape}')

    # # Set box corners
    # gt_boxes2d /= downsample_factor
    # gt_boxes2d[:, :2] = torch.floor(gt_boxes2d[:, :2])
    # gt_boxes2d[:, 2:] = torch.ceil(gt_boxes2d[:, 2:])
    # gt_boxes2d = gt_boxes2d.long()

    # Set all values within each box to True
    # gt_boxes2d = gt_boxes2d.split(num_gt_per_img, dim=0)

    B, N = len(gt_boxes2d), len(gt_boxes2d[0])
    # iterate batch size
    for b in range(B):
        # iterate num_cameras in per sample
        for n in range(N):
            # iterate num_objects in per camera
            for i in range(gt_boxes2d[b][n].shape[0]):
                # x, y, w, h = bbox
                # depth_map[y:y + h, x:x + w] = depth
                # print(f'gt_boxes2d: {gt_boxes2d[b][n][i]}')
                x, y, w, h = gt_boxes2d[b][n][i]
                fg_mask[b * N + n, y:y + h, x:x + w] = True

    return fg_mask
