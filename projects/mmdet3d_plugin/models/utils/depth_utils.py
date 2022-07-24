import math
import torch
from torch.nn import functional as F


def bin_depths(depth_map, mode="LID", depth_min=1e-3, depth_max=60, num_depth_bins=80, target=False):
    """
    Converts depth map into bin indices
    Args:
        depth_map [torch.Tensor(*, H, W)]: Depth Map
        mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_depth_bins [int]: Number of depth bins
        target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        If target is true, return the indices. Otherwise, return the logits.
        * indices [torch.Tensor(*, H, W)]: Depth bin indices with dtype torch.long.
        * logits [torch.Tensor(*, H, W, num_depth_bins+1)]: Depth bin logits with dtype torch.float.
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_depth_bins
        indices = (depth_map - depth_min) / bin_size
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_depth_bins * (1 + num_depth_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = num_depth_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
            (math.log(1 + depth_max) - math.log(1 + depth_min))
    else:
        raise NotImplementedError

    # Remove indicies outside of bounds
    mask = (indices < 0) | (indices > num_depth_bins) | (~torch.isfinite(indices))
    indices[mask] = num_depth_bins
    if target:
        # Convert to integer
        indices = indices.type(torch.long)
        return indices
    # [*, H, W, num_depth_bins + 1]
    logits = F.one_hot(indices, num_classes=num_depth_bins + 1).float()
    return logits
