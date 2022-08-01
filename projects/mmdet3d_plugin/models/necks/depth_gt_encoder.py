import torch
import torch.nn as nn
import torch.nn.functional as F
# from .transformer import TransformerEncoder, TransformerEncoderLayer
# from mmcv.runner import BaseModule
from mmdet.models import NECKS
import math
# from mmcv.cnn.bricks.transformer import (
#     # BaseTransformerLayer,
#     # MultiScaleDeformableAttention,
#     # TransformerLayerSequence,
#     build_transformer_layer_sequence,
#     # build_positional_encoding,
# )

from typing import (
    List,
    Optional
)


@NECKS.register_module()
class DepthGTEncoder(nn.Module):

    def __init__(self,
                 num_depth_bins=80,
                 depth_min=1e-3,
                 depth_max=60.0,
                 embed_dims=256,
                 num_levels=4,
                 gt_depth_maps_down_scale=8,
                 depth_gt_encoder_down_scale=4,
                 ):
        """
        Initialize depth gt encoder
        Args:
            model_cfg [EasyDict]: Depth classification network config
        """
        super().__init__()
        depth_num_bins = int(num_depth_bins)
        depth_min = float(depth_min)
        depth_max = float(depth_max)
        self.depth_max = depth_max
        self.depth_num_bins = depth_num_bins

        bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))
        bin_indice = torch.linspace(0, depth_num_bins - 1, depth_num_bins)
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
        bin_value = torch.cat([bin_value, torch.tensor([depth_max])], dim=0)
        self.depth_bin_values = nn.Parameter(bin_value, requires_grad=False)

        # Create modules
        d_model = embed_dims

        # self.depth_head = nn.Sequential(
        #     # nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
        #     nn.Conv2d(1 + depth_num_bins, d_model, kernel_size=(3, 3), padding=1),
        #     nn.GroupNorm(32, num_channels=d_model),
        #     nn.ReLU(),
        #     nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
        #     nn.GroupNorm(32, num_channels=d_model),
        #     nn.ReLU())
        """
        default down scale of gt_depth_maps: 8
        we need to consider the down scale of input feature map: input_down_scale
        input_down_scale = depth_maps_down_scale(default: 8) * depth_gt_encoder_down_scale
        """
        num_conv_layer = int(math.log(depth_gt_encoder_down_scale, 2) - 1)
        self.depth_head = nn.ModuleList()
        self.depth_head.append(nn.Sequential(
            nn.Conv2d(1 + depth_num_bins, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
        ))
        for i in range(num_conv_layer):
            self.depth_head.append(nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
                nn.GroupNorm(32, num_channels=d_model),
                nn.ReLU(),
            ))

        self.depth_pos_embed = nn.Embedding(int(self.depth_max) + 1, 256)

        self.num_levels = num_levels
        self.depth_gt_encoder_down_scale = depth_gt_encoder_down_scale

    def forward(
        self,
        mlvl_feats: Optional[List[torch.Tensor]] = None,
        mask: Optional[List[torch.Tensor]] = None,
        pos: Optional[List[torch.Tensor]] = None,
        gt_depth_maps: torch.Tensor = None,
    ):
        """Forward function.
        Args:
            mlvl_feats (Optional[List[torch.Tensor]]): Features from the upstream
                network, each is a 5D-tensor with shape
                `Optional[List[torch.Tensor]]: [B, N, C, H, W].`
            mask (Optional[List[torch.Tensor]): mask for feature map to fit in transformer encoder
                `Optional[List[torch.Tensor]]: [B, N, H, W]`
            pos (Optional[List[torch.Tensor]]): position embedding for input feature images
                `Optional[List[torch.Tensor]]: [B, N, C, H, W]`
            gt_depth_maps (torch.Tensor): gt_depth_map with default down sample scale=8.
                `[B, N, H, W]`

        Returns:
            depth_logits: one hot encoding to represent the predict depth_maps,
                in depth_gt_encoder default is None
                `[B, N, D, H, W]`
            depth_pos_embed: depth_embedding from gt_depth_maps or predcted_depth_maps
                for cross_depth_atten
                `[B, N, C, H, W]`
            weighted_depth: weight-sum value of predicted_depth_maps or gt_depth_maps
                `[B, N, H, W]`
        """

        depth_logits = None

        B, N, H, W, _ = gt_depth_maps.shape
        # gt_depth_maps: [B*N, H, W, D] -> [B*N, D, H, W]
        gt_depth_maps = gt_depth_maps.flatten(0, 1).permute(0, 3, 1, 2)
        # gt_depth_maps: [B*N, D, H, W]
        # gt_depth_maps = F.one_hot(gt_depth_maps, num_classes=self.depth_num_bins + 1).permute(0, 3, 1, 2).float()
        depth_probs = gt_depth_maps.clone()
        # print(f'gt_depth_maps; {gt_depth_maps.shape}')

        # gt_depth_embs: [B*N, C, H, W]
        # gt_depth_embs = self.depth_head(depth_probs)
        gt_depth_embs = gt_depth_maps
        for layer in self.depth_head:
            gt_depth_embs = layer(gt_depth_embs)

        # Down Scale from depth_gt_encoder_down_scale
        depth_probs = F.interpolate(depth_probs, scale_factor=1 / self.depth_gt_encoder_down_scale)
        weighted_depth = (depth_probs * self.depth_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)

        # depth embeddings with depth positional encodings
        BN, C, H, W = gt_depth_embs.shape
        depth_embed = gt_depth_embs

        depth_pos_embed_ip = self.interpolate_depth_embed(weighted_depth)
        depth_embed = depth_embed + depth_pos_embed_ip

        # depth_embed: [B, N, C, H, W]
        depth_embed = depth_embed.reshape(B, N, -1, H, W)
        # weighted_depth: [B, N, H, W]
        weighted_depth = weighted_depth.reshape(B, N, H, W)

        return depth_logits, depth_embed, weighted_depth

    def interpolate_depth_embed(self, depth):
        depth = depth.clamp(min=0, max=self.depth_max)
        pos = self.interpolate_1d(depth, self.depth_pos_embed)
        pos = pos.permute(0, 3, 1, 2)
        return pos

    def interpolate_1d(self, coord, embed):
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=embed.num_embeddings - 1)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta
