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


@NECKS.register_module()
class DepthGTEncoder(nn.Module):

    def __init__(self,
                 num_depth_bins=80,
                 depth_min=1e-3,
                 depth_max=60.0,
                 embed_dims=256,
                 encoder=None,
                 num_levels=4,
                 with_gt_depth_maps=False,
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
        # self.downsample = nn.Sequential(
        #     nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
        #     nn.GroupNorm(32, d_model))
        # self.proj = nn.Sequential(
        #     nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
        #     nn.GroupNorm(32, d_model))
        # self.upsample = nn.Sequential(
        #     nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
        #     nn.GroupNorm(32, d_model))

        # self.depth_head = nn.Sequential(
        #     # nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
        #     nn.Conv2d(1 + depth_num_bins, d_model, kernel_size=(3, 3), padding=1),
        #     nn.GroupNorm(32, num_channels=d_model),
        #     nn.ReLU(),
        #     nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
        #     nn.GroupNorm(32, num_channels=d_model),
        #     nn.ReLU())

        # default down scale of gt_depth_maps: 8
        # we need to consider the down scale of input feature map: input_down_scale
        # input_down_scale = depth_maps_down_scale(default: 8) * depth_gt_encoder_down_scale
        num_conv_layer = int(math.log(depth_gt_encoder_down_scale, 2) - 1)
        # print(f'num_conv_layer: {num_conv_layer}')
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

        # if not with_gt_depth_maps:
        #     self.depth_classifier = nn.Conv2d(d_model, depth_num_bins + 1, kernel_size=(1, 1))

        # if encoder is not None:
        #     self.depth_encoder = build_transformer_layer_sequence(encoder)

        self.depth_pos_embed = nn.Embedding(int(self.depth_max) + 1, 256)

        self.num_levels = num_levels
        self.depth_gt_encoder_down_scale = depth_gt_encoder_down_scale

    def forward(
        self,
        mlvl_feats=None,
        mask=None,
        pos=None,
        gt_depth_maps=None,
    ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            gt_depth_maps (Tensor): [B, N, H, W]
            default down scale of gt_depth_maps: 8
            we need to consider the down scale of input feature map: input_down_scale
            input_down_scale = depth_maps_down_scale(default: 8) * depth_gt_encoder_down_scale

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        # assert len(mlvl_feats) == self.num_levels
        # # mlvl_feats (tuple[Tensor]): [B, N, C, H, W]
        # B, N, C, H, W = mlvl_feats[1].shape
        # src = mlvl_feats[1].flatten(0, 1)
        # print(f'src: {src.shape}')
        # # flatten_feats (tuple[Tensor]): [B*N, C, H, W]
        # flatten_feats = []
        # for idx, feat in enumerate(mlvl_feats):
        #     flatten_feats.append(feat.flatten(0, 1))

        # # foreground depth map
        # src_16 = self.proj(flatten_feats[1])
        # src_32 = self.upsample(F.interpolate(flatten_feats[2], size=src_16.shape[-2:]))
        # src_8 = self.downsample(flatten_feats[0])
        # src = (src_8 + src_16 + src_32) / 3

        # src = self.depth_head(src)
        # print(f'src: {src.shape}')

        depth_logits = None

        # if gt_depth_maps is not None:
        B, N, H, W = gt_depth_maps.shape
        # gt_depth_maps: [B*N, H, W]
        gt_depth_maps = gt_depth_maps.flatten(0, 1)
        # gt_depth_maps: [B*N, D, H, W]
        gt_depth_maps = F.one_hot(gt_depth_maps, num_classes=self.depth_num_bins + 1).permute(0, 3, 1, 2).float()
        # print(f'gt_depth_maps: {gt_depth_maps.shape}')
        depth_probs = gt_depth_maps.clone()

        # gt_depth_embs: [B*N, C, H, W]
        # gt_depth_embs = self.depth_head(depth_probs)
        gt_depth_embs = gt_depth_maps
        for layer in self.depth_head:
            gt_depth_embs = layer(gt_depth_embs)

        # print(f'gt_depth_embs: {gt_depth_embs.shape}')
        # Down Scale from depth_gt_encoder_down_scale
        depth_probs = F.interpolate(depth_probs, scale_factor=1 / self.depth_gt_encoder_down_scale)
        # print(f'depth_probs: {depth_probs.shape}')

        # else:
        #     # depth_logits:[B*N, D, H, W]
        #     depth_logits = self.depth_classifier(src)

        #     # depth_probs:[B*N, D, H, W]
        #     depth_probs = F.softmax(depth_logits, dim=1)
        #     # print(f'depth_probs: {depth_probs.shape}')

        weighted_depth = (depth_probs * self.depth_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)

        # print(f'src: {src.shape}')

        # depth embeddings with depth positional encodings
        # BN, C, H, W = src.shape
        # src = src.flatten(2).permute(2, 0, 1)
        # mask = mask.flatten(0, 1)
        # pos = pos.flatten(2).permute(2, 0, 1)

        BN, C, H, W = gt_depth_embs.shape
        # gt_depth_embs: [H*W, B*N, C]
        # gt_depth_embs = gt_depth_embs.flatten(2).permute(2, 0, 1)

        # depth_embed = self.depth_encoder(gt_depth_embs, mask, pos)
        # depth_embed = depth_embed.permute(1, 2, 0).reshape(BN, C, H, W)
        depth_embed = gt_depth_embs

        depth_pos_embed_ip = self.interpolate_depth_embed(weighted_depth)
        depth_embed = depth_embed + depth_pos_embed_ip

        # depth_logits: [B, N, D, H, W]
        # depth_logits = depth_logits.reshape(B, N, -1, H, W)

        # depth_embed: [B, N, C, H, W]
        depth_embed = depth_embed.reshape(B, N, -1, H, W)
        # weighted_depth: [B, N, H, W]
        weighted_depth = weighted_depth.reshape(B, N, H, W)

        # print(f'depth_logits: {depth_logits.shape}')
        # print(f'depth_embed: {depth_embed.shape}')
        # print(f'weighted_depth: {weighted_depth.shape}')

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
