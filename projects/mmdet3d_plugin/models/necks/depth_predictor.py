import torch
import torch.nn as nn
import torch.nn.functional as F
# from .transformer import TransformerEncoder, TransformerEncoderLayer
# from mmcv.runner import BaseModule
from mmdet.models import NECKS
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence,
                                         build_positional_encoding
                                         )


@NECKS.register_module()
class DepthPredictor(nn.Module):

    def __init__(self,
                 num_depth_bins=80,
                 depth_min=1e-3,
                 depth_max=60.0,
                 embed_dims=256,
                 encoder=None,
                 num_levels=4,
                 ):
        """
        Initialize depth predictor and depth encoder
        Args:
            model_cfg [EasyDict]: Depth classification network config
        """
        super().__init__()
        depth_num_bins = int(num_depth_bins)
        depth_min = float(depth_min)
        depth_max = float(depth_max)
        self.depth_max = depth_max

        bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))
        bin_indice = torch.linspace(0, depth_num_bins - 1, depth_num_bins)
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
        bin_value = torch.cat([bin_value, torch.tensor([depth_max])], dim=0)
        self.depth_bin_values = nn.Parameter(bin_value, requires_grad=False)

        # Create modules
        d_model = embed_dims
        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model))
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))
        self.upsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))

        self.depth_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU())

        self.depth_classifier = nn.Conv2d(d_model, depth_num_bins + 1, kernel_size=(1, 1))

        if encoder is not None:
            self.depth_encoder = build_transformer_layer_sequence(encoder)
        # depth_encoder_layer = TransformerEncoderLayer(
        #     d_model, nhead=8, dim_feedforward=256, dropout=0.1)

        # self.depth_encoder = TransformerEncoder(depth_encoder_layer, 1)

        self.depth_pos_embed = nn.Embedding(int(self.depth_max) + 1, 256)

        self.num_levels = num_levels

    def forward(self,
                mlvl_feats,
                mask=None,
                pos=None):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        assert len(mlvl_feats) == self.num_levels
        # mlvl_feats (tuple[Tensor]): [B, N, C, H, W]
        B, N, C, H, W = mlvl_feats[0].shape
        # print(f'mlvl_feats: {mlvl_feats[0].shape}')

        # flatten_feats (tuple[Tensor]): [B*N, C, H, W]
        flatten_feats = []
        for idx, feat in enumerate(mlvl_feats):
            flatten_feats.append(feat.flatten(0, 1))

        # foreground depth map
        src_16 = self.proj(flatten_feats[1])
        src_32 = self.upsample(F.interpolate(flatten_feats[2], size=src_16.shape[-2:]))
        src_8 = self.downsample(flatten_feats[0])
        src = (src_8 + src_16 + src_32) / 3

        src = self.depth_head(src)
        depth_logits = self.depth_classifier(src)

        depth_probs = F.softmax(depth_logits, dim=1)
        weighted_depth = (depth_probs * self.depth_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)

        # print(f'src: {src.shape}')

        # depth embeddings with depth positional encodings
        BN, C, H, W = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        # mask = mask.flatten(1)
        # pos = pos.flatten(2).permute(2, 0, 1)

        depth_embed = self.depth_encoder(src, mask, pos)
        depth_embed = depth_embed.permute(1, 2, 0).reshape(BN, C, H, W)

        depth_pos_embed_ip = self.interpolate_depth_embed(weighted_depth)
        depth_embed = depth_embed + depth_pos_embed_ip

        # # depth_logits: [B, N, D, H, W]
        depth_logits = depth_logits.reshape(B, N, -1, H, W)
        # # depth_embed: [B, N, C, H, W]
        depth_embed = depth_embed.reshape(B, N, -1, H, W)
        # # weighted_depth: [B, N, H, W]
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
