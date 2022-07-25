import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import build_attention
from mmcv.runner.base_module import BaseModule

from typing import Tuple


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@ATTENTION.register_module()
class DeformableCrossAttention(BaseModule):
    """An DeformableCrossAttention module used in DeformableDetr3DTransformerDecoder. 
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 6.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=6,
                 num_levels=4,
                 num_points=4,
                 num_cams=6,
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 attn_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super(DeformableCrossAttention, self).__init__(init_cfg)

        # dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        # self.dropout = nn.Dropout(dropout)
        self.pc_range = torch.tensor(pc_range, dtype=torch.float32)

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        # self.attention_weights = nn.Linear(embed_dims,
        #                                    num_cams * num_levels * num_points)

        # self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        # Modified for DeformableCrossAttention
        self.attention = build_attention(attn_cfg)

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        # constant_init(self.attention_weights, val=0., bias=0.)
        # xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def project_ego_to_image(self, reference_points: Tensor, lidar2img: Tensor, img_shape: Tuple[int]) -> Tuple[Tensor, Tensor]:
        """Project ego-pose coordinate to image coordinate
        Args:
            reference_points (Tensor): 3D (x, y, z) reference points in ego-pose
                with shape `[B, num_query, num_levels, 3]`.
            lidar2img (Tensor): Transform matrix from lidar (ego-pose) coordinate to
                image coordinate with shape [B, num_cameras, 4, 4] or [B, num_cameras, 3, 4].
            img_shape (tuple): Image shape (height, width, channel).
                Note that this is not the input shape of the frustum.
                This is the shape with respect to the intrinsic.
        Returns:
            uv: The normalized projected points (u, v) of each camera with shape
                [num_cameras, B, num_query, num_levels, 2]. All elements is range in [0, 1],
                top-left (0, 0), bottom-right (1, 1).
            mask: The mask of the valid projected points with shape
                [num_cameras, B, num_query, num_levels].
        """
        assert reference_points.shape[0] == lidar2img.shape[
            0], f'The number in the batch dimension must be equal. reference_points: {reference_points.shape}, lidar2img: {lidar2img.shape}'

        lidar2img = lidar2img[:, :, :3]
        # convert to homogeneous coordinate. [batch, num_query, num_levels, 4]
        reference_points = torch.cat([
            reference_points,
            reference_points.new_ones((*reference_points.shape[:-1], 1))
        ], dim=-1)
        # [num_cameras, batch, num_query, num_levels, 3]
        uvd: Tensor = torch.einsum('bnij,bqlj->nbqli', lidar2img, reference_points)
        uv = uvd[..., :2] / uvd[..., -1:]
        img_H, img_W, _ = img_shape
        # normalize to [0, 1]
        uv /= uv.new_tensor([img_W, img_H]).reshape(1, 1, 1, 1, 2)
        # [num_cameras, batch, num_query, num_levels, 2]
        mask = (torch.isfinite(uv)
                & (uv[..., 0:1] >= 0.0)
                & (uv[..., 0:1] <= 1.0)
                & (uv[..., 1:2] >= 0.0)
                & (uv[..., 1:2] <= 1.0))
        mask = torch.all(mask, dim=-1)

        return uv, mask

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                img_metas=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                `[num_query, B, embed_dims]`
            key (Tensor): The key tensor with shape
                `[num_key, B, embed_dims]`
            value (Tensor): The value tensor with shape
                `(num_key, B, embed_dims)`. [B, N, C, H, W]
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
                `[num_query, B, embed_dims]`

            reference_points (Tensor): 
                `[B, num_query, 3]`
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape '[bs, num_key]'
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape
                `[num_levels, 2]`
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
                `[num_levels, ]`
        Returns:
             Tensor: forwarded results with shape 
             `[num_query, B, embed_dims]`
        """

        # if key is None:
        #     key = query
        # if value is None:
        #     value = key
        if residual is None:
            # inp_residual: [nm_query, B, emb_dims]
            inp_residual = query
        # if query_pos is not None:
        #     query = query + query_pos

        # # change to (B, num_query, embed_dims)
        # query = query.permute(1, 0, 2)

        # bs, num_query, _ = query.size()

        # # attention_weights: [B, 1, num_query, num_cams, num_points, num_levels]
        # attention_weights = self.attention_weights(query).view(
        #     bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)

        # # reference_points: [B, num_query, 3]
        # reference_points_3d, output, mask = feature_sampling(
        #     mlvl_feats, reference_points, self.pc_range, kwargs['img_metas'])
        # # reference_points_3d: [B, num_query, 3]

        # output = torch.nan_to_num(output)
        # mask = torch.nan_to_num(mask)

        # attention_weights = attention_weights.sigmoid() * mask

        # # output: [B, dim, num_query, num_cams , num_points, num_levels]
        # output = output * attention_weights

        # # output_embedding: [B, dim, num_query]
        # output = output.sum(-1).sum(-1).sum(-1)

        # # output: [num_query, B, dim]
        # output = output.permute(2, 0, 1)

        # output = self.output_proj(output)
        # # pos_feat: (num_query, B, embed_dims)
        # pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        # return self.dropout(output) + inp_residual + pos_feat
####
        num_query, batch, embed_dims = query.shape

        num_levels, _ = spatial_shapes.shape
        # query_bev_pos = query_bev_pos.float()

        # pos_feat: [num_query, B, embed_dim]
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points)).permute(1, 0, 2)

        pc_min, pc_max = self.pc_range[:3].to(reference_points.device), self.pc_range[3:].to(reference_points.device)
        reference_points = reference_points * (pc_max - pc_min) + pc_min
        # reference_points: [B, num_query, num_levels, 3]
        reference_points = reference_points.unsqueeze(2).repeat_interleave(num_levels, dim=2)
        # print(f'reference_points: {reference_points.shape}')

        # lidar2img: [B, num_cameras, 4, 4]
        lidar2img = query.new_tensor([img_meta['lidar2img'] for img_meta in img_metas])

        # img_shape: [H, W, C]
        img_shape = img_metas[0]['img_shape'][0]

        # reference_points: [num_cameras, B, num_query, num_levels, 2]
        # masks: [num_cameras, B, num_query, num_levels]
        reference_points, masks = self.project_ego_to_image(
            reference_points,
            lidar2img,
            img_shape,  # eliminate the batch dim
        )

        # masks: [num_cameras, B, num_query, num_levels]
        #     -> [num_cameras, num_query, B]
        masks = masks.transpose(1, 2).any(-1)

        attention_features = query.new_zeros((self.num_cams, num_query, batch, embed_dims))
        for i, (ref_points, mask, val) in enumerate(zip(reference_points, masks, value)):
            # [num_query, B, embed_dims]
            attn = self.attention(
                query=query,
                value=val,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                reference_points=ref_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs,
            )
            # mask: [num_query, B]
            attention_features[i, mask] = attn[mask]

        # TODO: Use weighted sum
        # [num_query, batch]
        num_hits = masks.sum(0)
        # [num_query, batch, embed_dims]
        attention_features = attention_features.sum(0) / num_hits.unsqueeze(-1)
        attention_features = torch.nan_to_num(
            attention_features,
            nan=0.,
            posinf=0.,
            neginf=0.,
        )

        return attention_features + inp_residual + pos_feat


def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    # reference_points (B, num_queries, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
                 & (reference_points_cam[..., 0:1] < 1.0)
                 & (reference_points_cam[..., 1:2] > -1.0)
                 & (reference_points_cam[..., 1:2] < 1.0))
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B * N, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B * N, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam, 1, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask
