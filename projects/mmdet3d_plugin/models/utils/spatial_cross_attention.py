from typing import Dict, Optional, Tuple

import numpy as np
import torch
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import build_attention
from mmcv.runner.base_module import BaseModule
from torch import Tensor


@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    def __init__(
        self,
        num_points=4,
        d_bound=[-3., 5.],
        attn_cfg=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.num_points = num_points
        self.d_bound = d_bound
        self.attention = build_attention(attn_cfg)
        self.init_weights()

    def project_ego_to_image(self, reference_points: Tensor, lidar2img: Tensor, img_shape: Tuple[int]) -> Tuple[Tensor, Tensor]:
        """Project ego-pose coordinate to image coordinate

        Args:
            reference_points (Tensor): 3D (x, y, z) reference points in ego-pose
                with shape [B, num_query, num_levels, 3].
            lidar2img (Tensor): Transform matrix from lidar (ego-pose) coordinate to
                image coordinate with shape [B, num_cameras, 4, 4] or [B, num_cameras, 3, 4].
            img_shape (tuple): Image shape (height, width).
                Note that this is not the input shape of the frustum.
                This is the shape with respect to the intrinsic.
        Returns:
            uv: The projected points (u, v) of each camera with shape
                [num_cameras, B, num_query, num_levels, 2].
            mask: The mask of the valid projected points with shape
                [num_cameras, B, num_query, num_levels].
        """
        assert reference_points.shape[0] == lidar2img.shape[0], f'The number in the batch dimension must be equal. reference_points: {reference_points.shape}, lidar2img: {lidar2img.shape}'

        lidar2img = lidar2img[:, :, :3]

        # [num_cameras, batch, X, Y, num_points, 3]
        uvd = torch.einsum('bnij,bqlj->nbqli', lidar2img, reference_points)
        uv = uvd[..., :2] / uvd[..., -1:]
        img_H, img_W = img_shape
        # normalize to [0, 1]
        uv /= Tensor([img_W, img_H], dtype=uvd.dtype, device=uvd.device).reshape(*uv.shape[:-1], 2)
        mask = ~torch.isnan(uv)
        mask &= ((uv[..., 0:1] > -1.0)
                 & (uv[..., 0:1] < 1.0)
                 & (uv[..., 1:2] > -1.0)
                 & (uv[..., 1:2] < 1.0))
        mask = torch.all(mask, dim=-1)

        return uv, mask

    def get_reference_points(self, bev_pos: Tensor, num_levels: int) -> Tensor:
        """Sample `self.num_points` of reference points for every BEV position

        Args:
            bev_pos (Tensor): BEV position with shape [num_query, B, 2]. The first value
                is the x position, and the second one is the y position.
            num_levels (int): The number of feature levels.

        Returns:
            Tensor. A tensor containing the reference points (x, y, z) with shape
                [B, num_query * `self.num_points`, num_levels, 3], which means
                `self.num_points` of reference points are sampled for each BEV position.
                The sampling range of the z coordinate is `self.d_bound`.
        """
        # [num_query, B, num_points, 1]
        z = np.random.uniform(self.d_bound[0], self.d_bound[1], size=(*bev_pos.shape[:2], self.num_points, 1))
        z = Tensor(z, dtype=bev_pos.dtype, device=bev_pos.device)
        # [num_query, B, num_points, 2]
        bev_pos = bev_pos.unsqueeze(2).repeat_interleave(self.num_points, dim=2)
        assert bev_pos.shape == (*z.shape[:-1], 2)
        # [num_query, B, num_points, 3]
        reference_points = torch.cat([bev_pos, z], dim=-1)
        # [B, num_query * num_points, 3]
        reference_points = reference_points.transpose(0, 1).flatten(1, 2)
        # [B, num_query * num_points, num_levels, 3]
        reference_points = reference_points.unsqueeze(3).repeat_interleave(num_levels, dim=3)
        return reference_points

    def expand_query(self, query: Tensor, query_pos: Tensor, query_bev_pos: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Expand query to `self.num_points` times

        Args:
            query (Tensor): The query tensor with shape [num_query, B, embed_dims].
            query_pos (Tensor): The positional encoding for `query` with shape
                [num_query, B, embed_dims].
            query_bev_pos (Tensor): The 2D (x, y) position of each `query` with shape
                [num_query, bs, 2].

        Returns:
            query (Tensor): The query tensor with shape
                [num_query * `self.num_points`, B, embed_dims].
            query_pos (Tensor): The positional encoding for `query` with shape
                [num_query * `self.num_points`, B, embed_dims].
            query_bev_pos (Tensor): The 2D (x, y) position of each `query` with shape
                [num_query * `self.num_points`, B, 2].

        """
        query = query.repeat_interleave(self.num_points, dim=0)
        # TODO: Maybe add some position embedding here?
        query_pos = query_pos.repeat_interleave(self.num_points, dim=0)
        query_bev_pos = query_bev_pos.repeat_interleave(self.num_points, dim=0)
        return query, query_pos, query_bev_pos

    def forward(
        self,
        query: Tensor,
        key=None,
        value: Tensor = None,
        query_pos: Optional[Tensor] = None,
        query_bev_pos: Tensor = None,
        key_padding_mask: Optional[Tensor] = None,
        spatial_shapes: Tensor = None,
        level_start_index: Optional[Tensor] = None,
        img_metas: Dict[str, Tensor] = None,
        **kwargs,
    ):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(B, num_cameras, embed_dims, H, W)`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            query_bev_pos (Tensor): The 2D (x, y) position of each `query` with shape
                (num_query, bs, 2).
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...]. If not given, it will be
                generated from `spatial_shapes`.

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        # Generate `level_start_index` if it is None and `spatial_shapes` exists
        if spatial_shapes is not None and level_start_index is None:
            level_start_index = torch.cat([
                spatial_shapes.new_zeros((1, )),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            ])

        assert query.shape[0] == query_bev_pos.shape[0]
        assert query.shape[1] == query_bev_pos.shape[1] == value.shape[0]
        assert spatial_shapes.shape[0] == level_start_index.shape[0]

        num_query, batch, embed_dims = query.shape
        _, num_cameras, _, _, _ = value.shape
        # [B, num_cameras, embed_dims, H, W] -> [B, num_cameras, embed_dims, H*W] -> [num_cameras, H*W, B, embed_dims]
        value = value.flatten(-2).permute(1, 3, 0, 2)

        num_levels, _ = spatial_shapes.shape
        # [B, num_query*num_points, num_levels, 3]
        reference_points = self.get_reference_points(query_bev_pos, num_levels)

        # reference_points: [num_cameras, B, num_query*num_points, num_levels, 2]
        # masks: [num_cameras, B, num_query*num_points, num_levels]
        reference_points, masks = self.project_ego_to_image(
            reference_points,
            img_metas['lidar2img'],
            img_metas['img_shape'][0]  # eliminate the batch dim
        )
        # masks: [num_cameras, B, num_query*num_points, num_levels]
        #     -> [num_cameras, num_query*num_points, B]
        #     -> [num_cameras, num_query, num_points, B]
        masks = masks.transpose(1, 2).any(-1).view(num_cameras, num_query, self.num_points, batch)

        # query: [num_query*num_points, B, embed_dims]
        # query_pos: [num_query*num_points, B, embed_dims]
        # query_bev_pos: [num_query*num_points, B, 2]
        query, query_pos, query_bev_pos = self.expand_query(query, query_pos, query_bev_pos)

        attention_features = query.new_zeros((num_cameras, num_query, self.num_points, batch, embed_dims))
        for i, (ref_points, mask, val) in enumerate(zip(reference_points, masks, value)):
            # [num_query*num_points, B, embed_dims]
            attn = self.attention(
                query=query,
                value=val,
                query_pos=query_pos,
                query_bev_pos=query_bev_pos,
                key_padding_mask=key_padding_mask,
                reference_points=ref_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                img_metas=img_metas,
                **kwargs,
            )
            attn = attn.view(num_query, self.num_points, batch, embed_dims)
            # mask: [num_query, num_points, B]
            attention_features[i, mask] = attn[mask]

        # TODO: Use weighted sum
        # [num_query, batch, embed_dims]
        attention_features = attention_features.sum((0, 2))
        return attention_features
