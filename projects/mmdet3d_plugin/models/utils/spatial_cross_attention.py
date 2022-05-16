from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import build_attention
from mmcv.runner.base_module import BaseModule
from torch import Tensor


def flatten_features(mlvl_features: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
    """Flatten multi-level features and return the flattened features,
        spatial shapes, and level_start_index.

    Args:
        mlvl_features (list(Tensor)): List of features from different level
            and different cameras. The i-th element has shape
            [B, num_cameras, C, H_i, W_i].

    Returns:
        flat_features (Tensor): Flattened features from all levels with shape
            [num_cameras, \sum_{i=0}^{L} H_i * W_i, B, C], where L is the
            number of levels.
        spatial_shapes (Tensor): Spatial shape of features in different levels.
            With shape [num_levels, 2], last dimension represents (H, W).
        level_start_index (Tensor): The start index of each level. A tensor has shape
            [num_levels, ] and can be represented as [0, H_0*W_0, H_0*W_0+H_1*W_1, ...].
    """
    assert all([feat.dim() == 5 for feat in mlvl_features]), 'The shape of each element of `mlvl_features` must be [B, num_cameras, C, H_i, W_i].'
    # [B, num_cameras, C, \sum_{i=0}^{num_levels} H_i * W_i]
    flat_features = torch.cat([feat.flatten(-2) for feat in mlvl_features], dim=-1)
    # [num_cameras, \sum_{i=0}^{num_levels} H_i * W_i, B, C]
    flat_features = flat_features.permute(1, 3, 0, 2)

    spatial_shapes = torch.tensor([feat.shape[-2:] for feat in mlvl_features], dtype=torch.long, device=mlvl_features[0].device)
    level_start_index = torch.cat([
        spatial_shapes.new_zeros((1, )),
        spatial_shapes.prod(1).cumsum(0)[:-1]
    ])
    return flat_features, spatial_shapes, level_start_index


@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    def __init__(
        self,
        num_points=4,
        d_bound=[-3., 5.],
        attn_cfg=None,
        init_cfg=None,
    ):
        """An attention module using multi-camera features.

        Args:
            num_points (int): The number of reference points sampled from each query.
                Default: 4.
            d_bound (list(int)): A list of int containing the lower and upper bound
                of z coordinate. Default: [-3.0, 5.0].
            attn_cfg (dict): The attention config of the attention sub-module.
                By default, `MultiScaleDeformableAttention` is preferable.
            init_cfg (dict): Initial config pass to the `BaseModule`. Default: None.
        """
        super().__init__(init_cfg)
        self.num_points = num_points
        self.d_bound = d_bound
        self.attention = build_attention(attn_cfg)

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
            uv: The normalized projected points (u, v) of each camera with shape
                [num_cameras, B, num_query, num_levels, 2]. All elements is range in [0, 1],
                top-left (0, 0), bottom-right (1, 1).
            mask: The mask of the valid projected points with shape
                [num_cameras, B, num_query, num_levels].
        """
        assert reference_points.shape[0] == lidar2img.shape[0], f'The number in the batch dimension must be equal. reference_points: {reference_points.shape}, lidar2img: {lidar2img.shape}'

        lidar2img = lidar2img[:, :, :3]
        # convert to homogeneous coordinate. [batch, num_query, num_levels, 4]
        reference_points = torch.cat([
            reference_points,
            reference_points.new_ones((*reference_points.shape[:-1], 1))
        ], dim=-1)
        # [num_cameras, batch, num_query, num_levels, 3]
        uvd: Tensor = torch.einsum('bnij,bqlj->nbqli', lidar2img, reference_points)
        uv = uvd[..., :2] / uvd[..., -1:]
        img_H, img_W = img_shape
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
        num_query, batch, _ = bev_pos.shape
        # [num_query*num_points, B, 1]
        z = np.random.uniform(self.d_bound[0], self.d_bound[1], size=(num_query * self.num_points, batch, 1))
        z = bev_pos.new_tensor(z)
        # [num_query*num_points, B, 2]
        bev_pos = bev_pos.repeat_interleave(self.num_points, dim=0)
        # [num_query*num_points, B, 3]
        reference_points = torch.cat([bev_pos, z], dim=-1)
        # [B, num_query*num_points, 3]
        reference_points = reference_points.transpose(0, 1)
        # [B, num_query*num_points, num_levels, 3]
        reference_points = reference_points.unsqueeze(2).repeat_interleave(num_levels, dim=2)
        return reference_points

    def expand_query(self, query: Tensor, query_pos: Optional[Tensor], query_bev_pos: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
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
        if query_pos is not None:
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
                `(num_cameras, num_key, bs, embed_dims)`.
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

        assert None not in (query, value, query_bev_pos, spatial_shapes)
        assert query.shape[0] == query_bev_pos.shape[0]
        assert query.shape[1] == query_bev_pos.shape[1] == value.shape[2]
        assert spatial_shapes.shape[0] == level_start_index.shape[0]

        num_query, batch, embed_dims = query.shape

        num_levels, _ = spatial_shapes.shape
        query_bev_pos = query_bev_pos.float()
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
        # [num_query, batch]
        num_hits = masks.sum((0, 2))
        # [num_query, batch, embed_dims]
        attention_features = attention_features.sum((0, 2)) / num_hits.unsqueeze(-1)
        attention_features = torch.nan_to_num(
            attention_features,
            nan=0.,
            posinf=0.,
            neginf=0.,
        )
        return attention_features


if __name__ == '__main__':
    device = torch.device('cuda:0')
    lidar2img = torch.tensor([
        [[[667.5999, -11.3459, -1.8519, -891.7272],
          [38.3278, 67.5572, -559.0681, 760.1029],
          [0.5589, 0.8291, 0.0114, -1.1843]],

         [[412.9886, 506.6375, 1.8396, -668.0064],
          [-20.6782, 71.7123, -553.2657, 870.4648],
          [-0.3188, 0.9478, -0.0041, -0.1035]],

         [[-365.1992, 355.5753, 9.6372, -12.7062],
          [-78.5942, 2.7520, -354.6458, 559.4022],
          [-0.9998, -0.0013, 0.0186, -0.0372]],

         [[-643.1426, -139.6496, -12.1919, 556.9597],
          [-20.3761, -62.4205, -556.1431, 853.0998],
          [-0.3483, -0.9370, -0.0269, -0.0905]],

         [[-259.5370, -605.4091, -16.6692, 100.7252],
          [42.5486, -48.6904, -556.4778, 741.4697],
          [0.5614, -0.8272, -0.0248, -1.1877]],

         [[369.1537, -550.5783, -9.0176, -553.2021],
          [71.9614, 7.6353, -557.7432, 728.3124],
          [0.9998, 0.0181, -0.0075, -1.5520]]],


        [[[-259.5004, -605.4585, -15.3986, 99.6544],
          [41.3351, -49.3306, -556.5129, 741.5051],
          [0.5614, -0.8272, -0.0250, -1.1854]],

         [[667.6012, -10.9248, -3.3214, -886.1404],
          [37.0460, 67.0861, -559.2112, 760.1919],
          [0.5584, 0.8295, 0.0094, -1.1797]],

         [[-643.1819, -139.5975, -10.6162, 555.6953],
          [-21.5902, -63.1311, -556.0172, 852.9597],
          [-0.3485, -0.9370, -0.0249, -0.0912]],

         [[413.0042, 506.6281, 0.1605, -668.2244],
          [-21.8812, 70.8614, -553.3291, 870.4777],
          [-0.3188, 0.9478, -0.0048, -0.1033]],

         [[-365.1406, 355.6276, 9.9227, -12.8987],
          [-79.3611, 2.2571, -354.4784, 559.3275],
          [-0.9998, -0.0012, 0.0208, -0.0378]],

         [[369.3259, -550.4586, -9.2696, -550.8873],
          [70.7297, 7.0912, -557.9079, 728.5782],
          [0.9998, 0.0185, -0.0097, -1.5458]]]
    ], device=device)
    img_shape = (256, 704)
    feat_H, feat_W = img_shape[0] // 16, img_shape[1] // 16
    embed_dims = 8
    num_levels = 3
    bev_x, bev_y = 4, 4

    cfg = dict(
        type='SpatialCrossAttention',
        attn_cfg=dict(
            type='MultiScaleDeformableAttention',
            num_levels=num_levels,
            embed_dims=embed_dims,
        )
    )
    model: SpatialCrossAttention = build_attention(cfg).to(device)

    batch, num_cameras, _, _ = lidar2img.shape
    img_metas = dict(
        lidar2img=lidar2img,
        img_shape=[img_shape] * batch,
    )
    num_query = bev_x * bev_y

    query = torch.rand((num_query, batch, embed_dims), device=device)
    query_bev_pos = torch.stack(
        torch.meshgrid(torch.arange(bev_x), torch.arange(bev_y)),
        dim=-1
    ).flatten(0, 1).unsqueeze(1).repeat_interleave(batch, dim=1).to(device)
    value = [
        torch.rand((batch, num_cameras, embed_dims, feat_H >> i, feat_W >> i), device=device)
        for i in range(num_levels)
    ]
    print(f'val: {[feat.shape for feat in value]}')
    value, spatial_shapes, level_start_index = flatten_features(value)
    print(f'query: {query.shape}, query_bev_pos: {query_bev_pos.shape}')
    print(f'flat_val: {value.shape},\nspatial_shapes: {spatial_shapes},\nlevel_start_index: {level_start_index}')
    output = model(
        query=query,
        value=value,
        query_bev_pos=query_bev_pos,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        img_metas=img_metas,
    )
    # print(output)
