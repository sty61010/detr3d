import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence,
                                         build_attention,
                                         )
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from typing import Dict, List, Optional, Tuple
from .detr3d_transformer import Detr3DCrossAtten


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


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Detr3DTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of DETR3D.
    Args:
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(self, *args, post_norm_cfg=dict(type='LN'), **kwargs):
        super(Detr3DTransformerEncoder, self).__init__(*args, **kwargs)
        # if post_norm_cfg is not None:
        #     self.post_norm = build_norm_layer(
        #         post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        # else:
        #     assert not self.pre_norm, f'Use prenorm in ' \
        #                               f'{self.__class__.__name__},' \
        #                               f'Please specify post_norm_cfg'
        self.post_norm = None

    def forward(self, *args, **kwargs):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(Detr3DTransformerEncoder, self).forward(*args, **kwargs)
        # if self.post_norm is not None:
        #     x = self.post_norm(x)
        return x


@TRANSFORMER.register_module()
class DeformableDetr3DTransformer(BaseModule):
    """Implements the Deformable Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 grid_size=[0.512, 0.512, 2],
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 ** kwargs):
        super(DeformableDetr3DTransformer, self).__init__(**kwargs)
        # self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()

        """Initialize grid for bev grid: [x_voxels, y_voxels, 2] 
            (last dimesion for x, y coordinate)
        """
        self.grid = self.init_grid(grid_size=grid_size, pc_range=pc_range)

    def init_layers(self):
        """Initialize layers of the DeformableDer3DTransformer."""
        # self.level_embeds = nn.Parameter(
        #     torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
            if isinstance(m, DeformableCrossAttention):
                m.init_weight()
            if isinstance(m, Detr3DCrossAtten):
                m.init_weight()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        # normal_(self.level_embeds)

    def init_grid(self, grid_size, pc_range):
        """Initializes Grid Generator for frustum features
        Args:
            grid_size (list): Voxel shape [X, Y, Z]
            pc_range (list): Voxelization point cloud range [X_min, Y_min, Z_min, X_max, Y_max, Z_max]
            d_bound (list): Depth bound [depth_start, depth_end, depth_step]
        """
        self.grid_size = torch.tensor(grid_size)
        pc_range = torch.tensor(pc_range).reshape(2, 3)
        self.pc_min = pc_range[0]
        self.pc_max = pc_range[1]
        num_xyz_points = ((self.pc_max - self.pc_min) // self.grid_size).long()
        # print(f'num_xyz_points: {num_xyz_points}')
        x, y, z = [torch.linspace(pc_min, pc_max, num_points)
                   for pc_min, pc_max, num_points, size in zip(self.pc_min, self.pc_max, num_xyz_points, self.grid_size)]
        # gird: [X, Y, Z, 3]
        # self.grid = torch.stack(torch.meshgrid(x, y, z), dim=-1)
        # grid: [X, Y, Z, 4]
        # self.grid = torch.cat([self.grid, torch.ones((*self.grid.shape[:3], 1))], dim=-1)

        # [X, Y, Z, 2]
        return torch.stack(torch.meshgrid(x, y), dim=-1)

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                reg_branches=None,
                **kwargs):
        """Forward function for `DeformableDetr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                `[B, num_cameras, C, H_i, W_i]`
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [B, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs, num_query, embed_dims)

        """
        assert query_embed is not None

        # Check parameters
        bs = mlvl_feats[0].size(0)
        bev_grid = self.grid.clone()
        bev_grid = bev_grid.unsqueeze(0).repeat(bs, 1, 1, 1)
        # bev_grid: [bs, x_range, y_range, 2]
        bev_grid = bev_grid.permute(1, 2, 0, 3).view(-1, bs, 2)

        # mlvl_feats[0]: [B, num_cameras, C, H_i, W_i]
        # mlvl_masks[0]: [B, embed_dims, h, w].

        # Modified from only decoder
        # value[0]: [B, num_cameras, C, H_i, W_i]
        # spatial_shapes: [num_levels, 2]
        # level_start_index: [num_levels, ]
        value, spatial_shapes, level_start_index = flatten_features(mlvl_feats)

        # query: [num_query, bs, embed_dims]
        # query_pos: [num_query, bs, embed_dims]
        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)

        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        # reference_points: [bs, num_query, 3]
        reference_points = self.reference_points(query_pos).sigmoid()

        init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        # inter_states: [num_camera, B, num_query, embed_dims]
        # inter_references: [num_camera, B, num_query, 3]
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=value,
            query_pos=query_pos,
            mlvl_feats=mlvl_feats,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out


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
    assert all([feat.dim() == 5 for feat in mlvl_features]
               ), 'The shape of each element of `mlvl_features` must be [B, num_cameras, C, H_i, W_i].'
    # flat_features: [B, num_cameras, C, \sum_{i=0}^{num_levels} H_i * W_i]
    flat_features = torch.cat([feat.flatten(-2) for feat in mlvl_features], dim=-1)
    # flat_features: [num_cameras, \sum_{i=0}^{num_levels} H_i * W_i, B, C]
    flat_features = flat_features.permute(1, 3, 0, 2)

    spatial_shapes = torch.tensor([feat.shape[-2:] for feat in mlvl_features],
                                  dtype=torch.long, device=mlvl_features[0].device)
    level_start_index = torch.cat([
        spatial_shapes.new_zeros((1, )),
        spatial_shapes.prod(1).cumsum(0)[:-1]
    ])
    return flat_features, spatial_shapes, level_start_index


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DeformableDetr3DTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DeformableDetr3DTransformerDecoder.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(DeformableDetr3DTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                mlvl_feats=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `DeformableDetr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `[num_query, bs, embed_dims]`.
            reference_points (Tensor): The reference
                points of offset. has shape
                `[bs, num_query, 3]`
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            output(Tensor):
                `[num_query, B, embed_dims]`
            reference_points (Tensor): The reference
                points of offset. has shape
                `[bs, num_query, 3)]`
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points
            output = layer(
                output,
                *args,
                mlvl_feats=mlvl_feats,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[
                    ..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


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
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        # self.attention_weights = nn.Linear(embed_dims,
        #                                    num_cams * num_levels * num_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        # Modifirf for DeformableCrossAttention
        self.attention = build_attention(attn_cfg)

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        # constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

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
                mlvl_feats=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
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
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                `[B, num_cameras, C, H_i, W_i]`
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

        # reference_points: [B, num_query * num_points, num_levels, 3]
        # reference_points = self.get_reference_points(query_bev_pos, num_levels)
        reference_points = reference_points.unsqueeze(2).repeat_interleave(num_levels, dim=2)

        img_metas = kwargs['img_metas']
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        # lidar2img: [B, num_cameras, 4, 4]
        lidar2img = torch.from_numpy(np.asarray(lidar2img)).to(reference_points)

        # img_shape: [H, W, C]
        img_shape = img_metas[0]['img_shape'][0]

        # reference_points: [num_cameras, B, num_query*num_points, num_levels, 2]
        # masks: [num_cameras, B, num_query*num_points, num_levels]
        reference_points, masks = self.project_ego_to_image(
            reference_points,
            lidar2img,
            img_shape,  # eliminate the batch dim
        )

        # masks: [num_cameras, B, num_query*num_points, num_levels]
        #     -> [num_cameras, num_query*num_points, B]
        #     -> [num_cameras, num_query, num_points, B]
        masks = masks.transpose(1, 2).any(-1).view(self.num_cams, num_query, self.num_points, batch)

        # query: [num_query*num_points, B, embed_dims]
        # query_pos: [num_query*num_points, B, embed_dims]
        # query_bev_pos: [num_query * num_points, B, 2]
        # query, query_pos, query_bev_pos = self.expand_query(query, query_pos, query_bev_pos)
        query = query.repeat_interleave(self.num_points, dim=0)

        attention_features = query.new_zeros((self.num_cams, num_query, self.num_points, batch, embed_dims))
        for i, (ref_points, mask, val) in enumerate(zip(reference_points, masks, value)):
            # [num_query * num_points, B, embed_dims]
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
        attention_features = self.dropout(self.output_proj(attention_features))

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
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
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
