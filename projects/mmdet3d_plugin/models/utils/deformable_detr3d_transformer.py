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
from .dca import DeformableCrossAttention


def inverse_sigmoid(x: Tensor, eps: float = 1e-5):
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
        # [x_range, y_range, 2]
        bev_grid = self.grid.clone()
        x_range, y_range, _ = bev_grid.shape
        # [x_range, y_range, bs, 2] -> [x_range*y_range, bs, 2]
        bev_grid = bev_grid.unsqueeze(2).repeat_interleave(bs).flatten(0, 1)

        # mlvl_feats[0]: [B, num_cameras, C, H_i, W_i]
        # mlvl_masks[0]: [B, embed_dims, h, w].

        # Modified from only decoder
        # value[i]: [B, num_cameras, C, H_i, W_i]
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
        # [x_range*y_range, bs, embed_dims]
        # TODO: get bev features from encoder and remove this line
        bev_features = value
        # inter_states: [num_camera, B, num_query, embed_dims]
        # inter_references: [num_camera, B, num_query, 3]
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_features,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=bev_features.new_tensor([[x_range, y_range]]),
            level_start_index=bev_features.new_tensor([0, x_range * y_range]),
            reg_branches=reg_branches,
            **kwargs
        )

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out


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

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        reg_branches: Optional[nn.Module] = None,
        **kwargs
    ):
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
                value=value,
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
