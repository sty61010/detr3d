import warnings
import copy
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence,
                                         build_attention
                                         )
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from typing import Dict, List, Optional, Tuple
from .detr3d_transformer import Detr3DCrossAtten
from .dca import DeformableCrossAttention
from .spatial_cross_attention import SpatialCrossAttention


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
class DeformableDetr3DTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of DeformableDetr3D.
    Args:
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(self, *args, post_norm_cfg=dict(type='LN'), **kwargs):
        super(DeformableDetr3DTransformerEncoder, self).__init__(*args, **kwargs)
        # if post_norm_cfg is not None:
        #     self.post_norm = build_norm_layer(
        #         post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        # else:
        #     assert not self.pre_norm, f'Use prenorm in ' \
        #                               f'{self.__class__.__name__},' \
        #                               f'Please specify post_norm_cfg'
        self.post_norm = None

    def forward(self,
                query,
                key=None,
                value=None,
                # query_bev_pos=None,
                # spatial_shapes=None,
                # level_start_index=None,
                img_metas=None,
                self_attn_args=dict(),
                cross_attn_args=dict(),
                **kwargs):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super().forward(query=query,
                            key=key,
                            value=value,
                            # query_bev_pos=query_bev_pos,
                            # spatial_shapes=spatial_shapes,
                            # level_start_index=level_start_index,
                            img_metas=img_metas,
                            self_attn_args=self_attn_args,
                            cross_attn_args=cross_attn_args,
                            **kwargs)
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
                 grid_size=[4.096, 4.096, 8],
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 ** kwargs):
        super(DeformableDetr3DTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.two_stage_num_proposals = two_stage_num_proposals

        """ Initialize grid for bev grid: [x_voxels, y_voxels, 2]
            (last dimesion for x, y coordinate)
        """
        self.grid = self.init_grid(grid_size=grid_size, pc_range=pc_range)
        # bev_query: [x_range * y_range, C]
        # self.bev_query = nn.Embedding(self.grid[0] * self.grid[1],
        #                               self.embed_dims)
        self.bev_query = nn.Parameter(torch.Tensor(self.grid.shape[0] * self.grid.shape[1],
                                                   self.embed_dims))
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the DeformableDer3DTransformer."""
        self.bev_reference_points = nn.Linear(self.embed_dims, 2)
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
            # if isinstance(m, SpatialCrossAttention):
            #     m.init_weight()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.bev_reference_points, distribution='uniform', bias=0.)
        normal_(self.bev_query)

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

        # [X, Y, 2]
        return torch.stack(torch.meshgrid(x, y), dim=-1)

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                img_metas,
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
        # query_bev_pos: [x_range, y_range, 2]
        query_bev_pos = self.grid.clone().to(mlvl_feats[0].device)
        # query_bev_pos: [x_range, y_range, bs, 2] -> [x_range * y_range, bs, 2]
        query_bev_pos = query_bev_pos.unsqueeze(2).repeat_interleave(bs, 2).flatten(0, 1)

        # bev_query: [x_range * y_range, C] -> [x_range * y_range, bs, C]
        bev_query = self.bev_query.unsqueeze(1).repeat_interleave(bs, 1).to(mlvl_feats[0].device)

        # mlvl_feats[0]: [B, num_cameras, C, H_i, W_i]
        # mlvl_masks[0]: [B, embed_dims, h, w].

        # encoder
        ###
        # value[i]: [num_cameras, \sum_{i=0}^{L} H_i * W_i, B, C]
        # spatial_shapes: [num_levels, 2]
        # level_start_index: [num_levels, ]
        value, spatial_shapes, level_start_index = flatten_features(mlvl_feats)
        # [num_query, bs, 1, 2]
        reference_points = self.bev_reference_points(bev_query).sigmoid().unsqueeze(-2)
        # [x_range*y_range, bs, embed_dims]
        # TODO: get bev features from encoder and remove this line
        bev_memory = self.encoder(
            query=bev_query,
            value=value,
            self_attn_args=dict(
                reference_points=reference_points,
                spatial_shapes=spatial_shapes.new_tensor([[self.grid.shape[0], self.grid.shape[1]]]),
                level_start_index=level_start_index.new_tensor([0, self.grid.shape[0] * self.grid.shape[1]])
            ),
            cross_attn_args=dict(
                query_bev_pos=query_bev_pos,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            ),
            img_metas=img_metas,
        )
        # print(f'bev_memory: {bev_memory.shape}')

        ###

        # Modified from only decoder

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
            value=bev_memory,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes.new_tensor([[self.grid.shape[0], self.grid.shape[1]]]),
            level_start_index=level_start_index.new_tensor([0, self.grid.shape[0] * self.grid.shape[1]]),
            reg_branches=reg_branches,
            **kwargs
        )

        # inter_states, inter_references = self.decoder(
        # query=query,
        # key=None,
        # value=value,
        # query_pos=query_pos,
        # mlvl_feats=mlvl_feats,
        # reference_points=reference_points,
        # spatial_shapes=spatial_shapes,
        # level_start_index=level_start_index,
        # reg_branches=reg_branches,
        # img_metas=img_metas,
        # **kwargs)

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out


@TRANSFORMER_LAYER.register_module()
class DeformableDetr3DTransformerLayer(BaseTransformerLayer):
    """DeformableDetr3DTransformerLayer for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default: None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        super().__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            batch_first=batch_first,
            init_cfg=init_cfg,
            **kwargs
        )

    def forward(self,
                query: Tensor,
                key: Optional[Tensor] = None,
                value: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                key_pos: Optional[Tensor] = None,
                attn_masks: Optional[Tensor] = None,
                query_key_padding_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                self_attn_args=dict(),
                cross_attn_args=dict(),
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.
            self_attn_args (Dict): Additional arguments passed to the self-attention module.
            cross_attn_args (Dict): Additional arguments passed to the cross-attention module.
        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                f'attn_masks {len(attn_masks)} must be equal ' \
                f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **self_attn_args,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **cross_attn_args,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


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
                reference_points=reference_points_input[..., None, :2],
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
