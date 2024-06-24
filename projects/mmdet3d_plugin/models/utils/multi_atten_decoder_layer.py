import warnings
import copy
import torch
from torch import Tensor
from mmcv.cnn import (build_norm_layer)
from mmcv.runner.base_module import BaseModule, ModuleList

from mmcv.utils import (
    ConfigDict,
    build_from_cfg,
)
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    FEEDFORWARD_NETWORK,
    TRANSFORMER_LAYER,
)

from typing import Optional


def build_attention(cfg, default_args=None):
    """Builder for attention."""
    return build_from_cfg(cfg, ATTENTION, default_args)


def build_feedforward_network(cfg, default_args=None):
    """Builder for feed-forward network (FFN)."""
    return build_from_cfg(cfg, FEEDFORWARD_NETWORK, default_args)


@TRANSFORMER_LAYER.register_module()
class MultiAttentionDecoderLayer(BaseModule):
    """MultiAttentionDecoderLayer for vision transformer.
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

        super().__init__(init_cfg)

        self.batch_first = batch_first

        assert set(operation_order) & {
            'self_attn', 'norm', 'ffn', 'cross_attn', 'cross_view_attn', 'cross_depth_attn', } == \
            set(operation_order), f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all four operation type ' \
            f"{['self_attn', 'norm', 'ffn', 'cross_attn', 'cross_view_attn','cross_depth_attn',]}"

        num_attn = operation_order.count('self_attn') + \
            operation_order.count('cross_attn') + \
            operation_order.count('cross_view_attn') + \
            operation_order.count('cross_depth_attn')

        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn',
                                  'cross_attn',
                                  'cross_view_attn',
                                  'cross_depth_attn',
                                  ]:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index],
                                          dict(type='FFN')))

        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

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
                depth_pos_embed: Optional[Tensor] = None,
                view_features: Optional[Tensor] = None,
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
            depth_pos_embed (Tensor): depth embedding from gt_depth_maps or depth_predictor
                shape `depth_pos_embed: [N*H*W, B, C]`
            view_features (Tensor): features from multi-view image through backbone.
                shape `view_features: [num_cameras, \sum_{i=0}^{L} H_i * W_i, B, C]`
`
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

            elif layer == 'cross_view_attn':
                query = self.attentions[attn_index](
                    query,
                    None,
                    # view_features: [num_cameras, \sum_{i=0}^{L} H_i * W_i, B, C]
                    view_features,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **cross_attn_args,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'cross_depth_attn':
                # depth_pos_embed: [N*H*W, B, C]
                depth_key = depth_value = depth_pos_embed
                query = self.attentions[attn_index](
                    query,
                    depth_key,
                    depth_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
