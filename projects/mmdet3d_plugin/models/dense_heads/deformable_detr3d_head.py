import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmdet3d.models.builder import build_neck
from mmdet.core import multi_apply, reduce_mean
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmdet.models.utils.transformer import inverse_sigmoid

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.models.utils import depth_utils


@HEADS.register_module()
class DeformableDetr3DHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        depth_predictor (obj:`ConfigDict`): ConfigDict is used for building
            depth_predictor, which use feature map to predict weight depth
            distribution and depth embedding.
            `Optional[ConfigDict]`
        depth_gt_encoder (obj:`ConfigDict`): ConfigDict is used for building
            deoth_gt_encoder, which use convolutional layer to suppress gt_depth_maps
            to gt_depth_embedding.
            `Optional[ConfigDict]`
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 depth_predictor=None,
                 depth_gt_encoder=None,
                 loss_ddn=None,
                 loss_depth=False,
                 ** kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.num_cls_fcs = num_cls_fcs - 1
        super(DeformableDetr3DHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.positional_encoding = None
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

        """
        Operation for depth embedding
            depth_bin_cfg: Config for depth_utils.bin_depths
            depth_predictor (obj:`ConfigDict`): ConfigDict is used for building
                depth_predictor, which use feature map to predict weight depth
                distribution and depth embedding.
                `Optional[ConfigDict]`
            depth_gt_encoder (obj:`ConfigDict`): ConfigDict is used for building
                deoth_gt_encoder, which use convolutional layer to suppress gt_depth_maps
                to gt_depth_embedding.
                `Optional[ConfigDict]`
        """
        self.depth_bin_cfg = None
        self.depth_predictor = None
        self.depth_gt_encoder = None
        self.loss_ddn = None
        self.depth_maps_down_scale = 8
        self.gt_depth_maps_down_scale = 8
        if depth_predictor is not None:
            self.depth_predictor = build_neck(depth_predictor)
            self.depth_bin_cfg = dict(
                mode="LID",
                depth_min=depth_predictor.get("depth_min"),
                depth_max=depth_predictor.get("depth_max"),
                num_depth_bins=depth_predictor.get("num_depth_bins"),
            )

        if depth_gt_encoder is not None:
            self.depth_gt_encoder = build_neck(depth_gt_encoder)
            self.depth_bin_cfg = dict(
                mode="LID",
                depth_min=depth_gt_encoder.get("depth_min"),
                depth_max=depth_gt_encoder.get("depth_max"),
                num_depth_bins=depth_gt_encoder.get("num_depth_bins"),
            )
            self.gt_depth_maps_down_scale = depth_gt_encoder.get("gt_depth_maps_down_scale")

        if loss_ddn is not None:
            self.loss_ddn = build_loss(loss_ddn)
            self.depth_maps_down_scale = loss_ddn.get("downsample_factor")

        self.loss_depth = loss_depth

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward(
            self,
            mlvl_feats,
            img_metas,
            gt_bboxes_3d=None,
    ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).

            img_metas: A list of dict containing the `lidar2img` tensor.
            gt_bboxes_3d: The ground truth list of `LiDARInstance3DBoxes`.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds(torch.Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format
                (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy).
                `[num_layer, B, num_queries, 10]`
        """

        # Modified for deformable detr
        # mlvl_feats: (tuple[Tensor]): [bs, num_cams, C, H, W]
        batch_size, num_cams, C, H, W = mlvl_feats[0].size()
        # for feat in mlvl_feats:
        #     print(f'feat: {feat.shape}')

        # [input_img_h, input_img_w]: pad_shape: [H, W]
        # input_img_h, input_img_w = mlvl_feats[0].shape[-2], mlvl_feats[0].shape[-1]
        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        # print(f'input_img_h: {input_img_h}, input_img_w: {input_img_w}')

        # img_masks: [bs, num_cams, H, W]
        # img_masks = mlvl_feats[0].new_ones(
        #     (batch_size, input_img_h, input_img_w))
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, num_cams, input_img_h, input_img_w))
        # print(f'img_masks: {img_masks.shape}')

        # for img_id in range(batch_size):
        #     img_h, img_w, _ = img_metas[img_id]['img_shape']
        #     img_masks[img_id, :img_h, :img_w] = 0
        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                img_masks[img_id, cam_id, :img_h, :img_w] = 0

        # mlvl_masks (tuple[Tensor]): [bs, num_cams, H, W]
        mlvl_masks = []
        mlvl_positional_encodings = []

        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks,
                              size=feat.shape[-2:]).to(torch.bool))
            # mlvl_masks.append(
            #     F.interpolate(img_masks[None],
            #                   size=feat.shape[-2:]).to(torch.bool).squeeze(0))
        #     mlvl_positional_encodings.append(
        #         self.positional_encoding(mlvl_masks[-1]))

        # for mask in mlvl_masks:
        #     print(f'mask: {mask.shape}')
        query_embeds = self.query_embedding.weight

        # Operations for depth embedding
        depth_pos_embed = None
        pred_depth_map_logits = None
        weighted_depth = None
        if self.depth_predictor is not None:
            # pred_depth_map_logits: [B, N, D, H, W]
            # depth_pos_embed: [B, N, C, H, W]
            # weighted_depth: [B, N, H, W]
            pred_depth_map_logits, depth_pos_embed, weighted_depth = self.depth_predictor(
                mlvl_feats=mlvl_feats,
                mask=None,
                pos=None,
            )
            # print(f'pred_depth_map_logits: {pred_depth_map_logits.shape[:]}')

        if self.depth_gt_encoder is not None:
            assert gt_bboxes_3d is not None
            # gt_depth_maps with depth_gt_encoder: [B, N, H, W, num_depth_bins], dtype: torch.float32
            gt_depth_maps, gt_bboxes_2d = self.get_depth_map_and_gt_bboxes_2d(
                gt_bboxes_list=gt_bboxes_3d,
                img_metas=img_metas,
                target=False,
                device=mlvl_feats[0].device,
                depth_maps_down_scale=self.gt_depth_maps_down_scale,
            )
            gt_depth_maps = gt_depth_maps.to(mlvl_feats[0].device)
            # print(f'gt_depth_maps: {gt_depth_maps.shape}')
            # We do not need pred_depth_map_logits and weighted_depth to compute
            # loss_ddn when using gt_depth_maps
            _, depth_pos_embed, _ = self.depth_gt_encoder(
                mlvl_feats=mlvl_feats,
                mask=None,
                pos=None,
                gt_depth_maps=gt_depth_maps,
            )

        # hs: [num_layers, num_query, bs, embed_dims]
        # init_reference: [bs, num_query, 3]
        # inter_references: [num_layers, bs, num_query, 3]
        hs, init_reference, inter_references = self.transformer(
            mlvl_feats=mlvl_feats,
            mlvl_masks=mlvl_masks,
            query_embed=query_embeds,
            mlvl_pos_embeds=mlvl_positional_encodings,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            img_metas=img_metas,
            depth_pos_embed=depth_pos_embed,
        )
        # hs: [num_layers, bs, num_query, embed_dims]
        hs = hs.permute(0, 2, 1, 3)
        # hs: [num_layers, bs, num_query, embed_dims] without nan to avoid numeric errors
        hs = torch.nan_to_num(hs)

        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'pred_depth_map_logits': pred_depth_map_logits,
            'weighted_depth': weighted_depth,
        }
        return outs

    def get_depth_map_and_gt_bboxes_2d(
        self,
        gt_bboxes_list: List[LiDARInstance3DBoxes],
        img_metas: List[Dict[str, torch.Tensor]],
        target: bool = True,
        device: Optional[torch.device] = None,
        depth_maps_down_scale: int = 8,
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        """Get depth map and the 2D ground truth bboxes.

        Args:
            gt_bboxes_list: The ground truth list of `LiDARInstance3DBoxes`.
            img_metas: A list of dict containing the `lidar2img` tensor.
            target: If true, the returned `gt_depth_maps` will only have indices instead of another class dimension. Default: True.
            device: The device of the input image feature map.
            depth_maps_down_scale: The down scale of gt_depth_maps. Default: 8.

        Returns:
            gt_depth_maps: Thr ground truth depth maps with shape
                [batch, num_cameras, depth_map_H, depth_map_W] if `target` is true.
                Otherwise, the shape is [batch, num_cameras, depth_map_H, depth_map_W, num_depth_bins].
            gt_bboxes_2d: A list of list of tensor containing 2D ground truth bboxes (x, y, w, h)
                for each sample and each camera. Each tensor has shape [N_i, 4].
                Below is the brief explanation of a single batch:
                [B, N, ....]
                [[gt_bboxes_tensor_camera_0, gt_bboxes_tensor_camera_1, ..., gt_bboxes_tensor_camera_n] (sample 0),
                 [gt_bboxes_tensor_camera_0, gt_bboxes_tensor_camera_1, ..., gt_bboxes_tensor_camera_n] (sample 1),
                 ...].
        """
        img_H, img_W, _ = img_metas[0]['img_shape'][0]
        depth_map_H, depth_map_W = img_H // depth_maps_down_scale, img_W // depth_maps_down_scale

        gt_depth_maps = []
        gt_bboxes_2d = []

        resize_scale = img_H // depth_map_H
        assert resize_scale == img_W // depth_map_W

        for gt_bboxes, img_meta in zip(gt_bboxes_list, img_metas):
            # Check the gt_bboxes.tensor in case the empty bboxes
            # new version of mmdetection3d do not provide empety tensor
            if len(gt_bboxes.tensor) != 0:
                # [num_objects, 8, 3]
                gt_bboxes_corners = gt_bboxes.corners
                # [num_objects, 3].
                gt_bboxes_centers = gt_bboxes.gravity_center
            else:
                # [num_objects, 8, 3]
                gt_bboxes_corners = torch.empty([0, 8, 3], device=device)
                # [num_objects, 3].
                gt_bboxes_centers = torch.empty([0, 3], device=device)

            # [num_cameras, 3, 4]
            lidar2img = gt_bboxes_corners.new_tensor(img_meta['lidar2img'])[:, :3]
            assert tuple(lidar2img.shape) == (6, 3, 4)

            # Convert to homogeneous coordinate. [num_objects, 8, 4]
            gt_bboxes_corners = torch.cat([
                gt_bboxes_corners,
                gt_bboxes_corners.new_ones((*gt_bboxes_corners.shape[:-1], 1))
            ], dim=-1)

            # Convert to homogeneous coordinate. [num_objects, 4]
            gt_bboxes_centers = torch.cat([
                gt_bboxes_centers,
                gt_bboxes_centers.new_ones((*gt_bboxes_centers.shape[:-1], 1))
            ], dim=-1)

            # [num_cameras, num_objects, 8, 3]
            corners_uvd: torch.Tensor = torch.einsum('nij,mlj->nmli', lidar2img, gt_bboxes_corners)
            # [num_cameras, num_objects, 3]
            centers_uvd: torch.Tensor = torch.einsum('nij,mj->nmi', lidar2img, gt_bboxes_centers)
            # [num_cameras, num_objects]
            depth_targets = centers_uvd[..., 2]
            # [num_cameras, num_objects, 8]
            corners_depth_targets = corners_uvd[..., 2]

            # [num_cameras, num_objects, 8, 2]
            # fix for devide to zero
            corners_uv = corners_uvd[..., :2] / (corners_uvd[..., -1:] + 1e-8)

            depth_maps_all_camera = []
            gt_bboxes_all_camera = []
            # Generate depth maps and gt_bboxes for each camera.
            for corners_uv_per_camera, depth_target, corners_depth_target in zip(corners_uv, depth_targets, corners_depth_targets):
                # [num_objects, 8]
                visible = (corners_uv_per_camera[..., 0] > 0) & (corners_uv_per_camera[..., 0] < img_W) & \
                    (corners_uv_per_camera[..., 1] > 0) & (corners_uv_per_camera[..., 1] < img_H) & \
                    (corners_depth_target > 1)

                # [num_objects, 8]
                in_front = (corners_depth_target > 0.1)

                # [N,]
                # Filter num_objects in each camera
                mask = visible.any(dim=-1) & in_front.all(dim=-1)

                # [N, 8, 2]
                corners_uv_per_camera = corners_uv_per_camera[mask]

                # [N,]
                depth_target = depth_target[mask]

                # Resize corner for bboxes
                corners_uv_per_camera = (corners_uv_per_camera / resize_scale)

                # Clamp for depth
                corners_uv_per_camera[..., 0] = torch.clamp(corners_uv_per_camera[..., 0], 0, depth_map_W)
                corners_uv_per_camera[..., 1] = torch.clamp(corners_uv_per_camera[..., 1], 0, depth_map_H)

                # [N, 4]: (x_min, y_min, x_max, y_max)
                xy_min, _ = corners_uv_per_camera.min(dim=1)
                xy_max, _ = corners_uv_per_camera.max(dim=1)
                bboxes = torch.cat([xy_min, xy_max], dim=1).int()

                # [N, 4]: (x_min, y_min, w, h)
                bboxes[:, 2:] -= bboxes[:, :2]

                sort_by_depth = torch.argsort(depth_target, descending=True)
                bboxes = bboxes[sort_by_depth]
                depth_target = depth_target[sort_by_depth]

                # Fill into resize depth map = origin img /  resize_scale^2
                # depth_map = gt_bboxes_corners.new_zeros((img_H, img_W))
                depth_map = gt_bboxes_corners.new_zeros((depth_map_H, depth_map_W))

                for bbox, depth in zip(bboxes, depth_target):
                    x, y, w, h = bbox
                    depth_map[y:y + h, x:x + w] = depth

                gt_bboxes_all_camera.append(bboxes)
                depth_maps_all_camera.append(depth_map)

            # Visualizatioin for debugging
            # for i in range(6):
            #     print(f'i: {i+1}')
            #     heatmap = depth_maps_all_camera[i].detach().cpu().numpy().astype(np.uint8)
            #     print(f'type: {type(heatmap)}, shape: {heatmap.shape}')

            #     print(heatmap.min(), heatmap.max())
            #     heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
            #     print(f'heatmap: {heatmap}')
            #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_OCEAN)
            #     print(f'heatmap.shape: {heatmap.shape}')
            #     cv2.imwrite(f'/home/cytseng/dhm{i + 1}.jpg', heatmap)

            # exit()

            # [num_cameras, depth_map_H, depth_map_W]
            depth_maps_all_camera = torch.stack(depth_maps_all_camera)

            gt_depth_maps.append(depth_maps_all_camera)
            gt_bboxes_2d.append(gt_bboxes_all_camera)

        # [batch, num_cameras, depth_map_H, depth_map_W]
        gt_depth_maps = torch.stack(gt_depth_maps)
        # [batch, num_cameras, depth_map_H, depth_map_W], dtype: torch.long if `target` is true.
        # Otherwise [batch, num_cameras, depth_map_H, depth_map_W, num_depth_bins], dtype: torch.float
        gt_depth_maps = depth_utils.bin_depths(gt_depth_maps, **self.depth_bin_cfg, target=target)
        return gt_depth_maps, gt_bboxes_2d

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for 6 images(single sample), Outputs from the regression head with
                normalized coordinate format
                (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy).
                `[num_queries, 10]`
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :9]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each sample, with normalized coordinate
                normalized coordinate format
                (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy).
                `[num_queries, 10]`
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images.
                Shape `[bs, num_query, cls_out_channels]`.
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images(single sample), Outputs from the regression head with
                normalized coordinate format
                (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy).
                `[B, num_queries, 10]`
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # print(f'cls_scores: {(cls_scores.shape)}')
        # print(f'bbox_preds: {(bbox_preds.shape)}')
        num_imgs = cls_scores.size(0)

        # Operation to transfrom from torch.Tensor[B, num_query, 10] to
        # list(B) of torch.Tensor[num_query, 10]
        # list(B) of tensor cls_scores_list: [num_queries, 10]
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        # list(B) of tensor bbox_preds_list: [num_queries, 10]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        # print(f'cls_scores_list: {(cls_scores_list[0].shape)}')
        # print(f'bbox_preds_list: {(bbox_preds_list[0].shape)}')
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             preds_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): The ground truth list of
            `LiDARInstance3DBoxes`.

            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).

            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    `[num_layer, bs, num_query, cls_out_channels]`.
                all_bbox_preds(torch.Tensor): Sigmoid regression outputs
                    of all decode layers. Outputs from the regression head with
                    normalized coordinate format
                    (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy).
                    `[num_layer, B, num_queries, 10]`
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
                pred_depth_map_logits (Tensor): one hot encoding to represent the predict depth_maps,
                    in depth_gt_encoder default is None, defualt downsample to 1/32
                    `[B, N, D, H, W]`
                weighted_depth (Tensor): weight-sum value of predicted_depth_maps or gt_depth_maps
                    `[B, N, H, W]`
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        gt_bboxes_3d = gt_bboxes_list
        # Operation from prediction
        # all_cls_scores(torch.Tensor): [num_layer, B, num_queries, 10]
        all_cls_scores = preds_dicts['all_cls_scores']
        # all_bbox_preds(torch.Tensor): [num_layer, B, num_queries, 10]
        all_bbox_preds = preds_dicts['all_bbox_preds']
        # print(f'all_cls_scores: {(all_cls_scores.shape)}')
        # print(f'all_bbox_preds: {(all_bbox_preds.shape)}')

        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        if self.loss_depth:
            all_bbox_preds = self.compute_d_ave(
                img_metas=img_metas,
                weighted_depth=preds_dicts['weighted_depth'],
                all_bbox_preds=preds_dicts['all_bbox_preds'],
            )

        # Operation from GT
        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        # TODO: call `get_depth_map_and_gt_bboxes_2d` here and pass them into self.loss_single.
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        # list(6 layer) of list(B) of tensor gt_labels_list: [num_objects,]
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        # list(6 layer) of list(B) of tensor gt_labels_list: [num_objects, 9]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        # print(f'enc_cls_scores: {enc_cls_scores}')
        # enc_cls_scores is None
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        # Operation for depth_map surpervised
        if self.loss_ddn is not None:
            # loss from pred_depth_map_logits:
            # pred_depth_map_logits: [B, N, D+1, H, W]
            pred_depth_map_logits = preds_dicts['pred_depth_map_logits']
            assert pred_depth_map_logits is not None
            # print(f'pred_depth_map_logits: {pred_depth_map_logits.shape}')

            # get gt_depth_maps:
            # gt_depth_maps with depth_gt_encoder: [B, N, H, W, num_depth_bins], dtype: torch.float32
            # gt_depth_maps with normal depth encoder: [B, N, H, W], dtype: torch.long
            """
                gt_depth_maps: Thr ground truth depth maps with shape
                    [batch, num_cameras, depth_map_H, depth_map_W] if `target` is true.
                    Otherwise, the shape is [batch, num_cameras, depth_map_H, depth_map_W, num_depth_bins].
                gt_bboxes_2d: A list of list of tensor containing 2D ground truth bboxes(x, y, w, h)
                    for each sample and each camera. Each tensor has shape [N_i, 4].
                    Below is the brief explanation of a single batch:
                    [B, N, ....]
                    [[gt_bboxes_tensor_camera_0, gt_bboxes_tensor_camera_1, ..., gt_bboxes_tensor_camera_n](sample 0),
                        [gt_bboxes_tensor_camera_0, gt_bboxes_tensor_camera_1, ..., gt_bboxes_tensor_camera_n](sample 1),
                        ...].
            """

            gt_depth_maps, gt_bboxes_2d = self.get_depth_map_and_gt_bboxes_2d(
                gt_bboxes_list=gt_bboxes_3d,
                img_metas=img_metas,
                # TODO: `target` should be true after removing depth_gt_encoder
                target=(self.depth_gt_encoder is None),
                device=device,
                depth_maps_down_scale=self.depth_maps_down_scale,
            )
            gt_depth_maps = gt_depth_maps.to(device)
            # print(f'gt_depth_maps: {gt_depth_maps.shape}')

            loss_ddn = self.loss_ddn(
                depth_logits=pred_depth_map_logits,
                depth_target=gt_depth_maps,
                gt_bboxes_2d=gt_bboxes_2d,
            )
            loss_dict['loss_ddn'] = loss_ddn
        # print(f'loss_dict: {loss_dict}')
        return loss_dict

    def compute_d_ave(
        self,
        img_metas: List[Dict[str, torch.Tensor]],
        weighted_depth: torch.Tensor,
        all_bbox_preds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute d_ave for loss_depth from weighted_depth and return new all_bbox_preds
            with update cz

        Args:
            img_metas: A list of dict containing the `lidar2img` tensor.
            weighted_depth (torch.Tensor): weight-sum value of predicted_depth_maps or gt_depth_maps
                `[B, N, H, W]`
            all_bbox_preds(torch.Tensor): Sigmoid regression outputs
                of all decode layers. Outputs from the regression head with
                normalized coordinate format
                (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy).
                `[num_layer, B, num_queries, 10]`

        Returns:
            all_bbox_preds_new(torch.Tensor): new all_bbox_preds
                with update cz and normalized coordinate format
                (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy).
                `[num_layer, B, num_queries, 10]`
        """

        # weighted_depth: [B, N, H, W] -> [B*N, C, H, W]
        weighted_depth = weighted_depth.flatten(0, 1).unsqueeze(1)
        # lidar2img: [B, num_cameras, 4, 4]
        lidar2img = all_bbox_preds.new_tensor([img_meta['lidar2img'] for img_meta in img_metas])
        print(f'lidar2img: {lidar2img.shape}')

        # outputs_centers: [num_layer, B, num_query, 3] -> [B, num_query, num_layer, 3]
        outputs_centers = torch.cat(
            (all_bbox_preds[..., 0:2], all_bbox_preds[..., 4:5]), dim=-1).permute(1, 2, 0, 3)

        # convert to homogeneous coordinate. [batch, num_query, num_layer, 4]
        outputs_centers = torch.cat([
            outputs_centers,
            outputs_centers.new_ones((*outputs_centers.shape[:-1], 1))
        ], dim=-1)
        # print(f'outputs_centers: {outputs_centers.shape}')

        # uvd: [num_cameras, batch, num_query, num_layer, 3]
        uvd: torch.Tensor = torch.einsum('bnij,bqlj->nbqli', lidar2img[:, :, :3], outputs_centers)
        N, B, Q, L, _ = uvd.shape

        # uv: [num_cameras, batch, num_query, num_layer, 2]
        uv = uvd[..., :2] / (uvd[..., -1:] + 1e-8)
        img_H, img_W, _ = img_metas[0]['img_shape'][0]

        # normalize to [0, 1] -> [-1, 1]
        uv = (uv / uv.new_tensor([img_W, img_H]).reshape(1, 1, 1, 1, 2)) * 2 - 1

        # uv:[N, B, Q, L, 2] -> [B*N, Q, L, 2]
        uv = uv.flatten(0, 1).detach()
        # print(f'uv: {uv.shape}')
        # d_from_weighted_depth: [num_cameras*batch, num_query, num_layer, 1]
        d_from_weighted_depth = F.grid_sample(
            weighted_depth,
            uv,
            mode='bilinear',
            align_corners=True,
        )
        # d_from_weighted_depth: [num_cameras, batch, num_query, num_layer, 1]
        d_from_weighted_depth = d_from_weighted_depth.reshape(N, B, Q, L, -1)

        # d: [num_cameras, batch, num_query, num_layer, 1]
        d = uvd[..., 2:]

        # d_ave: [num_cameras, batch, num_query, num_layer, 1]
        d_ave = (d + d_from_weighted_depth) / 2
        print(f'd_ave: {d_ave.shape}')

        # uvd_new: [num_cameras, batch, num_query, num_layer, 3]
        uvd_new = torch.cat((uvd[..., :2], d_ave), dim=-1)

        # convert to homogeneous coordinate.
        # uvd_new: [num_cameras, batch, num_query, num_layer, 4]
        uvd_new = torch.cat([
            uvd_new,
            uvd_new.new_ones((*uvd_new.shape[:-1], 1))
        ], dim=-1)
        print(f'uvd_new: {uvd_new.shape}')

        # img2lidar: [B, num_cameras, 4, 4]
        img2lidar = torch.linalg.inv(lidar2img)
        print(f'img2lidar: {img2lidar.shape}')

        # TODO: Please check the transformation from img2lidar for new centers
        # and aggregate center from different camera in lidar-coordinate
        outputs_centers_new = torch.einsum('bnij,nbqlj->bqli', img2lidar[:, :, :3], uvd_new)
        print(f'outputs_centers_new: {outputs_centers_new.shape}')

        all_bbox_preds_new = all_bbox_preds
        all_bbox_preds_new[..., :2] = outputs_centers_new[..., :2]
        all_bbox_preds_new[..., 4:5] = outputs_centers_new[..., :-1]

        return all_bbox_preds_new

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, 9)
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
