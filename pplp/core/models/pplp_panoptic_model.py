import numpy as np

import tensorflow as tf

from pplp.builders import pplp_fc_layers_builder
from pplp.builders import pplp_loss_builder
from pplp.core import anchor_panoptic_projector
from pplp.core import anchor_panoptic_encoder
from pplp.core import box_3d_panoptic_encoder
from pplp.core import box_8c_panoptic_encoder
from pplp.core import box_4c_panoptic_encoder

from pplp.core import box_list
from pplp.core import box_list_ops

from pplp.core import model
from pplp.core import orientation_encoder
from pplp.core.models.rpn_pplp_panoptic_model import RpnModel

from tensorflow.contrib import slim

from transforms3d.quaternions import quat2mat, mat2quat
from transforms3d.euler import euler2mat, mat2euler


class PPLPModel(model.DetectionModel):
    ##############################
    # Keys for Predictions
    ##############################
    # Mini batch (mb) ground truth
    PRED_MB_CLASSIFICATIONS_GT = 'pplp_mb_classifications_gt'
    PRED_MB_OFFSETS_GT = 'pplp_mb_offsets_gt'
    PRED_MB_ORIENTATIONS_GT = 'pplp_mb_orientations_gt'
    PRED_MB_ORIENTATIONS_QUATS_GT = 'pplp_mb_orient_quats_gt'

    # Mini batch (mb) predictions
    PRED_MB_CLASSIFICATION_LOGITS = 'pplp_mb_classification_logits'
    PRED_MB_CLASSIFICATION_SOFTMAX = 'pplp_mb_classification_softmax'
    PRED_MB_OFFSETS = 'pplp_mb_offsets'
    PRED_MB_ANGLE_VECTORS = 'pplp_mb_angle_vectors'

    # Top predictions after BEV NMS
    PRED_TOP_CLASSIFICATION_LOGITS = 'pplp_top_classification_logits'
    PRED_TOP_CLASSIFICATION_SOFTMAX = 'pplp_top_classification_softmax'

    PRED_TOP_PREDICTION_ANCHORS = 'pplp_top_prediction_anchors'
    PRED_TOP_PREDICTION_BOXES_3D = 'pplp_top_prediction_boxes_3d'
    PRED_TOP_ORIENTATIONS = 'pplp_top_orientations'
    PRED_TOP_CROP_GROUP = 'pplp_top_crop_group'

    # Other box representations
    PRED_TOP_BOXES_8C = 'pplp_top_regressed_boxes_8c'
    PRED_TOP_BOXES_4C = 'pplp_top_prediction_boxes_4c'

    # Mini batch (mb) predictions (for debugging)
    PRED_MB_MASK = 'pplp_mb_mask'
    PRED_MB_POS_MASK = 'pplp_mb_pos_mask'
    PRED_MB_ANCHORS_GT = 'pplp_mb_anchors_gt'
    PRED_MB_CLASS_INDICES_GT = 'pplp_mb_gt_classes'

    PRED_MB_CROP_MASK = 'pplp_mb_crop_mask'
    PRED_MB_CROP_CLASS_INDICES_GT = 'pplp_mb_crop_gt_classes'

    # All predictions (for debugging)
    PRED_ALL_CLASSIFICATIONS = 'pplp_classifications'
    PRED_ALL_OFFSETS = 'pplp_offsets'
    PRED_ALL_ANGLE_VECTORS = 'pplp_angle_vectors'
    PRED_IMG_CROP_GT = 'rpn_image_crop_groundtruth'
    # PRED_IMG_CROP = 'rpn_image_crop'

    PRED_MAX_IOUS = 'pplp_max_ious'
    PRED_ALL_IOUS = 'pplp_anchor_ious'

    ##############################
    # Keys for Loss
    ##############################
    LOSS_FINAL_CLASSIFICATION = 'pplp_classification_loss'
    LOSS_FINAL_REGRESSION = 'pplp_regression_loss'

    # (for debugging)
    LOSS_FINAL_ORIENTATION = 'pplp_orientation_loss'
    LOSS_FINAL_LOCALIZATION = 'pplp_localization_loss'

    def __init__(self, model_config, train_val_test, dataset):
        """
        Args:
            model_config: configuration for the model
            train_val_test: "train", "val", or "test"
            dataset: the dataset that will provide samples and ground truth
        """

        # Sets model configs (_config)
        super(PPLPModel, self).__init__(model_config)

        self.dataset = dataset

        # Dataset config
        self._num_final_classes = self.dataset.num_classes + 1

        # Input config
        input_config = self._config.input_config
        self._bev_pixel_size = np.asarray([input_config.bev_dims_h,
                                           input_config.bev_dims_w])
        self._bev_depth = input_config.bev_depth

        self._img_pixel_size = np.asarray([input_config.img_dims_h,
                                           input_config.img_dims_w])
        self._img_depth = [input_config.img_depth]

        # AVOD config
        avod_config = self._config.avod_config
        self._proposal_roi_crop_size = \
            [avod_config.avod_proposal_roi_crop_size] * 2
        self._positive_selection = avod_config.avod_positive_selection
        self._nms_size = avod_config.avod_nms_size
        self._nms_iou_threshold = avod_config.avod_nms_iou_thresh
        self._path_drop_probabilities = self._config.path_drop_probabilities
        self._box_rep = avod_config.avod_box_representation

        if self._box_rep not in ['box_3d', 'box_8c', 'box_8co',
                                 'box_4c', 'box_4ca']:
            raise ValueError('Invalid box representation', self._box_rep)

        # Create the RpnModel
        self._rpn_pplp_panoptic_model = RpnModel(model_config, train_val_test, dataset)

        if train_val_test not in ["train", "val", "test"]:
            raise ValueError('Invalid train_val_test value,'
                             'should be one of ["train", "val", "test"]')
        self._train_val_test = train_val_test
        self._is_training = (self._train_val_test == 'train')

        self.sample_info = {}

    def build(self):
        rpn_pplp_panoptic_model = self._rpn_pplp_panoptic_model

        # Share the same prediction dict as RPN
        prediction_dict = rpn_pplp_panoptic_model.build()

        top_anchors = prediction_dict[RpnModel.PRED_TOP_ANCHORS]
        # top_anchors.shape is: (?, 6)
        ground_plane = rpn_pplp_panoptic_model.placeholders[RpnModel.PL_GROUND_PLANE]

        class_labels = rpn_pplp_panoptic_model.placeholders[RpnModel.PL_LABEL_CLASSES]

        img_feature_shifted = rpn_pplp_panoptic_model._img_feature_shifted
        if img_feature_shifted:
            top_img_feature = prediction_dict[RpnModel.PRED_TOP_IMG_FEATURE]  # shape=(?, 28, 28, 17), already tiled.
        # rgb_img_crop = prediction_dict[RpnModel.PRED_TOP_IMG_CROPS]  # shape=(?, 64, 64, 3)

        top_rpn_orient_pred = prediction_dict[RpnModel.PRED_TOP_ORIENT_PRED]  # shape=(?), for pplp orientation output, already tiled.
        top_rpn_crop_group = prediction_dict[RpnModel.PRED_TOP_CROP_GROUP]  # shape=(?), for pplp output selection, already tiled.

        # img_proposal_boxes_tf_order = prediction_dict[RpnModel.PRED_TOP_IMG_BBOX_TF_ORDER]  # shape=(?, 4) (y1, x1, y2, x2), tiled
        # img_proposal_boxes_tf_order = tf.Print(img_proposal_boxes_tf_order, ['-=-=-=-=-=-=-line 145(pplp) : img_proposal_boxes_tf_order =', img_proposal_boxes_tf_order], summarize=100)

        with tf.variable_scope('pplp_projection'):

            if self._config.expand_proposals_xz > 0.0:

                expand_length = self._config.expand_proposals_xz

                # Expand anchors along x and z
                with tf.variable_scope('expand_xz'):
                    expanded_dim_x = top_anchors[:, 3] + expand_length
                    expanded_dim_z = top_anchors[:, 5] + expand_length

                    expanded_anchors = tf.stack([
                        top_anchors[:, 0],
                        top_anchors[:, 1],
                        top_anchors[:, 2],
                        expanded_dim_x,
                        top_anchors[:, 4],
                        expanded_dim_z
                    ], axis=1)

                pplp_projection_in = expanded_anchors

            else:
                pplp_projection_in = top_anchors

            with tf.variable_scope('bev'):
                # Project top anchors into bev and image spaces
                bev_proposal_boxes, bev_proposal_boxes_norm = \
                    anchor_panoptic_projector.project_to_bev(
                        pplp_projection_in,
                        self.dataset.panoptic_utils.bev_extents)

                # Reorder projected boxes into [y1, x1, y2, x2]
                bev_proposal_boxes_tf_order = \
                    anchor_panoptic_projector.reorder_projected_boxes(
                        bev_proposal_boxes)
                bev_proposal_boxes_norm_tf_order = \
                    anchor_panoptic_projector.reorder_projected_boxes(
                        bev_proposal_boxes_norm)

                bev_feature_maps = rpn_pplp_panoptic_model.bev_feature_maps

            if not img_feature_shifted:
                with tf.variable_scope('img'):
                    image_shape = tf.cast(tf.shape(
                        rpn_pplp_panoptic_model.placeholders[RpnModel.PL_IMG_INPUT])[0:2],
                        tf.float32)
                    img_proposal_boxes, img_proposal_boxes_norm = \
                        anchor_panoptic_projector.tf_project_to_image_space(
                            pplp_projection_in,
                            rpn_pplp_panoptic_model.placeholders[RpnModel.PL_CALIB_P2],
                            image_shape)
                    # Only reorder the normalized img
                    img_proposal_boxes_norm_tf_order = \
                        anchor_panoptic_projector.reorder_projected_boxes(
                            img_proposal_boxes_norm)

                img_feature_maps = rpn_pplp_panoptic_model.img_mrcnn_feature_input_pl

        with tf.variable_scope('pplp_bev_conv'):
            bev_feature_maps_17 = slim.conv2d(
                bev_feature_maps,
                17, [1, 1],
                scope='bev_conv_17',
                normalizer_fn=slim.batch_norm,
                normalizer_params={
                    'is_training': self._is_training})

        if not (self._path_drop_probabilities[0] ==
                self._path_drop_probabilities[1] == 1.0):

            with tf.variable_scope('pplp_path_drop'):

                img_mask = rpn_pplp_panoptic_model.img_path_drop_mask
                bev_mask = rpn_pplp_panoptic_model.bev_path_drop_mask

                if img_feature_shifted:
                    top_img_feature = tf.multiply(top_img_feature,
                                                  img_mask)
                else:
                    img_feature_maps = tf.multiply(img_feature_maps,
                                                   img_mask)

                # bev_feature_maps = tf.multiply(bev_feature_maps,
                #                                bev_mask)
                bev_feature_maps_17 = tf.multiply(bev_feature_maps_17,
                                                  bev_mask)
        else:
            bev_mask = tf.constant(1.0)
            img_mask = tf.constant(1.0)

        # ROI Pooling
        with tf.variable_scope('pplp_roi_pooling'):
            def get_box_indices(boxes):
                proposals_shape = boxes.get_shape().as_list()
                if any(dim is None for dim in proposals_shape):
                    proposals_shape = tf.shape(boxes)
                ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
                multiplier = tf.expand_dims(
                    tf.range(start=0, limit=proposals_shape[0]), 1)
                return tf.reshape(ones_mat * multiplier, [-1])

            bev_boxes_norm_batches = tf.expand_dims(
                bev_proposal_boxes_norm, axis=0)

            # These should be all 0's since there is only 1 image
            tf_box_indices = get_box_indices(bev_boxes_norm_batches)

            # Do ROI Pooling on BEV
            # bev_feature_maps = tf.Print(bev_feature_maps, ['line 232(pplp): bev_feature_maps.shape =', tf.shape(bev_feature_maps)], summarize=1000)
            # bev_proposal_boxes_norm_tf_order = tf.Print(bev_proposal_boxes_norm_tf_order, ['line 233(pplp) : bev_proposal_boxes_norm_tf_order =', bev_proposal_boxes_norm_tf_order], summarize=1000)
            # tf_box_indices = tf.Print(tf_box_indices, ['line 234(pplp) : tf_box_indices =', tf_box_indices], summarize=1000)
            bev_rois = tf.image.crop_and_resize(
                bev_feature_maps_17,
                bev_proposal_boxes_norm_tf_order,
                tf_box_indices,
                self._proposal_roi_crop_size,
                # method='bilinear',
                # extrapolation_value=0,
                name='bev_rois'
                )

            if img_feature_shifted:
                # img_feature_input comes from self.img_mrcnn_feature_input_pl which dimension is: [?x56x56x17]
                img_rois = tf.image.resize_images(  # shape=(?, 28, 28, 17)
                    top_img_feature,  # shape=(?, 28, 28, 17)
                    self._proposal_roi_crop_size  # default [28, 28]
                    # method=ResizeMethod.BILINEAR,
                    # align_corners=True,
                    # preserve_aspect_ratio=False
                )
            else:
                # Do ROI Pooling on image
                img_rois = tf.image.crop_and_resize(
                    img_feature_maps,  # (?, height, width, 17)
                    img_proposal_boxes_norm_tf_order,
                    top_rpn_crop_group,
                    self._proposal_roi_crop_size,
                    name='img_rois')

        # Fully connected layers (Box Predictor)
        pplp_layers_config = self.model_config.layers_config.avod_config
        # img_rois = tf.Print(img_rois, ['line 259(pplp) : tf.shape(img_rois) =', tf.shape(img_rois)], summarize=100)
        # bev_rois = tf.Print(bev_rois, ['line 260(pplp) : tf.shape(bev_rois) =', tf.shape(bev_rois)], summarize=100)
        fc_layers_type = pplp_layers_config.WhichOneof('fc_layers')
        if fc_layers_type == 'fusion_fc_layers':
            fc_output_layers = \
                pplp_fc_layers_builder.build(
                    layers_config=pplp_layers_config,
                    input_rois=[bev_rois, img_rois],
                    input_weights=[bev_mask, img_mask],
                    num_final_classes=self._num_final_classes,
                    box_rep=self._box_rep,
                    top_anchors=top_anchors,
                    ground_plane=ground_plane,
                    is_training=self._is_training)
        else:
            fc_output_layers = \
                pplp_fc_layers_builder.build(
                    layers_config=pplp_layers_config,
                    input_rois=[bev_rois, img_rois],
                    input_weights=[bev_mask, img_mask],
                    num_final_classes=self._num_final_classes,
                    box_rep=self._box_rep,
                    top_anchors=top_anchors,
                    ground_plane=ground_plane,
                    is_training=self._is_training)

        all_cls_logits = \
            fc_output_layers[pplp_fc_layers_builder.KEY_CLS_LOGITS]
        all_offsets = fc_output_layers[pplp_fc_layers_builder.KEY_OFFSETS]

        with tf.variable_scope('softmax'):
            all_cls_softmax = tf.nn.softmax(
                all_cls_logits)

        ######################################################
        # Subsample mini_batch for the loss function
        ######################################################
        # Get the ground truth tensors
        anchors_gt = rpn_pplp_panoptic_model.placeholders[RpnModel.PL_LABEL_ANCHORS]  #shape = [?x6], (x,y,z,w,h,l)
        if self._box_rep in ['box_3d', 'box_4ca']:
            boxes_3d_gt = rpn_pplp_panoptic_model.placeholders[RpnModel.PL_LABEL_BOXES_3D]  #shape = [?x7], (x,y,z,w,h,l,ry)
        elif self._box_rep in ['box_8c', 'box_8co', 'box_4c']:
            boxes_3d_gt = rpn_pplp_panoptic_model.placeholders[RpnModel.PL_LABEL_BOXES_3D]
        else:
            raise NotImplementedError('Ground truth tensors not implemented')

        # Project anchor_gts to 2D bev
        with tf.variable_scope('pplp_gt_projection'):
            bev_anchor_boxes_gt, _ = anchor_panoptic_projector.project_to_bev(
                anchors_gt, self.dataset.panoptic_utils.bev_extents)
            # Now bev_anchor_boxes_gt is in the BEV, and the top left point is (0,0) point.
            # Camera x is not 0 any more.
            # bev_anchor_boxes_gt = tf.Print(bev_anchor_boxes_gt, ['-=-=-=-=-=-=-line 301(pplp) : self.dataset.panoptic_utils.bev_extents =', self.dataset.panoptic_utils.bev_extents], summarize=100)
            # bev_anchor_boxes_gt = tf.Print(bev_anchor_boxes_gt, ['-=-=-=-=-=-=-line 302(pplp) : bev_anchor_boxes_gt =', bev_anchor_boxes_gt], summarize=100)
            bev_anchor_boxes_gt_tf_order = \
                anchor_panoptic_projector.reorder_projected_boxes(bev_anchor_boxes_gt)

        with tf.variable_scope('pplp_bev_box_list'):
            # Convert BEV boxes to box_list format
            # bev_anchor_boxes_gt_tf_order = tf.Print(bev_anchor_boxes_gt_tf_order, ['-=-=-=-=-=-=-line 379(pplp) : bev_anchor_boxes_gt_tf_order =', bev_anchor_boxes_gt_tf_order], summarize=100)
            # bev_proposal_boxes_tf_order = tf.Print(bev_proposal_boxes_tf_order, ['-=-=-=-=-=-=-line 380(pplp) : bev_proposal_boxes_tf_order =', bev_proposal_boxes_tf_order], summarize=100)
            anchor_box_list_gt = box_list.BoxList(bev_anchor_boxes_gt_tf_order)
            anchor_box_list = box_list.BoxList(bev_proposal_boxes_tf_order)

            mb_mask, mb_class_label_indices, mb_gt_indices = \
                self.sample_mini_batch(
                    anchor_box_list_gt=anchor_box_list_gt,
                    anchor_box_list=anchor_box_list,
                    class_labels=class_labels)

        # Create classification one_hot vector
        with tf.variable_scope('pplp_one_hot_classes'):
            # mb_class_label_indices = tf.Print(mb_class_label_indices, ['-=-=-=-=-=-=-line 325(pplp) : mb_class_label_indices =', mb_class_label_indices], summarize=100)
            mb_classification_gt = tf.one_hot(
                mb_class_label_indices,
                depth=self._num_final_classes,
                on_value=1.0 - self._config.label_smoothing_epsilon,
                off_value=(self._config.label_smoothing_epsilon /
                           self.dataset.num_classes))
            # mb_classification_gt = tf.Print(mb_classification_gt, ['-=-=-=-=-=-=-line 332(pplp) : mb_classification_gt =', mb_classification_gt], summarize=100)

        # TODO: Don't create a mini batch in test mode
        # Mask predictions
        with tf.variable_scope('pplp_apply_mb_mask'):
            # Classification
            mb_classifications_logits = tf.boolean_mask(
                all_cls_logits, mb_mask)
            mb_classifications_softmax = tf.boolean_mask(
                all_cls_softmax, mb_mask)

            # Offsets
            mb_offsets = tf.boolean_mask(all_offsets, mb_mask)

        # Encode anchor offsets
        with tf.variable_scope('pplp_encode_mb_anchors'):
            # Get ground plane for box_4c conversion
            ground_plane = self._rpn_pplp_panoptic_model.placeholders[
                self._rpn_pplp_panoptic_model.PL_GROUND_PLANE]

            # Convert gt boxes_3d -> box_4c
            mb_boxes_3d_gt = tf.gather(boxes_3d_gt, mb_gt_indices)  # This is used for offset regression, so we use mb_gt_indices
            mb_boxes_4c_gt = box_4c_panoptic_encoder.tf_box_3d_to_box_4c(
                mb_boxes_3d_gt, ground_plane)

            # print('top_anchors = ', top_anchors)  # shape=(?, 6)
            # top_anchors = tf.Print(top_anchors, ['---------line 439(pplp) : top_anchors =', top_anchors], summarize=100)
            # Convert proposals: anchors -> box_3d -> box_4c
            proposal_boxes_3d = \
                box_3d_panoptic_encoder.anchors_to_box_3d(top_anchors, fix_lw=False)
            # proposal_boxes_3d.shape=(?, 7) but the last colunm is always 0
            proposal_boxes_4c = \
                box_4c_panoptic_encoder.tf_box_3d_to_box_4c(proposal_boxes_3d,
                                                            ground_plane)
            # proposal_boxes_4c.shape=(?, 10), now proposal_boxes_4c has not added with offsets yet
            # - box_4c format: [x1, x2, x3, x4, z1, z2, z3, z4, h1, h2]
            # - corners are in the xz plane, numbered clockwise starting at the top right
            # - h1 is the height above the ground plane to the bottom of the box
            # - h2 is the height above the ground plane to the top of the box

            # Get mini batch
            mb_boxes_4c = tf.boolean_mask(proposal_boxes_4c, mb_mask)
            mb_offsets_gt = box_4c_panoptic_encoder.tf_box_4c_to_offsets(
                mb_boxes_4c, mb_boxes_4c_gt)

        ######################################################
        # ROI summary images
        ######################################################
        avod_mini_batch_size = \
            self.dataset.panoptic_utils.mini_batch_panoptic_utils.avod_mini_batch_size
        with tf.variable_scope('bev_pplp_rois'):
            mb_bev_anchors_norm = tf.boolean_mask(
                bev_proposal_boxes_norm_tf_order, mb_mask)
            mb_bev_box_indices = tf.zeros_like(mb_gt_indices, dtype=tf.int32)

            # Show the ROIs of the BEV input density map
            # for the mini batch anchors
            bev_input_rois = tf.image.crop_and_resize(
                self._rpn_pplp_panoptic_model._bev_preprocessed,
                mb_bev_anchors_norm,
                mb_bev_box_indices,
                (32, 32))

            bev_input_roi_summary_images = tf.split(
                bev_input_rois, self._bev_depth, axis=3)
            tf.summary.image('bev_pplp_rois',
                             bev_input_roi_summary_images[-1],
                             max_outputs=avod_mini_batch_size)

        ######################################################
        # Final Predictions
        ######################################################
        # Get orientations from angle vectors
        pplp_layers_config = self.model_config.layers_config.avod_config
        fc_layers_type = pplp_layers_config.WhichOneof('fc_layers')

        # Apply offsets to regress proposals
        with tf.variable_scope('pplp_regression'):
            # Convert predictions box_4c -> box_3d
            prediction_boxes_4c = \
                box_4c_panoptic_encoder.tf_offsets_to_box_4c(proposal_boxes_4c,
                                                             all_offsets)
            # proposal_boxes_4c.shape=(?, 10), now proposal_boxes_4c has added with offsets
            # - box_4c format: [x1, x2, x3, x4, z1, z2, z3, z4, h1, h2]
            # - corners are in the xz plane, numbered clockwise starting at the top right
            # - h1 is the height above the ground plane to the bottom of the box
            # - h2 is the height above the ground plane to the top of the box

            prediction_boxes_3d = \
                box_4c_panoptic_encoder.tf_box_4c_to_box_3d(prediction_boxes_4c,
                                                            ground_plane)
            # prediction_boxes_3d.shape=(?, 7) but the last colunm is somehow random
            # Convert to anchor format for nms
            prediction_anchors = \
                box_3d_panoptic_encoder.tf_box_3d_to_anchor(prediction_boxes_3d)

        # Apply Non-oriented NMS in BEV
        with tf.variable_scope('pplp_nms'):
            bev_extents = self.dataset.panoptic_utils.bev_extents

            with tf.variable_scope('bev_projection'):
                # Project predictions into BEV
                pplp_bev_boxes, _ = anchor_panoptic_projector.project_to_bev(
                    prediction_anchors, bev_extents)
                pplp_bev_boxes_tf_order = \
                    anchor_panoptic_projector.reorder_projected_boxes(
                        pplp_bev_boxes)

            # Get top score from second column onward
            all_top_scores = tf.reduce_max(all_cls_logits[:, 1:], axis=1)

            # Apply NMS in BEV
            nms_indices = tf.image.non_max_suppression(
                pplp_bev_boxes_tf_order,
                all_top_scores,
                max_output_size=self._nms_size,
                iou_threshold=self._nms_iou_threshold)

            # Gather predictions from NMS indices
            top_classification_logits = tf.gather(all_cls_logits,
                                                  nms_indices)
            top_classification_softmax = tf.gather(all_cls_softmax,
                                                   nms_indices)
            top_prediction_anchors = tf.gather(prediction_anchors,
                                               nms_indices)

            top_prediction_boxes_3d = tf.gather(
                prediction_boxes_3d, nms_indices)
            top_prediction_boxes_4c = tf.gather(
                prediction_boxes_4c, nms_indices)
            # nms_indices = tf.Print(nms_indices, ['-=-=-=-=-=-=-line 542(pplp) : nms_indices =', nms_indices], summarize=100)
            # top_rpn_crop_group = tf.Print(top_rpn_crop_group, ['-=-=-=-=-=-=-line 543(pplp) : top_rpn_crop_group =', top_rpn_crop_group], summarize=100)
            top_orient_pred = tf.gather(top_rpn_orient_pred, nms_indices)
            top_crop_group = tf.gather(top_rpn_crop_group, nms_indices)
            # top_crop_group = tf.Print(top_crop_group, ['-=-=-=-=-=-=-line 546(pplp) : top_crop_group =', top_crop_group], summarize=100)
            # top_classification_logits = tf.Print(top_classification_logits, ['-=-=-=-=-=-=-line 547(pplp) : top_classification_logits =', top_classification_logits], summarize=100)
            # top_classification_softmax = tf.Print(top_classification_softmax, ['-=-=-=-=-=-=-line 548(pplp) : top_classification_softmax =', top_classification_softmax], summarize=100)
            # top_prediction_anchors = tf.Print(top_prediction_anchors, ['-=-=-=-=-=-=-line 549(pplp) : top_prediction_anchors =', top_prediction_anchors], summarize=100)
            # top_prediction_boxes_3d = tf.Print(top_prediction_boxes_3d, ['-=-=-=-=-=-=-line 550(pplp) : top_prediction_boxes_3d =', top_prediction_boxes_3d], summarize=100)
            # top_prediction_boxes_4c = tf.Print(top_prediction_boxes_4c, ['-=-=-=-=-=-=-line 551(pplp) : top_prediction_boxes_4c =', top_prediction_boxes_4c], summarize=100)

            pick_highest_within_group = False
            if pick_highest_within_group:
                # Now we sort all the results by their scores (top_classification_softmax[:, 1]),Then pick the highest score for each group.
                score_ranked_index = tf.nn.top_k(top_classification_softmax[:, 1], tf.size(top_classification_softmax[:, 1]))
                top_classification_logits = tf.gather(top_classification_logits, score_ranked_index.indices)
                top_classification_softmax = tf.gather(top_classification_softmax, score_ranked_index.indices)
                top_prediction_anchors = tf.gather(top_prediction_anchors, score_ranked_index.indices)
                top_prediction_boxes_3d = tf.gather(top_prediction_boxes_3d, score_ranked_index.indices)
                top_prediction_boxes_4c = tf.gather(top_prediction_boxes_4c, score_ranked_index.indices)
                top_orient_pred = tf.gather(top_orient_pred, score_ranked_index.indices)
                top_crop_group = tf.gather(top_crop_group, score_ranked_index.indices)

                # top_classification_softmax = tf.Print(top_classification_softmax, ['-=-=-=-=-=-=-line 563(pplp) : top_classification_softmax =', top_classification_softmax], summarize=100)
                # top_crop_group = tf.Print(top_crop_group, ['-=-=-=-=-=-=-line 564(pplp) : top_crop_group =', top_crop_group], summarize=100)

                def find_unrepeat_first(input):
                    # Given an input of integer array, find the first unrepeat indices.
                    # Example: input = [3 3 0 0 2 3 3 2 1 1 0 0 1 0 0 3 3 0 3 3 3 3 3]
                    # the first '0' is the 2th number, the first '1' is the 8th number,
                    # the first '2' is the 4th number, the first '3' is the 0th number,
                    # So the output should be:
                    # output = [2 8 4 0]

                    input = np.asarray(input, dtype=np.int)
                    print('input = ', input)
                    unique_list, indices = np.unique(input, return_index=True)
                    print('unique_list = ', unique_list)
                    print('indices = ', indices)

                    return indices
                top_element_in_group = tf.py_func(find_unrepeat_first, [top_crop_group], tf.int64)

                top_classification_logits = tf.gather(top_classification_logits, top_element_in_group)
                top_classification_softmax = tf.gather(top_classification_softmax, top_element_in_group)
                top_prediction_anchors = tf.gather(top_prediction_anchors, top_element_in_group)
                top_prediction_boxes_3d = tf.gather(top_prediction_boxes_3d, top_element_in_group)
                top_prediction_boxes_4c = tf.gather(top_prediction_boxes_4c, top_element_in_group)
                top_orient_pred = tf.gather(top_orient_pred, top_element_in_group)
                top_crop_group = tf.gather(top_crop_group, top_element_in_group)

            # The orientation from OrientNet are all translated into camera center,
            # so we have to translate them back to their 3D orientation according
            # to their predicted 3D podisition.
            def rotate_orientations(orient, boxes_3d):
                final_boxes_3d = boxes_3d
                # print('boxes_3d =', boxes_3d)
                for j in range(len(orient)):
                    if np.isnan(boxes_3d[j][0]):
                        # If this result is from an invalid Mask RCNN crop, then we
                        # plot the 3D location out of the valid area.
                        final_boxes_3d[j, :] = [0.0, 0.0, -10.0, 1.0, 1.0, 1.0, 0.0]
                    else:
                        # use translation to get apparant orientation of object in the
                        # camera view.
                        pitch_angle = np.arctan2(boxes_3d[j][0], boxes_3d[j][2]+boxes_3d[j][5]/2);
                        roll_angle = -np.arctan2(boxes_3d[j][1], boxes_3d[j][2]+boxes_3d[j][5]/2);
                        # When the camera coordinate is defined as:
                        # x points to the ground; y points down to the floor;
                        # z shoot out from the camera.
                        # Then angle definition should be:
                        # (make sure the definition is the same in all orientation-related files!!)
                        # yaw_angle = np.arctan2(label_boxes_3d[j][0], label_boxes_3d[j][2]);
                        # pitch_angle = np.arctan2(label_boxes_3d[j][1]-label_boxes_3d[j][5]/2, label_boxes_3d[j][2]);
                        # print('For people #', j, ': pitch_angle=', pitch_angle, '; roll_angle=', roll_angle, '; yaw_angle=', label_boxes_3d[j][6])
                        rot = euler2mat(roll_angle, pitch_angle, 0, axes='rxyz')
                        R_old = euler2mat(0, orient[j], 0, axes='rxyz')
                        R_new = np.dot(rot, R_old)
                        new_orient = mat2euler(R_new, axes='ryzx')
                        # Yes, very weird! When we tranlate from euler2quat, we use axes='rxyz', now we should use axes='ryzx'.
                        # For annimation: https://quaternions.online/
                        # print('new_orient = ', new_orient)
                        # When pred_boxes_3d[i ,:] are all np.nan, new_orient =  (-0.0, nan, nan)
                        final_boxes_3d[j, 6] = new_orient[0]  # pick the first index since we use y-z-x
                return final_boxes_3d
            top_prediction_boxes_3d = tf.py_func(rotate_orientations, [top_orient_pred, top_prediction_boxes_3d], tf.float32)
            top_orient_pred = top_prediction_boxes_3d[:, 6]

            # prediction_dict[self.PRED_IMG_CROP] = rgb_img_crop  # shape=(?, 64, 64, 3)

            prediction_dict[self.PRED_TOP_PREDICTION_BOXES_3D] = \
                top_prediction_boxes_3d
            prediction_dict[self.PRED_TOP_BOXES_4C] = top_prediction_boxes_4c

        if self._train_val_test in ['train', 'val']:
            # Additional entries are added to the shared prediction_dict
            # Mini batch predictions
            prediction_dict[self.PRED_MB_CLASSIFICATION_LOGITS] = \
                mb_classifications_logits
            prediction_dict[self.PRED_MB_CLASSIFICATION_SOFTMAX] = \
                mb_classifications_softmax
            prediction_dict[self.PRED_MB_OFFSETS] = mb_offsets

            # Mini batch ground truth
            prediction_dict[self.PRED_MB_CLASSIFICATIONS_GT] = \
                mb_classification_gt  # shape = [?x2] for all candidate anchors
            prediction_dict[self.PRED_MB_OFFSETS_GT] = mb_offsets_gt

            # Top NMS predictions
            prediction_dict[self.PRED_TOP_CLASSIFICATION_LOGITS] = \
                top_classification_logits
            prediction_dict[self.PRED_TOP_CLASSIFICATION_SOFTMAX] = \
                top_classification_softmax

            prediction_dict[self.PRED_TOP_PREDICTION_ANCHORS] = \
                top_prediction_anchors

            prediction_dict[self.PRED_TOP_ORIENTATIONS] = top_orient_pred

            prediction_dict[self.PRED_TOP_CROP_GROUP] = top_crop_group

            # Mini batch predictions (for debugging)
            prediction_dict[self.PRED_MB_MASK] = mb_mask
            # prediction_dict[self.PRED_MB_POS_MASK] = mb_pos_mask
            prediction_dict[self.PRED_MB_CLASS_INDICES_GT] = \
                mb_class_label_indices  # shape = [?] for all candidate anchors

            # All predictions (for debugging)
            prediction_dict[self.PRED_ALL_CLASSIFICATIONS] = \
                all_cls_logits
            prediction_dict[self.PRED_ALL_OFFSETS] = all_offsets

            # Path drop masks (for debugging)
            prediction_dict['bev_mask'] = bev_mask
            prediction_dict['img_mask'] = img_mask

        else:
            # self._train_val_test == 'test'
            prediction_dict[self.PRED_TOP_CLASSIFICATION_SOFTMAX] = \
                top_classification_softmax
            prediction_dict[self.PRED_TOP_PREDICTION_ANCHORS] = \
                top_prediction_anchors

        return prediction_dict

    def sample_mini_batch(self, anchor_box_list_gt, anchor_box_list,
                          class_labels):

        with tf.variable_scope('pplp_create_mb_mask'):
            # Get IoU for every anchor
            all_ious = box_list_ops.iou(anchor_box_list_gt, anchor_box_list)
            # all_ious = tf.Print(all_ious, ['-=-=-=-=-=-=-line 753(pplp) : all_ious =', all_ious], summarize=100)
            # all_ious = tf.Print(all_ious, ['-=-=-=-=-=-=-line 754(pplp) : tf.shape(all_ious) =', tf.shape(all_ious)], summarize=100) #  tf.shape(all_ious) = [4 265]
            max_ious = tf.reduce_max(all_ious, axis=0)
            # max_ious = tf.Print(max_ious, ['-=-=-=-=-=-=-line 756(pplp) : max_ious =', max_ious], summarize=100)
            # max_ious =[0 0 0 0 0 0 0.225453332 0 0 0 0 0 0 0 0 0 0 0 0 0 0.417117327 0 0.404093683 0 0 0.00502740964 0 0.199674159 0 0.288573176 0 0 0.0159720983 0 0 0 0.169192255 0.00572701544 0 0 0 0 0 0 0.0461278595 0 0 0 0 0 0 0 0 0.128080159 0 0 0 0 0 0.615373969 0 0 0 0 0 0 0 0 0 0 0 0 0.309623659 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.0571997948 0 0 0 0 0 0 0 0 0 0 0 0.106632032...]
            # max_ious = tf.Print(max_ious, ['-=-=-=-=-=-=-line 757(pplp) : tf.shape(max_ious) =', tf.shape(max_ious)], summarize=100) #  tf.shape(max_ious) = [265]
            max_iou_indices = tf.argmax(all_ious, axis=0)
            # max_iou_indices = tf.Print(max_iou_indices, ['-=-=-=-=-=-=-line 759(pplp) : max_iou_indices =', max_iou_indices], summarize=100)
            # max_iou_indices =[0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 2 0 0 2 0 3 0 0 0 0 2 0 0 0 2 2 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 3 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0...]
            # max_iou_indices = tf.Print(max_iou_indices, ['-=-=-=-=-=-=-line 760(pplp) : tf.shape(max_iou_indices) =', tf.shape(max_iou_indices)], summarize=100) #  tf.shape(max_iou_indices) = [265]

            # Sample a pos/neg mini-batch from anchors with highest IoU match
            mini_batch_panoptic_utils = self.dataset.panoptic_utils.mini_batch_panoptic_utils
            mb_mask, mb_pos_mask = mini_batch_panoptic_utils.sample_avod_mini_batch(
                max_ious)
            # mb_pos_mask = tf.Print(mb_pos_mask, ['-=-=-=-=-=-=-line 768(pplp) : tf.shape(mb_pos_mask) =', tf.shape(mb_pos_mask)], summarize=100)
            # mb_pos_mask = tf.Print(mb_pos_mask, ['-=-=-=-=-=-=-line 769(pplp) : mb_pos_mask =', mb_pos_mask], summarize=1000)
            # mb_mask = tf.Print(mb_mask, ['-=-=-=-=-=-=-line 770(pplp) : tf.shape(mb_mask) =', tf.shape(mb_mask)], summarize=100)
            # mb_mask = tf.Print(mb_mask, ['-=-=-=-=-=-=-line 771(pplp) : mb_mask =', mb_mask], summarize=1000)
            # max_iou_indices = tf.Print(max_iou_indices, ['line 687(pplp) : max_iou_indices =', max_iou_indices], summarize=100)

            # class_labels: a tensor of shape [num_of_classes] indicating the
            #     class labels as indices. For instance indices=[0, 1, 2, 3]
            #     indicating 'background, car, pedestrian, cyclist' etc.
            # For pedestrian training only, class_labels =[1]

            mb_class_label_indices = mini_batch_panoptic_utils.mask_class_label_indices(
                mb_pos_mask, mb_mask, max_iou_indices, class_labels)
            # mb_class_label_indices = tf.Print(mb_class_label_indices, ['-=-=-=-=-=-=-line 781(pplp) : tf.shape(mb_class_label_indices) =', tf.shape(mb_class_label_indices)], summarize=100) #  tf.shape(mb_gt_indices) = [265]
            # mb_class_label_indices = tf.Print(mb_class_label_indices, ['-=-=-=-=-=-=-line 782(pplp) : mb_class_label_indices =', mb_class_label_indices], summarize=1000) #  tf.shape(mb_gt_indices) = [265]
            # mb_class_label_indices : a tensor of boolean mask for class label
            #     indices. This gives the indices for the positive classes and
            #     masks negatives or background classes by zero's.

            mb_gt_indices = tf.boolean_mask(max_iou_indices, mb_mask)
            # mb_gt_indices = tf.Print(mb_gt_indices, ['-=-=-=-=-=-=-line 786(pplp) : tf.shape(mb_gt_indices) =', tf.shape(mb_gt_indices)], summarize=100) #  tf.shape(mb_gt_indices) = [265]
            # mb_gt_indices = tf.Print(mb_gt_indices, ['-=-=-=-=-=-=-line 787(pplp) : mb_gt_indices =', mb_gt_indices], summarize=1000) #  tf.shape(mb_gt_indices) = [265]

        return mb_mask, mb_class_label_indices, mb_gt_indices

    def create_feed_dict(self):
        feed_dict = self._rpn_pplp_panoptic_model.create_feed_dict()
        self.sample_info = self._rpn_pplp_panoptic_model.sample_info
        return feed_dict

    def loss(self, prediction_dict):
        # Note: The loss should be using mini-batch values only
        loss_dict, rpn_loss = self._rpn_pplp_panoptic_model.loss(prediction_dict)
        losses_output = pplp_loss_builder.build(self, prediction_dict)

        classification_loss = \
            losses_output[pplp_loss_builder.KEY_CLASSIFICATION_LOSS]

        final_reg_loss = losses_output[pplp_loss_builder.KEY_REGRESSION_LOSS]

        pplp_loss = losses_output[pplp_loss_builder.KEY_PPLP_LOSS]

        offset_loss_norm = \
            losses_output[pplp_loss_builder.KEY_OFFSET_LOSS_NORM]

        loss_dict.update({self.LOSS_FINAL_CLASSIFICATION: classification_loss})
        loss_dict.update({self.LOSS_FINAL_REGRESSION: final_reg_loss})

        # Add localization and orientation losses to loss dict for plotting
        loss_dict.update({self.LOSS_FINAL_LOCALIZATION: offset_loss_norm})

        ang_loss_norm = losses_output.get(
            pplp_loss_builder.KEY_ANG_LOSS_NORM)
        if ang_loss_norm is not None:
            loss_dict.update({self.LOSS_FINAL_ORIENTATION: ang_loss_norm})

        with tf.variable_scope('model_total_loss'):
            total_loss = rpn_loss + pplp_loss

        return loss_dict, total_loss
