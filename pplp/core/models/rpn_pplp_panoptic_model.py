# Add this block for ROS python conflict
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    sys.path.remove('$HOME/segway_kinetic_ws/devel/lib/python2.7/dist-packages')
except ValueError:
    pass

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from pplp.builders import feature_extractor_builder
from pplp.core import anchor_panoptic_encoder
from pplp.core import anchor_panoptic_filter
from pplp.core import anchor_panoptic_projector
from pplp.core import box_3d_panoptic_encoder
from pplp.core import constants
from pplp.core import losses
from pplp.core import model
from pplp.core import summary_utils
from pplp.core.anchor_generators import grid_anchor_3d_generator
from pplp.datasets.panoptic import panoptic_aug

from wavedata.tools.obj_detection import evaluation
from transforms3d.quaternions import quat2mat, mat2quat
from transforms3d.euler import euler2mat, mat2euler

import time
import cv2


class RpnModel(model.DetectionModel):
    ##############################
    # Keys for Placeholders
    ##############################
    PL_BEV_INPUT = 'bev_input_pl'
    PL_IMG_INPUT = 'img_input_pl'
    PL_IMG_MASK_INPUT = 'img_mask_input_pl'
    PL_IMG_MASK_FULL_INPUT = 'img_mask_full_input_pl'
    PL_IMG_MRCNN_FEATURE_INPUT = 'img_mrcnn_feature_input_pl'
    PL_IMG_MRCNN_FEATURE_FULL_INPUT = 'img_mrcnn_feature_full_input_pl'
    PL_IMG_MRCNN_BBOX_INPUT = 'img_mrcnn_bbox_input_pl'
    PL_IMG_MRCNN_KEYPOINTS_INPUT = 'img_mrcnn_keypoints_input_pl'
    PL_IMG_MRCNN_KEYPOINTS_NORM_INPUT = 'img_mrcnn_keypoints_norm_input_pl'
    PL_IMG_MASK_SHIFTED = 'img_mask_shifted_pl'
    PL_IMG_MRCNN_FEATURE_SHIFTED = 'img_mrcnn_feature_shifted_pl'
    PL_IMG_MRCNN_BBOX_SHIFTED = 'img_mrcnn_bbox_shifted_pl'
    PL_IMG_REPEAT_TIMES = 'img_repeat_times_pl'
    PL_ANCHORS = 'anchors_pl'

    PL_ORIENT_PRED = 'orient_pred_pl'
    PL_CROP_GROUP = 'crop_group_pl'
    PL_BEV_ANCHORS = 'bev_anchors_pl'
    PL_BEV_ANCHORS_NORM = 'bev_anchors_norm_pl'
    PL_IMG_ANCHORS = 'img_anchors_pl'
    PL_IMG_ANCHORS_NORM = 'img_anchors_norm_pl'
    PL_LABEL_ANCHORS = 'label_anchors_pl'
    PL_LABEL_BOXES_3D = 'label_boxes_3d_pl'
    PL_LABEL_BOXES_2D = 'label_boxes_2d_pl'
    PL_LABEL_BOXES_2D_NORM = 'label_boxes_2d_norm_pl'
    PL_LABEL_CLASSES = 'label_classes_pl'
    PL_LABEL_BOXES_QUATERNION = 'label_boxes_quat_pl'

    PL_ANCHOR_IOUS = 'anchor_ious_pl'
    PL_ANCHOR_OFFSETS = 'anchor_offsets_pl'
    PL_ANCHOR_CLASSES = 'anchor_classes_pl'

    # Sample info, including keys for projection to image space
    # (e.g. camera matrix, image index, etc.)
    PL_CALIB_P2 = 'frame_calib_p2'
    PL_IMG_IDX = 'current_img_idx'
    PL_GROUND_PLANE = 'ground_plane'

    ##############################
    # Keys for Predictions
    ##############################
    PRED_ANCHORS = 'rpn_anchors'

    PRED_MB_OBJECTNESS_GT = 'rpn_mb_objectness_gt'
    PRED_MB_OFFSETS_GT = 'rpn_mb_offsets_gt'

    PRED_MB_MASK = 'rpn_mb_mask'
    PRED_MB_OBJECTNESS = 'rpn_mb_objectness'
    PRED_MB_OFFSETS = 'rpn_mb_offsets'

    PRED_TOP_INDICES = 'rpn_top_indices'
    PRED_TOP_ANCHORS = 'rpn_top_anchors'
    PRED_TOP_OBJECTNESS_SOFTMAX = 'rpn_top_objectness_softmax'

    PRED_TOP_IMG_FEATURE = 'rpn_top_img_feature'
    PRED_TOP_IMG_CROPS = 'rpn_top_img_crops'
    PRED_TOP_IMG_BBOX_TF_ORDER = 'rpn_top_img_bbox_tf_order'
    PRED_TOP_ORIENT_PRED = 'rpn_top_orient_pred'
    PRED_TOP_CROP_GROUP = 'rpn_top_crop_group'

    ##############################
    # Keys for Loss
    ##############################
    LOSS_RPN_OBJECTNESS = 'rpn_objectness_loss'
    LOSS_RPN_REGRESSION = 'rpn_regression_loss'

    def __init__(self, model_config, train_val_test, dataset):
        """
        Args:
            model_config: configuration for the model
            train_val_test: "train", "val", or "test"
            dataset: the dataset that will provide samples and ground truth
        """

        # Sets model configs (_config)
        super(RpnModel, self).__init__(model_config)

        if train_val_test not in ["train", "val", "test"]:
            raise ValueError('Invalid train_val_test value,'
                             'should be one of ["train", "val", "test"]')
        self._train_val_test = train_val_test

        self._is_training = (self._train_val_test == 'train')
        self._img_feature_shifted = False  # This flag helps you to choose the
        # input image feature. Whether it is shifted into small pieces or it is
        # resized to full image size for each pedestrian.

        # Input config
        input_config = self._config.input_config
        self._bev_pixel_size = np.asarray([input_config.bev_dims_h,
                                           input_config.bev_dims_w])
        self._bev_depth = input_config.bev_depth

        self._img_pixel_size = np.asarray([input_config.img_dims_h,
                                           input_config.img_dims_w])
        self._img_depth = input_config.img_depth

        # Rpn config
        rpn_config = self._config.rpn_config
        self._proposal_roi_crop_size = \
            [rpn_config.rpn_proposal_roi_crop_size] * 2
        # rpn_config.rpn_proposal_roi_crop_size = 7
        # self._proposal_roi_crop_size = [7, 7]
        self._fusion_method = rpn_config.rpn_fusion_method

        if self._train_val_test in ["train", "val"]:
            self._nms_size = rpn_config.rpn_train_nms_size
        else:
            self._nms_size = rpn_config.rpn_test_nms_size

        self._nms_iou_thresh = rpn_config.rpn_nms_iou_thresh

        # Feature Extractor Nets
        self._bev_feature_extractor = \
            feature_extractor_builder.get_extractor(
                self._config.layers_config.bev_feature_extractor)
        self._img_feature_extractor = \
            feature_extractor_builder.get_extractor(
                self._config.layers_config.img_feature_extractor)

        # Network input placeholders
        self.placeholders = dict()

        # Inputs to network placeholders
        self._placeholder_inputs = dict()

        # Information about the current sample
        self.sample_info = dict()

        # Dataset
        self.dataset = dataset
        self.dataset.train_val_test = self._train_val_test
        self._area_extents = self.dataset.panoptic_utils.area_extents
        self._bev_extents = self.dataset.panoptic_utils.bev_extents
        self._cluster_sizes, _ = self.dataset.get_cluster_info()
        self._anchor_strides = self.dataset.panoptic_utils.anchor_strides
        self._anchor_generator = \
            grid_anchor_3d_generator.GridAnchor3dGenerator()

        self._path_drop_probabilities = self._config.path_drop_probabilities
        self._train_on_all_samples = self._config.train_on_all_samples
        self._eval_all_samples = self._config.eval_all_samples
        # Overwrite the dataset's variable with the config
        self.dataset.train_on_all_samples = self._train_on_all_samples

        if self._train_val_test in ["val", "test"]:
            # Disable path-drop, this should already be disabled inside the
            # evaluator, but just in case.
            self._path_drop_probabilities[0] = 1.0
            self._path_drop_probabilities[1] = 1.0

    def _add_placeholder(self, dtype, shape, name):
        placeholder = tf.placeholder(dtype, shape, name)
        self.placeholders[name] = placeholder
        return placeholder

    def _set_up_input_pls(self):
        """Sets up input placeholders by adding them to self._placeholders.
        Keys are defined as self.PL_*.
        """
        # Combine config data
        bev_dims = np.append(self._bev_pixel_size, self._bev_depth)

        with tf.variable_scope('bev_input'):
            # Placeholder for BEV image input, to be filled in with feed_dict
            bev_input_placeholder = self._add_placeholder(tf.float32, bev_dims,
                                                          self.PL_BEV_INPUT)
            # bev_input_placeholder shape=(700, 800, 1)

            self._bev_input_batches = tf.expand_dims(  # shape=(1, 700, 800, 1)
                bev_input_placeholder, axis=0)

            self._bev_preprocessed = \
                self._bev_feature_extractor.preprocess_input(  # shape=(1, 700, 800, 1)
                    self._bev_input_batches, self._bev_pixel_size)

            # Summary Images
            bev_summary_images = tf.split(  # shape=(700, 800, 1)
                bev_input_placeholder, self._bev_depth, axis=2)
            tf.summary.image("bev_maps", bev_summary_images,
                             max_outputs=self._bev_depth)

        with tf.variable_scope('img_input'):
            # Take variable size input images
            img_input_placeholder = self._add_placeholder(  # shape=(?, ?, 3)
                tf.float32,
                [None, None, self._img_depth],
                self.PL_IMG_INPUT)

            self._img_input_batches = tf.expand_dims(  # shape=(1, ?, ?, 3)
                img_input_placeholder, axis=0)

        # ------------- For feature maps directly from Mask RCNN: -------------
        if self._img_feature_shifted:
            with tf.variable_scope('img_mrcnn_keypoints_norm_input'):
                # Placeholder for mrcnn keypoints (pixel value in the whole picture)
                # keypoints: (dtype=float32), [N, 17, [x,y]]
                self.img_mrcnn_keypoints_input_pl = self._add_placeholder(  # shape=(?, 17, 2)
                    tf.float32,
                    [None, 17, 2],
                    self.PL_IMG_MRCNN_KEYPOINTS_NORM_INPUT)

        with tf.variable_scope('img_mrcnn_mask_input'):
            # Placeholder for image masks or full_masks
            # full_masks: (dtype=uint8), [batch, height, width, N] Instance masks
            # masks: array(dtype=float32), [M, 28, 28]
            # N is the number of people detected by MaskRCNN
            # M is the number of all candidates
            if self._img_feature_shifted:
                img_mrcnn_mask_input_placeholder = self._add_placeholder(  # shape=(M, 28, 28)
                    tf.float32,
                    [None, 28, 28],
                    self.PL_IMG_MASK_INPUT)

                self.img_mrcnn_mask_input_tile_17_pl = tf.tile(  # shape=(M, 28, 28, 17)
                    tf.reshape(img_mrcnn_mask_input_placeholder, [-1, 28, 28, 1]),
                    [1, 1, 1, 17],
                    name=None)
            else:
                img_mrcnn_mask_input_placeholder = self._add_placeholder(  # shape=(N, height, width)
                    tf.float32,
                    [None, self._img_pixel_size[0], self._img_pixel_size[1]],
                    self.PL_IMG_MASK_FULL_INPUT)
                self.img_mrcnn_mask_input_tile_17_pl = tf.tile(  # shape=(N, height, width, 17)
                    tf.reshape(img_mrcnn_mask_input_placeholder, [-1, self._img_pixel_size[0], self._img_pixel_size[1], 1]),
                    [1, 1, 1, 17],
                    name=None)

        with tf.variable_scope('img_mrcnn_feature_input'):
            if self._img_feature_shifted:
                # Placeholder for image keypoint features:
                # features: [M, 28, 28, num_keypoints]
                img_mrcnn_feature_input_pl = self._add_placeholder(
                    tf.float32,
                    [None, 28, 28, 17],
                    self.PL_IMG_MRCNN_FEATURE_INPUT)
            else:
                # Placeholder for image keypoint features(Full size):
                # features: [N, height, width, num_keypoints]
                img_mrcnn_feature_input_pl = self._add_placeholder(
                    tf.float32,
                    [None, self._img_pixel_size[0], self._img_pixel_size[1], 17],
                    self.PL_IMG_MRCNN_FEATURE_FULL_INPUT)

            # Apply masks to image features([M, 28, 28, num_keypoints] or [N, height, width, num_keypoints])
            self.img_mrcnn_feature_input_pl = tf.multiply(img_mrcnn_feature_input_pl, tf.cast(self.img_mrcnn_mask_input_tile_17_pl, dtype=tf.float32))  # shape=(?, 28, 28, 17)

        # if self._img_feature_shifted:
        #     with tf.variable_scope('img_mrcnn_bbox_input'):
        #         # Placeholder for image keypoint features:
        #         # bbox: [batch, N, (y1, x1, y2, x2)] detection bounding boxes
        #         # This b-box is in pixels.
        #         self._img_mrcnn_bbox_input_pl = self._add_placeholder(
        #             tf.float32,
        #             [None, 4],
        #             self.PL_IMG_MRCNN_BBOX_INPUT)
        #
        #         y1 = self._img_mrcnn_bbox_input_pl[:, 0]/tf.cast(tf.shape(img_input_placeholder)[0], tf.float32)
        #         x1 = self._img_mrcnn_bbox_input_pl[:, 1]/tf.cast(tf.shape(img_input_placeholder)[1], tf.float32)
        #         y2 = self._img_mrcnn_bbox_input_pl[:, 2]/tf.cast(tf.shape(img_input_placeholder)[0], tf.float32)
        #         x2 = self._img_mrcnn_bbox_input_pl[:, 3]/tf.cast(tf.shape(img_input_placeholder)[1], tf.float32)
        #         normed_boxes = tf.stack([y1, x1, y2, x2], axis=1)
        #         # Crop the whole image into boxes, and resize each box into 28x28
        #         img_croped_input_64x64 = tf.image.crop_and_resize(
        #             self._img_input_batches,
        #             normed_boxes,
        #             tf.zeros([tf.shape(self._img_mrcnn_bbox_input_pl)[0]], tf.int32),
        #             [64, 64])
        #
        #         # img_mrcnn_mask_input_28x28x3 = tf.tile(
        #         #     tf.reshape(img_mrcnn_mask_input_placeholder, [-1, 28, 28, 1]),
        #         #     [1, 1, 1, 3],
        #         #     name=None
        #         # )
        #         #
        #         # Make croped mask for 64x64 input
        #         img_mrcnn_mask_64x64 = tf.image.resize_images(  # shape=(64, 64, 1)
        #             tf.reshape(img_mrcnn_mask_input_placeholder, [-1, 28, 28, 1]),
        #             [64, 64]
        #             # method=ResizeMethod.BILINEAR,
        #             # align_corners=True
        #             # preserve_aspect_ratio=False
        #         )
        #         img_mrcnn_mask_64x64x3 = tf.tile(  # shape=(1, 64, 64, 3)
        #             tf.reshape(img_mrcnn_mask_64x64, [-1, 64, 64, 1]),
        #             [1, 1, 1, 3],
        #             name=None
        #         )
        #         # Set non-mask values to zeros
        #         self.img_croped_masked_input = tf.multiply(img_croped_input_64x64,  tf.cast(img_mrcnn_mask_64x64x3, dtype=tf.float32))

            with tf.variable_scope('orient_input'):
                # Placeholder for orientation of each candidate:
                # orient_pred: [batch, N], N is the number of all candidates, already tiled.
                self._orient_pred_pl = self._add_placeholder(
                    tf.float32,
                    [None],
                    self.PL_ORIENT_PRED)

                self._crop_group_pl = self._add_placeholder(
                    tf.int32,
                    [None],
                    self.PL_CROP_GROUP)

            # ----------- For feature maps shifted by anchors positions: ----------
            if self._img_feature_shifted:
                with tf.variable_scope('img_mrcnn_mask_shifted'):
                    # Placeholder for image masks or full_masks
                    # full_masks: (dtype=uint8), [batch, height, width, N] Instance masks
                    # masks: array(dtype=float32), [N, 28, 28]
                    img_mrcnn_mask_shifted_placeholder = self._add_placeholder(
                        tf.float32,
                        [None, 28, 28],
                        self.PL_IMG_MASK_SHIFTED)

                    self.img_mrcnn_mask_shifted_tile_17_pl = tf.tile(
                        tf.reshape(img_mrcnn_mask_shifted_placeholder, [-1, 28, 28, 1]),
                        [1, 1, 1, 17],
                        name=None
                    )

                with tf.variable_scope('img_mrcnn_feature_shifted'):
                    # Placeholder for image keypoint features:
                    # features: [N, 28, 28, num_keypoints]
                    img_mrcnn_feature_shifted_pl = self._add_placeholder(
                        tf.float32,
                        [None, 28, 28, 17],
                        self.PL_IMG_MRCNN_FEATURE_SHIFTED)

                    # Set non-mask values to zeros
                    self.img_mrcnn_feature_shifted_pl = tf.multiply(img_mrcnn_feature_shifted_pl, tf.cast(self.img_mrcnn_mask_shifted_tile_17_pl, dtype=tf.float32))

                with tf.variable_scope('img_mrcnn_bbox_shifted'):
                    # Placeholder for image keypoint features:
                    # bbox: [batch, N, (y1, x1, y2, x2)] detection bounding boxes
                    # This b-box is in pixels.
                    self._img_mrcnn_bbox_shifted_pl = self._add_placeholder(
                        tf.float32,
                        [None, 4],
                        self.PL_IMG_MRCNN_BBOX_SHIFTED)

                    y1 = self._img_mrcnn_bbox_shifted_pl[:, 0]/tf.cast(tf.shape(img_input_placeholder)[0], tf.float32)
                    x1 = self._img_mrcnn_bbox_shifted_pl[:, 1]/tf.cast(tf.shape(img_input_placeholder)[1], tf.float32)
                    y2 = self._img_mrcnn_bbox_shifted_pl[:, 2]/tf.cast(tf.shape(img_input_placeholder)[0], tf.float32)
                    x2 = self._img_mrcnn_bbox_shifted_pl[:, 3]/tf.cast(tf.shape(img_input_placeholder)[1], tf.float32)
                    normed_boxes = tf.stack([y1, x1, y2, x2], axis=1)
                    # Crop the whole image into boxes, and resize each box into 28x28
                    img_croped_shifted = tf.image.crop_and_resize(
                        self._img_input_batches,
                        normed_boxes,
                        tf.zeros([tf.shape(self._img_mrcnn_bbox_shifted_pl)[0]], tf.int32),
                        [28, 28])

                    img_mrcnn_mask_shifted_tile_3_pl = tf.tile(
                        tf.reshape(img_mrcnn_mask_shifted_placeholder, [-1, 28, 28, 1]),
                        [1, 1, 1, 3],
                        name=None
                    )
                    # Set non-mask values to zeros
                    self.img_croped_masked_shifted = tf.multiply(img_croped_shifted, tf.cast(img_mrcnn_mask_shifted_tile_3_pl, dtype=tf.float32))

        with tf.variable_scope('pl_labels'):
            self._add_placeholder(tf.float32, [None, 6],
                                  self.PL_LABEL_ANCHORS)
            self._add_placeholder(tf.float32, [None, 7],
                                  self.PL_LABEL_BOXES_3D)
            self._add_placeholder(tf.float32, [None, 4],
                                  self.PL_LABEL_BOXES_2D)
            self._add_placeholder(tf.float32, [None, 4],
                                  self.PL_LABEL_BOXES_2D_NORM)
            self._add_placeholder(tf.float32, [None],
                                  self.PL_LABEL_CLASSES)
            self._add_placeholder(tf.float32, [None, 4],
                                  self.PL_LABEL_BOXES_QUATERNION)

        # Placeholders for anchors
        with tf.variable_scope('pl_anchors'):
            self._add_placeholder(tf.float32, [None, 6],
                                  self.PL_ANCHORS)
            self._add_placeholder(tf.float32, [None],
                                  self.PL_ANCHOR_IOUS)
            self._add_placeholder(tf.float32, [None, 6],
                                  self.PL_ANCHOR_OFFSETS)
            self._add_placeholder(tf.float32, [None],
                                  self.PL_ANCHOR_CLASSES)

            with tf.variable_scope('bev_anchor_projections'):
                self._add_placeholder(tf.float32, [None, 4],
                                      self.PL_BEV_ANCHORS)
                self._bev_anchors_norm_pl = self._add_placeholder(
                    tf.float32, [None, 4], self.PL_BEV_ANCHORS_NORM)
                self._img_repeat_times_pl = self._add_placeholder(
                    tf.int32, [None], self.PL_IMG_REPEAT_TIMES)

            if not self._img_feature_shifted:
                with tf.variable_scope('img_anchor_projections'):
                    self._add_placeholder(tf.float32, [None, 4],
                                          self.PL_IMG_ANCHORS)
                    self._img_anchors_norm_pl = self._add_placeholder(
                        tf.float32, [None, 4], self.PL_IMG_ANCHORS_NORM)

            with tf.variable_scope('sample_info'):
                # the calib matrix shape is (3 x 3) in Panoptic dataset
                self._add_placeholder(
                    tf.float32, [3, 3], self.PL_CALIB_P2)
                self._add_placeholder(tf.int32,
                                      shape=[1],
                                      name=self.PL_IMG_IDX)
                self._add_placeholder(tf.float32, [4], self.PL_GROUND_PLANE)

    def _set_up_feature_extractors(self):
        """Sets up feature extractors and stores feature maps and
        bottlenecks as member variables.
        """

        self.bev_feature_maps, self.bev_end_points = \
            self._bev_feature_extractor.build(
                self._bev_preprocessed,
                self._bev_pixel_size,
                self._is_training)

        with tf.variable_scope('bev_bottleneck'):
            self.bev_bottleneck = slim.conv2d(
                self.bev_feature_maps,
                1, [1, 1],
                scope='bottleneck',
                normalizer_fn=slim.batch_norm,
                normalizer_params={
                    'is_training': self._is_training})

        if self._img_feature_shifted:
            print('!!!!!!!!!!!! rpn_pplp_panoptic_model(line 420) Using shifted feature as image feature input!!!!!!!!!!!!')
            with tf.variable_scope('img_bottleneck'):
                self.img_bottleneck = slim.conv2d(
                    self.img_mrcnn_feature_shifted_pl,
                    1, [1, 1],
                    scope='bottleneck',
                    normalizer_fn=slim.batch_norm,
                    normalizer_params={
                        'is_training': self._is_training})
        else:
            with tf.variable_scope('img_bottleneck'):
                self.img_bottleneck = slim.conv2d(  # shape=(?, height, width, 1)
                    self.img_mrcnn_feature_input_pl,  # shape=(?, height, width, 17)
                    1, [1, 1],
                    scope='bottleneck',
                    normalizer_fn=slim.batch_norm,
                    normalizer_params={
                        'is_training': self._is_training})

        # # Visualize the end point feature maps being used
        # for feature_map in list(self.bev_end_points.items()):
        #     if 'conv' in feature_map[0]:
        #         summary_utils.add_feature_maps_from_dict(self.bev_end_points,
        #                                                  feature_map[0])
        #
        # for feature_map in list(self.img_end_points.items()):
        #     if 'conv' in feature_map[0]:
        #         summary_utils.add_feature_maps_from_dict(self.img_end_points,
        #                                                  feature_map[0])

    def build(self):

        # print('&&&&&&&&&&&&&&& Build RPN model start &&&&&&&&&&&&&&&')
        # Setup input placeholders
        self._set_up_input_pls()

        # Setup feature extractors
        self._set_up_feature_extractors()

        bev_proposal_input = self.bev_bottleneck  # shape=(1, 350, 400, 1),
        img_input_batches = self._img_input_batches  # shape=(1, 1920, 1280, 3),
        img_feature_input = self.img_bottleneck  # shape=(?, 28, 28, 1) or (?, height, width, 1)

        # img_feature_input = tf.Print(img_feature_input, ['line 389: tf.shape(img_feature_input) =', tf.shape(img_feature_input)], summarize=1000)
        # bev_proposal_input = tf.Print(bev_proposal_input, ['line 390: tf.shape(bev_proposal_input) =', tf.shape(bev_proposal_input)], summarize=1000)

        fusion_mean_div_factor = 2.0

        # If both img and bev probabilites are set to 1.0, don't do
        # path drop.
        if not (self._path_drop_probabilities[0] ==
                self._path_drop_probabilities[1] == 1.0):
            with tf.variable_scope('rpn_path_drop'):

                random_values = tf.random_uniform(shape=[3],
                                                  minval=0.0,
                                                  maxval=1.0)

                img_mask, bev_mask = self.create_path_drop_masks(
                    self._path_drop_probabilities[0],
                    self._path_drop_probabilities[1],
                    random_values)

                img_feature_input = tf.multiply(img_feature_input,
                                                img_mask)

                # # For summary only:
                # img_mask_tile_3_pl = tf.tile(
                #     tf.reshape(img_mask, [-1, -1, -1, 1]),
                #     [1, 1, 1, 3],
                #     name=None
                # )
                # img_input_batches = tf.multiply(img_input_batches,
                #                                 img_mask_tile_3_pl)

                bev_proposal_input = tf.multiply(bev_proposal_input,
                                                 bev_mask)

                self.img_path_drop_mask = img_mask
                self.bev_path_drop_mask = bev_mask

                # Overwrite the division factor
                fusion_mean_div_factor = img_mask + bev_mask

        with tf.variable_scope('proposal_roi_pooling'):

            with tf.variable_scope('box_indices'):
                def get_box_indices(boxes):
                    proposals_shape = boxes.get_shape().as_list()
                    if any(dim is None for dim in proposals_shape):
                        proposals_shape = tf.shape(boxes)
                    ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
                    multiplier = tf.expand_dims(
                        tf.range(start=0, limit=proposals_shape[0]), 1)
                    return tf.reshape(ones_mat * multiplier, [-1])

                bev_boxes_norm_batches = tf.expand_dims(
                    self._bev_anchors_norm_pl, axis=0)

                # These should be all 0's since there is only 1 image
                # tf_box_indices = tf.zeros([tf.shape(self._bev_anchors_norm_pl)[0]], tf.int32)
                tf_box_indices = get_box_indices(bev_boxes_norm_batches)

            # sess_show = tf.InteractiveSession()
            # # We can just use 'c.eval()' without passing 'sess'
            # print('bev_proposal_input = ', bev_proposal_input.eval())
            # print('self._bev_anchors_norm_pl = ', self._bev_anchors_norm_pl.eval())
            # print('tf_box_indices = ', tf_box_indices.eval())
            # print('self._proposal_roi_crop_size = ', self._proposal_roi_crop_size.eval())
            # sess_show.close()
            # Do ROI Pooling on BEV
            # bev_proposal_input = tf.Print(bev_proposal_input, ['line 414: bev_proposal_input.size =', tf.shape(bev_proposal_input)], summarize=1000)    # In AVOD method: bev_proposal_input.size =][1 350 400 1]
            # bev_proposal_input = tf.Print(bev_proposal_input, ['line 415: bev_proposal_input =', bev_proposal_input], summarize=1000)
            # self._bev_anchors_norm_pl = tf.Print(self._bev_anchors_norm_pl, ['line 416: self._bev_anchors_norm_pl.size =', tf.shape(self._bev_anchors_norm_pl)], summarize=1000)    # In AVOD method: self._bev_anchors_norm_pl.size =][880 4] sometimes [2454 4]
            # self._bev_anchors_norm_pl = tf.Print(self._bev_anchors_norm_pl, ['line 417: self._bev_anchors_norm_pl =', self._bev_anchors_norm_pl], summarize=1000)
            # tf_box_indices = tf.Print(tf_box_indices, ['line 418: tf_box_indices.size =', tf.shape(tf_box_indices)], summarize=1000)    # In AVOD method: tf_box_indices.size =][880] sometimes [2454]
            # tf_box_indices = tf.Print(tf_box_indices, ['line 419: tf_box_indices =', tf_box_indices], summarize=1000)    # In AVOD method: tf_box_indices =][0 0 0 0 .... 0 0 0 0 0 0]
            bev_proposal_rois = tf.image.crop_and_resize(
                bev_proposal_input,
                self._bev_anchors_norm_pl,
                tf_box_indices,
                self._proposal_roi_crop_size)

            # bev_proposal_rois = tf.image.crop_and_resize(
            #     bev_proposal_input,
            #     tf.tile([[0.3, 0.3, 0.5, 0.5]], [tf.shape(img_feature_input)[0],1]),
            #     tf.tile([0], [tf.shape(img_feature_input)[0]]),
            #     self._proposal_roi_crop_size)

            if self._img_feature_shifted:
                # img_feature_input comes from self.img_bottleneck which dimension is: [?x28x28x17]
                img_proposal_rois = tf.image.resize_images(  # shape=(?, 7, 7, 1)
                    img_feature_input,  # shape=(?, 28, 28, 1)
                    self._proposal_roi_crop_size  # default [7, 7]
                    # method=ResizeMethod.BILINEAR,
                    # align_corners=True
                    # preserve_aspect_ratio=False
                )
                img_proposal_feature = self.img_mrcnn_feature_input_pl  # shape=(?, 28, 28, 17)
            else:
                # Do ROI Pooling on image
                # img_feature_input comes from self.img_bottleneck which dimension is: [?xHeightxWidthx1]
                img_proposal_rois = tf.image.crop_and_resize(
                    img_feature_input,  # shape=(?, Height, Width, 1)
                    self._img_anchors_norm_pl,
                    self._crop_group_pl,
                    self._proposal_roi_crop_size)

        with tf.variable_scope('proposal_roi_fusion'):
            rpn_fusion_out = None
            if self._fusion_method == 'mean':
                tf_features_sum = tf.add(bev_proposal_rois, img_proposal_rois)
                rpn_fusion_out = tf.divide(tf_features_sum,
                                           fusion_mean_div_factor)
            elif self._fusion_method == 'concat':
                rpn_fusion_out = tf.concat(
                    [bev_proposal_rois, img_proposal_rois], axis=3)
            else:
                raise ValueError('Invalid fusion method', self._fusion_method)

        # TODO: move this section into an separate AnchorPredictor class
        with tf.variable_scope('anchor_predictor', 'ap', [rpn_fusion_out]):
            tensor_in = rpn_fusion_out

            # Parse rpn layers config
            layers_config = self._config.layers_config.rpn_config
            l2_weight_decay = layers_config.l2_weight_decay

            if l2_weight_decay > 0:
                weights_regularizer = slim.l2_regularizer(l2_weight_decay)
            else:
                weights_regularizer = None

            with slim.arg_scope([slim.conv2d],
                                weights_regularizer=weights_regularizer):
                # Use conv2d instead of fully_connected layers.
                cls_fc6 = slim.conv2d(tensor_in,
                                      layers_config.cls_fc6,
                                      self._proposal_roi_crop_size,
                                      padding='VALID',
                                      scope='cls_fc6')

                cls_fc6_drop = slim.dropout(cls_fc6,
                                            layers_config.keep_prob,
                                            is_training=self._is_training,
                                            scope='cls_fc6_drop')

                cls_fc7 = slim.conv2d(cls_fc6_drop,
                                      layers_config.cls_fc7,
                                      [1, 1],
                                      scope='cls_fc7')

                cls_fc7_drop = slim.dropout(cls_fc7,
                                            layers_config.keep_prob,
                                            is_training=self._is_training,
                                            scope='cls_fc7_drop')

                cls_fc8 = slim.conv2d(cls_fc7_drop,
                                      2,
                                      [1, 1],
                                      activation_fn=None,
                                      scope='cls_fc8')

                objectness = tf.squeeze(
                    cls_fc8, [1, 2],
                    name='cls_fc8/squeezed')

                # Use conv2d instead of fully_connected layers.
                reg_fc6 = slim.conv2d(tensor_in,
                                      layers_config.reg_fc6,
                                      self._proposal_roi_crop_size,
                                      padding='VALID',
                                      scope='reg_fc6')

                reg_fc6_drop = slim.dropout(reg_fc6,
                                            layers_config.keep_prob,
                                            is_training=self._is_training,
                                            scope='reg_fc6_drop')

                reg_fc7 = slim.conv2d(reg_fc6_drop,
                                      layers_config.reg_fc7,
                                      [1, 1],
                                      scope='reg_fc7')

                reg_fc7_drop = slim.dropout(reg_fc7,
                                            layers_config.keep_prob,
                                            is_training=self._is_training,
                                            scope='reg_fc7_drop')

                reg_fc8 = slim.conv2d(reg_fc7_drop,
                                      6,
                                      [1, 1],
                                      activation_fn=None,
                                      scope='reg_fc8')

                offsets = tf.squeeze(
                    reg_fc8, [1, 2],
                    name='reg_fc8/squeezed')

        # Histogram summaries
        with tf.variable_scope('histograms_feature_extractor'):
            with tf.variable_scope('bev_vgg'):
                for end_point in self.bev_end_points:
                    tf.summary.histogram(
                        end_point, self.bev_end_points[end_point])

            # with tf.variable_scope('img_vgg'):  # Don't need it in PPLP method
            #     for end_point in self.img_end_points:
            #         tf.summary.histogram(
            #             end_point, self.img_end_points[end_point])

        with tf.variable_scope('histograms_rpn'):
            with tf.variable_scope('anchor_predictor'):
                fc_layers = [cls_fc6, cls_fc7, cls_fc8, objectness,
                             reg_fc6, reg_fc7, reg_fc8, offsets]
                for fc_layer in fc_layers:
                    # fix the name to avoid tf warnings
                    tf.summary.histogram(fc_layer.name.replace(':', '_'),
                                         fc_layer)

        # Return the proposals
        # print('offsets = ', offsets)
        with tf.variable_scope('proposals'):
            anchors = self.placeholders[self.PL_ANCHORS]  # anchors here are already tiled
            # anchors = tf.Print(anchors, ['line 599: anchors =', anchors], summarize=1000)
            # offsets = tf.Print(offsets, ['line 600: offsets =', offsets], summarize=1000)
            # Decode anchor regression offsets
            with tf.variable_scope('decoding'):
                regressed_anchors = anchor_panoptic_encoder.offset_to_anchor(
                        anchors, offsets)

            with tf.variable_scope('bev_projection'):
                _, bev_proposal_boxes_norm = anchor_panoptic_projector.project_to_bev(
                    regressed_anchors, self._bev_extents)

            with tf.variable_scope('softmax'):
                objectness_softmax = tf.nn.softmax(objectness)
                # objectness_softmax = tf.Print(objectness_softmax, ['line 601(rpn) : objectness_softmax =', objectness_softmax], summarize=1000)

            with tf.variable_scope('nms'):
                objectness_scores = objectness_softmax[:, 1]
                # objectness_scores = tf.Print(objectness_scores, ['line 619(rpn) : objectness_scores =', objectness_scores], summarize=1000)
                # bev_proposal_boxes_norm = tf.Print(bev_proposal_boxes_norm, ['line 620(rpn) : bev_proposal_boxes_norm =', bev_proposal_boxes_norm], summarize=1000)
                # Do NMS on regressed anchors
                top_indices = tf.image.non_max_suppression(  # Now we have too many proposals, only retain poroposals that are far away enough.
                    bev_proposal_boxes_norm, objectness_scores,
                    max_output_size=self._nms_size,
                    iou_threshold=self._nms_iou_thresh)

                # regressed_anchors = tf.Print(regressed_anchors, ['line 627(rpn) : regressed_anchors =', regressed_anchors], summarize=1000)
                # top_indices = tf.Print(top_indices, ['line 628(rpn) : top_indices =', top_indices], summarize=1000)
                top_anchors = tf.gather(regressed_anchors, top_indices)
                top_objectness_softmax = tf.gather(objectness_scores,
                                                   top_indices)
                # top_offsets = tf.gather(offsets, top_indices)
                # top_objectness = tf.gather(objectness, top_indices)
                if self._img_feature_shifted:
                    top_img_feature = tf.gather(img_proposal_feature, top_indices)  # shape=(?, 28, 28, 17), for pplp use

                top_orient_pred = tf.gather(self._orient_pred_pl, top_indices)  # shape=(?), for pplp orientation output
                top_crop_group = tf.gather(self._crop_group_pl, top_indices)  # shape=(?), for pplp result selection

        # Get mini batch
        all_ious_gt = self.placeholders[self.PL_ANCHOR_IOUS]
        # all_ious_gt = tf.Print(all_ious_gt, ['^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^line 673(pplp) : all_ious_gt =', all_ious_gt], summarize=1000)
        all_offsets_gt = self.placeholders[self.PL_ANCHOR_OFFSETS]
        # all_offsets_gt = tf.Print(all_offsets_gt, ['^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^line 621(pplp) : all_offsets_gt =', all_offsets_gt], summarize=1000)
        all_classes_gt = self.placeholders[self.PL_ANCHOR_CLASSES]

        with tf.variable_scope('mini_batch'):
            mini_batch_panoptic_utils = self.dataset.panoptic_utils.mini_batch_panoptic_utils
            mini_batch_mask, _ = \
                mini_batch_panoptic_utils.sample_rpn_mini_batch(all_ious_gt)
            # mini_batch_mask = tf.Print(mini_batch_mask, ['line 682(pplp) : mini_batch_mask =', mini_batch_mask], summarize=1000)

        # ROI summary images
        rpn_mini_batch_size = \
            self.dataset.panoptic_utils.mini_batch_panoptic_utils.rpn_mini_batch_size
        with tf.variable_scope('bev_rpn_rois'):
            mb_bev_anchors_norm = tf.boolean_mask(self._bev_anchors_norm_pl,
                                                  mini_batch_mask)
            mb_bev_box_indices = tf.zeros_like(
                tf.boolean_mask(all_classes_gt, mini_batch_mask),
                dtype=tf.int32)

            # Show the ROIs of the BEV input density map
            # for the mini batch anchors
            # mb_bev_anchors_norm = tf.Print(mb_bev_anchors_norm, ['line 637: mb_bev_anchors_norm =', mb_bev_anchors_norm], summarize=1000)
            # mb_bev_box_indices = tf.Print(mb_bev_box_indices, ['line 638: mb_bev_box_indices =', mb_bev_box_indices], summarize=1000)
            bev_input_rois = tf.image.crop_and_resize(
                self._bev_preprocessed,
                mb_bev_anchors_norm,
                mb_bev_box_indices,
                (32, 32))

            bev_input_roi_summary_images = tf.split(
                bev_input_rois, self._bev_depth, axis=3)
            tf.summary.image('bev_rpn_rois',
                             bev_input_roi_summary_images[-1],
                             max_outputs=rpn_mini_batch_size)

        with tf.variable_scope('img_feature_input'):
            tf.summary.image('img_feature_input_convoluted',
                             img_feature_input,  # [?xHeightxWidthx1] or [?x28x28x1]
                             max_outputs=rpn_mini_batch_size)
            tf.summary.image('img_proposal_rois_resized',
                             img_proposal_rois,
                             max_outputs=rpn_mini_batch_size)

        if self._img_feature_shifted:
            with tf.variable_scope('img_croped_masked_shifted'):

                tf.summary.image('img_croped_masked_shifted',
                                 self.img_croped_masked_shifted,
                                 max_outputs=rpn_mini_batch_size)

        # if not self._img_feature_shifted:
        #     with tf.variable_scope('rpn_fusion_rgb'):
        #         # For summary only:
        #         img_proposal_rgb = tf.image.crop_and_resize(
        #             img_input_batches,  # shape=(1, Height, Width, 3)
        #             self._img_anchors_norm_pl,
        #             tf_box_indices,
        #             self._proposal_roi_crop_size)
        #         # For summary only
        #         rpn_fusion_rgb_out = tf.concat(
        #             [bev_proposal_rois, img_proposal_rgb], axis=2)
        #         tf.summary.image('rpn_fusion_rgb',
        #                          rpn_fusion_rgb_out,
        #                          max_outputs=rpn_mini_batch_size)

        # Ground Truth Tensors
        with tf.variable_scope('one_hot_classes'):

            # Anchor classification ground truth
            # Object / Not Object
            min_pos_iou = \
                self.dataset.panoptic_utils.mini_batch_panoptic_utils.rpn_pos_iou_range[0]

            objectness_classes_gt = tf.cast(
                tf.greater_equal(all_ious_gt, min_pos_iou),
                dtype=tf.int32)
            objectness_gt = tf.one_hot(
                objectness_classes_gt, depth=2,
                on_value=1.0 - self._config.label_smoothing_epsilon,
                off_value=self._config.label_smoothing_epsilon)

        # Mask predictions for mini batch
        with tf.variable_scope('prediction_mini_batch'):
            # mini_batch_mask = tf.Print(mini_batch_mask, ['line 759(rpn) : mini_batch_mask =', mini_batch_mask], summarize=100)
            # objectness = tf.Print(objectness, ['line 760(rpn) : objectness =', objectness], summarize=100)
            # offsets = tf.Print(offsets, ['line 761(rpn) : offsets =', offsets], summarize=100)
            objectness_masked = tf.boolean_mask(objectness, mini_batch_mask)
            offsets_masked = tf.boolean_mask(offsets, mini_batch_mask)
            # objectness_masked = tf.Print(objectness_masked, ['line 764(rpn) : objectness_masked =', objectness_masked], summarize=100)
            # offsets_masked = tf.Print(offsets_masked, ['line 765(rpn) : offsets_masked =', offsets_masked], summarize=100)

        with tf.variable_scope('ground_truth_mini_batch'):
            objectness_gt_masked = tf.boolean_mask(
                objectness_gt, mini_batch_mask)
            # objectness_gt_masked = tf.Print(objectness_gt_masked, ['line 770(rpn) : objectness_gt_masked =', objectness_gt_masked], summarize=100)
            offsets_gt_masked = tf.boolean_mask(all_offsets_gt,
                                                mini_batch_mask)

        # Specify the tensors to evaluate
        predictions = dict()

        # Temporary predictions for debugging
        # predictions['anchor_ious'] = anchor_ious
        # predictions['anchor_offsets'] = all_offsets_gt

        if self._train_val_test in ['train', 'val']:
            # All anchors
            predictions[self.PRED_ANCHORS] = anchors

            # Mini-batch masks
            predictions[self.PRED_MB_MASK] = mini_batch_mask
            # Mini-batch predictions
            predictions[self.PRED_MB_OBJECTNESS] = objectness_masked
            predictions[self.PRED_MB_OFFSETS] = offsets_masked

            # Mini batch ground truth
            predictions[self.PRED_MB_OFFSETS_GT] = offsets_gt_masked
            predictions[self.PRED_MB_OBJECTNESS_GT] = objectness_gt_masked

            # Proposals after nms
            predictions[self.PRED_TOP_INDICES] = top_indices

        # For both self._train_val_test in ['train', 'val'] or in ['test']
        predictions[self.PRED_TOP_ANCHORS] = top_anchors
        predictions[
            self.PRED_TOP_OBJECTNESS_SOFTMAX] = top_objectness_softmax
        # Proposals for image feature in pplp net
        if self._img_feature_shifted:
            predictions[self.PRED_TOP_IMG_FEATURE] = top_img_feature  # shape=(?, 28, 28, 17), for pplp use

        predictions[self.PRED_TOP_ORIENT_PRED] = top_orient_pred  # shape=(?), for pplp orientation output
        predictions[self.PRED_TOP_CROP_GROUP] = top_crop_group  # shape=(?), for pplp result selection

        # print('&&&&&&&&&&&&&&& Build RPN model end &&&&&&&&&&&&&&&')
        return predictions

    def _make_file_path(self, classes_name, sub_str, sample_name, subsub_str=None):
        """Make a full file path to the mini batches

        Args:
            classes_name: name of classes ('Car', 'Pedestrian', 'Cyclist',
                'People')
            sub_str: a name for folder subname
            sample_name: sample name, e.g. '000123'

        Returns:
            The anchors info file path. Returns the folder if
                sample_name is None
        """
        mini_batch_dir = 'pplp/data/mini_batches/iou_2d/panoptic/train/lidar'
        if sample_name:
            if subsub_str:
                return mini_batch_dir + '/' + classes_name + \
                    '[' + sub_str + ']/' + \
                    subsub_str + '/' + \
                    sample_name + ".npy"
            else:
                return mini_batch_dir + '/' + classes_name + \
                    '[' + sub_str + ']/' + \
                    sample_name + ".npy"
        else:
            if subsub_str:
                return mini_batch_dir + '/' + classes_name + \
                    '[' + sub_str + ']/' + subsub_str
            else:
                return mini_batch_dir + '/' + classes_name + \
                    '[' + sub_str + ']'

    def _read_orient_prediction_from_file(self, classes_name, sample_name):
        """
        Reads the orientation info arrays from a file

        Args:
            classes_name (str): classes name, e.g. 'Car', 'Pedestrian',
                'Cyclist', 'People'
            sample_name (str): name of sample, e.g. '500100008677'

        Returns:
            results: {'boxes_3d': array(dtype=float32), shape = (Nx11)}
            [x1, y1, x2, y2, height, width, length, x,y,z, ry]
        """

        sub_str = 'orient_pred'
        results = {}

        file_name = self._make_file_path(classes_name,
                                         sub_str,
                                         sample_name)
        # print('self._read_orient_prediction_from_file :: file_name = ', file_name)
        # Load from npy file
        results = np.load(file_name)
        return results

    def create_feed_dict(self, sample_index=None):
        """ Fills in the placeholders with the actual input values.
            Currently, only a batch size of 1 is supported

        Args:
            sample_index: optional, only used when train_val_test == 'test',
                a particular sample index in the dataset
                sample list to build the feed_dict for

        Returns:
            a feed_dict dictionary that can be used in a tensorflow session
        """
        # print('~~~~~~~~~~~~~~~~ pplp/core/models/rpn_panoptic_model.py ~~~~~~~~~~~~~~~~')
        start_create_feed_dict = time.time()
        if self._train_val_test in ["train", "val"]:

            # sample_index should be None
            if sample_index is not None:
                raise ValueError('sample_index should be None. Do not load '
                                 'particular samples during train or val')

            # During training/validation, we need a valid sample
            # with anchor info for loss calculation
            sample = None
            anchors_info = []

            valid_sample = False
            while not valid_sample:
                if self._train_val_test == "train":
                    # Get the a random sample from the remaining epoch
                    # print('!!!!!!!!!!!!! TODO: Get a sample from the remaining epoch !!!!!!!!!!!!!!!!!!!!!!!')
                    samples = self.dataset.next_batch(batch_size=1, shuffle=True)

                else:  # self._train_val_test == "val"
                    # Load samples in order for validation
                    print('Load samples in order for validation')
                    samples = self.dataset.next_batch(batch_size=1,
                                                      shuffle=True)  # Default shuffle=False

                # Check if thers is any pedestrian detected by maskrcnn
                if not samples:
                    print('No pedestrian detected by MaskRCNN, get the next batch!')
                    continue
                # Only handle one sample at a time for now
                sample = samples[0]
                anchors_info = sample.get(constants.KEY_ANCHORS_INFO)

                # When training, if the mini batch is empty, go to the next
                # sample. Otherwise carry on with found the valid sample.
                # For validation, even if 'anchors_info' is empty, keep the
                # sample (this will help penalize false positives.)
                # We will substitue the necessary info with zeros later on.
                # Note: Training/validating all samples can be switched off.
                train_cond = (self._train_val_test == "train" and
                              self._train_on_all_samples)
                eval_cond = (self._train_val_test == "val" and
                             self._eval_all_samples)
                if anchors_info or train_cond or eval_cond:
                    valid_sample = True
        else:
            # For testing, any sample should work
            print('For testing, any sample should work')
            if sample_index is not None:
                print('load_samples()')
                samples = self.dataset.load_samples([sample_index])
            else:
                print('next_batch()')
                samples = self.dataset.next_batch(batch_size=1, shuffle=False)

            # Only handle one sample at a time for now
            sample = samples[0]
            anchors_info = sample.get(constants.KEY_ANCHORS_INFO)
        get_batch_time = time.time()
        print('get_batch takes ', (get_batch_time-start_create_feed_dict), 's')
        # When it is fast, it takes, 0.328s, when it is slow, it can take 2~3s.

        sample_name = sample.get(constants.KEY_SAMPLE_NAME)
        sample_augs = sample.get(constants.KEY_SAMPLE_AUGS)

        # Get ground truth data
        label_anchors = sample.get(constants.KEY_LABEL_ANCHORS)
        label_classes = sample.get(constants.KEY_LABEL_CLASSES)
        label_boxes_2d = sample.get(constants.KEY_LABEL_BOXES_2D)
        label_boxes_2d_norm = label_boxes_2d.copy().astype(np.float32)
        label_boxes_2d_norm[:, 0] = label_boxes_2d_norm[:, 0]/self._img_pixel_size[1].astype(np.float32)
        label_boxes_2d_norm[:, 1] = label_boxes_2d_norm[:, 1]/self._img_pixel_size[0].astype(np.float32)
        label_boxes_2d_norm[:, 2] = label_boxes_2d_norm[:, 2]/self._img_pixel_size[1].astype(np.float32)
        label_boxes_2d_norm[:, 3] = label_boxes_2d_norm[:, 3]/self._img_pixel_size[0].astype(np.float32)
        # We only need orientation from box_3d
        label_boxes_3d = sample.get(constants.KEY_LABEL_BOXES_3D)
        # Transform orientation to quaternion format (w, x, y, z)
        label_boxes_quat = []
        # print('label_boxes_3d = ', label_boxes_3d)
        get_sample_time = time.time()
        print('get_sample takes ', (get_sample_time-get_batch_time), 's')  # Super fast
        for j in range(len(label_boxes_3d)):
            # use translation to get apparant orientation of object in the
            # camera view.
            rot = euler2mat(0, label_boxes_3d[j][6], 0, axes='rxyz')
            quat = mat2quat(rot)  # (w,x,y,z)
            if label_boxes_quat == []:
                label_boxes_quat = [quat]
            else:
                label_boxes_quat = np.concatenate((label_boxes_quat, [quat]), axis=0)
            # label_boxes_quat = label_boxes_quat.copy().astype(np.float32)
        # print('label_boxes_quat = ', label_boxes_quat)
        get_qua_time = time.time()
        print('get_qua takes ', (get_qua_time-get_sample_time), 's')  # Super fast

        # Network input data
        image_input = sample.get(constants.KEY_IMAGE_INPUT)  # (1080, 1920, 3)
        bev_input = sample.get(constants.KEY_BEV_INPUT)  # (700, 800, 3)
        image_mask_input = sample.get(constants.KEY_IMAGE_MASK_INPUT)  # (N, 28, 28)
        image_full_mask_input = sample.get(constants.KEY_IMAGE_FULL_MASK_INPUT)  # (1080, 1920, N)
        image_mrcnn_feature_input = sample.get(constants.KEY_IMAGE_MRCNN_FEATURE_INPUT)  # (N, 28, 28, 17)
        image_mrcnn_bbox_input = sample.get(constants.KEY_IMAGE_MRCNN_BBOX_INPUT)  # (N, 4)
        self.image_mrcnn_bbox_input = image_mrcnn_bbox_input  # This b-box is in pixels.
        image_mrcnn_keypoints_input = sample.get(constants.KEY_IMAGE_MRCNN_KEYPOINTS_INPUT)
        # Image shape (h, w)
        image_shape = [image_input.shape[0], image_input.shape[1]]

        ground_plane = sample.get(constants.KEY_GROUND_PLANE)
        stereo_calib_p2 = sample.get(constants.KEY_STEREO_CALIB_P2)

        # Fill the placeholders for anchor information
        # print('image_input.shape = ', image_input.shape)
        # print('image_mask_input.shape = ', image_mask_input.shape)
        # print('image_full_mask_input.shape = ', image_full_mask_input.shape)
        # print('image_mrcnn_feature_input.shape = ', image_mrcnn_feature_input.shape)
        # print('image_mrcnn_bbox_input.shape = ', image_mrcnn_bbox_input.shape)
        # print('bev_input.shape = ', bev_input.shape)
        # print('anchors_info = ', anchors_info)
        # print('ground_plane = ', ground_plane)
        # print('image_shape = ', image_shape)
        # print('stereo_calib_p2 = ', stereo_calib_p2)
        print('sample_name = ', sample_name)
        # print('sample_augs = ', sample_augs)
        self.use_occupancy = True
        if self.use_occupancy:
            bev_occupency_map = bev_input[:, :, -1]  # The first few layers are the height map, then flowing with the density map, then flowing with the occupancy grid map. Here, we only need the occupancy grid map.
            bev_occupency_map = np.reshape(bev_occupency_map, (bev_occupency_map.shape[0], bev_occupency_map.shape[1], 1))
            # Fill placeholder with some mrcnn data
            self._placeholder_inputs[self.PL_BEV_INPUT] = bev_occupency_map  # bev_occupency_map.shape =  (700, 800, 1)
        else:
            self._placeholder_inputs[self.PL_BEV_INPUT] = bev_input[:, :, :-1]  # The first few layers are the height map, then flowing with the density map, then flowing with the occupancy grid map.
        self._placeholder_inputs[self.PL_IMG_INPUT] = image_input  # image_input.shape =  (1080, 1920, 3)
        get_feature_time = time.time()
        print('get_feature takes ', (get_feature_time-get_qua_time), 's')  # Super fast
        self._fill_anchor_pl_inputs(anchors_info=anchors_info,
                                    ground_plane=ground_plane,
                                    image_shape=image_shape,
                                    stereo_calib_p2=stereo_calib_p2,
                                    sample_name=sample_name,
                                    sample_augs=sample_augs)
        fill_anchor_pl_time = time.time()
        print('fill_anchor_pl takes ', (fill_anchor_pl_time-get_feature_time), 's')  # Super fast

        # get orientation prediction from .npy files whose results have the same order as the MaskRCNN detection.
        classes_name = 'Pedestrian'
        orient_prediciton = self._read_orient_prediction_from_file(classes_name, sample_name)
        # print('orient_prediciton = ', orient_prediciton)
        orient_pred = orient_prediciton.item().get('orient_pred')  # 1xN,(-pi, pi), N is the number of MaskRCNN results. Each number represents an orientation result from OrientNet.
        # print('orient_pred = ', orient_pred)
        read_orient_time = time.time()
        print('read_orient takes ', (read_orient_time-fill_anchor_pl_time), 's')  # Super fast

        # We want to use different image features for different the anchors,
        # But how shall we do that?: (Also need to change the strategy in function _fill_anchor_pl_inputs())

        # Method No.1
        # Let's resize the image features(28X28X17) back into its original size(PXQX17),
        # and then put it back in to an all-zero full-size tensor(1080X1920X17).
        # Now we have tensors that only contain one people.
        if not self._img_feature_shifted:
            print('Using Method No.1')
            image_feature_full_size_input = np.zeros([len(self._img_repeat_times), image_input.shape[0], image_input.shape[1], 17])
            image_mask_full_size_input = np.zeros([len(self._img_repeat_times), image_input.shape[0], image_input.shape[1]]).astype(bool)
            for i in range(len(self._img_repeat_times)):
                temp_feature_full_size_input = np.zeros([image_input.shape[0], image_input.shape[1], 17])
                # temp_mask_full_size_input = np.zeros([image_input.shape[0], image_input.shape[1]]).astype(bool)
                # print('image_mrcnn_feature_input[i].shape = ', image_mrcnn_feature_input[i].shape)  # (28, 28, 17)
                resized_feature = cv2.resize(image_mrcnn_feature_input[i], dsize=(image_mrcnn_bbox_input[i][3]-image_mrcnn_bbox_input[i][1], image_mrcnn_bbox_input[i][2]-image_mrcnn_bbox_input[i][0]), interpolation=cv2.INTER_CUBIC)
                # cv2.imshow("resized_feature", resized_feature[:, :, 1])
                # resized_mask = cv2.resize(image_mask_input[i], dsize=(image_mrcnn_bbox_input[i][3]-image_mrcnn_bbox_input[i][1], image_mrcnn_bbox_input[i][2]-image_mrcnn_bbox_input[i][0]), interpolation=cv2.INTER_CUBIC)
                # print('image_mrcnn_bbox_input[i] = ', image_mrcnn_bbox_input[i])
                # print('resized_feature.shape = ', resized_feature.shape)
                # print('temp_feature_full_size_input.shape = ', temp_feature_full_size_input.shape)  # (1080, 1920, 17)
                temp_feature_full_size_input[image_mrcnn_bbox_input[i][0]:image_mrcnn_bbox_input[i][2], image_mrcnn_bbox_input[i][1]:image_mrcnn_bbox_input[i][3]] = resized_feature
                # temp_mask_full_size_input[image_mrcnn_bbox_input[i][0]:image_mrcnn_bbox_input[i][2], image_mrcnn_bbox_input[i][1]:image_mrcnn_bbox_input[i][3]] = resized_mask.astype(bool)
                image_feature_full_size_input[i, ::] = temp_feature_full_size_input
                image_mask_full_size_input[i, ::] = image_full_mask_input[:, :, i].astype(bool)
                # cv2.imshow("temp_feature_full_size_input_layer1", temp_feature_full_size_input[:, :, 10])
                # cv2.waitKey(0)
                # cv2.imshow("image_full_mask_input", image_mask_full_size_input[i, ::].astype(np.uint8))
                # cv2.waitKey(0)
            # print('image_mask_full_size_input.shape = ', image_mask_full_size_input.shape)
            # print('image_feature_full_size_input.shape = ', image_feature_full_size_input.shape)  # (N, 1080, 1920, 17), N is the number of people detected by MaskRCNN
            # print('image_mrcnn_bbox_input.shape = ', image_mrcnn_bbox_input.shape)
            self._placeholder_inputs[self.PL_IMG_MASK_FULL_INPUT] = image_mask_full_size_input  # shape = (N, 1080, 1920), N is the number of people detected by MaskRCNN
            self._placeholder_inputs[self.PL_IMG_MRCNN_FEATURE_FULL_INPUT] = image_feature_full_size_input  # (N, 1080, 1920, 17), N is the number of people detected by MaskRCNN
            self._placeholder_inputs[self.PL_IMG_MRCNN_BBOX_INPUT] = image_mrcnn_bbox_input  # (N, 4), N is the number of people detected by MaskRCNN

            orient_pred_final = np.zeros(int(np.sum(self._img_repeat_times)))
            crop_group_final = np.zeros(int(np.sum(self._img_repeat_times)))
            temp_start = 0
            for i in range(len(self._img_repeat_times)):
                orient_pred_result = np.tile(orient_pred[i], (int(self._img_repeat_times[i])))
                crop_group_result = np.tile(i, (int(self._img_repeat_times[i])))
                orient_pred_final[temp_start : int(temp_start+self._img_repeat_times[i])] = orient_pred_result
                crop_group_final[temp_start : int(temp_start+self._img_repeat_times[i])] = crop_group_result
                temp_start = temp_start+int(self._img_repeat_times[i])

            self._placeholder_inputs[self.PL_ORIENT_PRED] = orient_pred_final  # orient_pred.shape =  (M), M is the number of all candidates, already tiled.
            self._placeholder_inputs[self.PL_CROP_GROUP] = crop_group_final  # crop_group_final.shape =  (M), M is the number of all candidates, already tiled.

        # Method No.2
        if self._img_feature_shifted:
            print('Using Method No.2')  # Even though Method No.2 seems faster
            # here, its actually not. Because Method No.2 does not contain the
            # full-size feature. In this case, we will run into more troubles in
            # the PredictorNet.

            # Because one mrcnn feature may correspond to several anchors at a time,
            # so we need to tile the mrcnn feature.
            image_mask_input_final = []
            image_mrcnn_feature_input_final = []
            image_mrcnn_bbox_input_final = []
            image_mrcnn_keypoints_input_final = []
            orient_pred_final = []
            crop_group_final = []

            # If we want to use the same image feature for all the anchors: (Also need to change the strategy in function _fill_anchor_pl_inputs())
            # print('self._img_repeat_times = ', self._img_repeat_times)
            for i in range(len(self._img_repeat_times)):
                # image_mask_input[i].shape =  (28, 28)
                # image_mrcnn_feature_input[i].shape =  (28, 28, 17)
                # image_mrcnn_bbox_input[i].shape =  (4,)
                # image_mrcnn_keypoints_input[i].shape =  (17, 2)  In full size pixel value.
                # orient_pred.shape = (N) each number represents an orientation prediction for each MaskRCNN crop.
                image_mask_input_result = np.tile(image_mask_input[i], (int(self._img_repeat_times[i]), 1, 1))
                image_mrcnn_feature_input_result = np.tile(image_mrcnn_feature_input[i], (int(self._img_repeat_times[i]), 1, 1, 1))
                image_mrcnn_bbox_input_result = np.tile(image_mrcnn_bbox_input[i], (int(self._img_repeat_times[i]), 1))
                # print('image_mrcnn_keypoints_input[i] = ', image_mrcnn_keypoints_input[i])
                # print('int(self._img_repeat_times[i]) = ', int(self._img_repeat_times[i]))
                image_mrcnn_keypoints_input_result = np.tile(image_mrcnn_keypoints_input[i], (int(self._img_repeat_times[i]), 1, 1))
                orient_pred_result = np.tile(orient_pred[i], (int(self._img_repeat_times[i])))
                crop_group_result = np.tile(i, (int(self._img_repeat_times[i])))

                if image_mask_input_final == []:
                    image_mask_input_final = image_mask_input_result
                    image_mrcnn_feature_input_final = image_mrcnn_feature_input_result
                    image_mrcnn_bbox_input_final = image_mrcnn_bbox_input_result
                    image_mrcnn_keypoints_input_final = image_mrcnn_keypoints_input_result
                    orient_pred_final = orient_pred_result
                    crop_group_final = crop_group_result
                else:
                    image_mask_input_final = np.concatenate((image_mask_input_final, image_mask_input_result), axis=0)
                    image_mrcnn_feature_input_final = np.concatenate((image_mrcnn_feature_input_final, image_mrcnn_feature_input_result), axis=0)
                    image_mrcnn_bbox_input_final = np.concatenate((image_mrcnn_bbox_input_final, image_mrcnn_bbox_input_result), axis=0)
                    image_mrcnn_keypoints_input_final = np.concatenate((image_mrcnn_keypoints_input_final, image_mrcnn_keypoints_input_result), axis=0)
                    orient_pred_final = np.concatenate((orient_pred_final, orient_pred_result), axis=0)
                    crop_group_final = np.concatenate((crop_group_final, crop_group_result), axis=0)

            image_mrcnn_keypoints_norm_input_final = image_mrcnn_keypoints_input_final.copy().astype(np.float32)
            image_mrcnn_keypoints_norm_input_final[:, :, 0] = image_mrcnn_keypoints_norm_input_final[:, :, 0]/self._img_pixel_size[1].astype(np.float32)
            image_mrcnn_keypoints_norm_input_final[:, :, 1] = image_mrcnn_keypoints_norm_input_final[:, :, 1]/self._img_pixel_size[0].astype(np.float32)

            # So here is my idea, the anchor bbox intersect with the feature bbox,
            # for each pixel in the anchor bbox, we calculate where that pixel is
            # in the feature bbox. If this pixel lies between [0, 28), then it means
            # that this pixel lies in the IoU of two bboxes, so that we can update this pixel.
            # (anchor_x1, anchor_y1)-----------------------------------------------
            # |                                                                   |
            # |                                                                   |
            # |                                                                   |
            # |                                                                   |
            # |                                                                   |
            # |                                                                   |
            # |                                                                   |
            # |                                                                   |
            # |                                                                   |
            # |                                                                   |
            # |                           (feature_x1, feature_y1)----------------|----------------
            # |                           |.......................................|               |
            # |                           |.......................................|               |
            # |                           |.......................................|               |
            # |                           |.......................................|               |
            # |---------------------------|------------------(anchor_x2, anchor_y2)               |
            #                             |                                                       |
            #                             |                                                       |
            #                             |                                                       |
            #                             |                                                       |
            #                             |-------------------------------(feature_x2, feature_y2)|
            image_mask_shifted_final = []
            image_mrcnn_feature_shifted_final = []
            image_mrcnn_bbox_shifted_final = []
            anchor_row = 0
            for i in range(len(self._img_repeat_times)):
                # Calculate the resolution of each pixel in feature map
                feature_y1 = image_mrcnn_bbox_input[i, 0]
                feature_x1 = image_mrcnn_bbox_input[i, 1]
                feature_y2 = image_mrcnn_bbox_input[i, 2]
                feature_x2 = image_mrcnn_bbox_input[i, 3]
                feature_res_x = (feature_x2-feature_x1)/28
                feature_res_y = (feature_y2-feature_y1)/28
                # print('feature_res_x = ', feature_res_x)
                # print('feature_res_y = ', feature_res_y)
                # print('self._img_repeat_times = ', self._img_repeat_times)
                for j in range(int(self._img_repeat_times[i])):
                    anchor_y1 = self._img_anchors[anchor_row, 0]
                    anchor_x1 = self._img_anchors[anchor_row, 1]
                    anchor_y2 = self._img_anchors[anchor_row, 2]
                    anchor_x2 = self._img_anchors[anchor_row, 3]
                    anchor_res_x = (anchor_x2-anchor_x1)/28
                    anchor_res_y = (anchor_y2-anchor_y1)/28
                    # print('anchor_res_x = ', anchor_res_x)
                    # print('anchor_res_y = ', anchor_res_y)
                    anchor_mask = np.zeros((1, 28, 28))
                    anchor_feature = np.zeros((1, 28, 28, 17))
                    anchor_feature_bbox = np.zeros((1, 4))
                    # First, calculate where the (feature_x1, feature_y1) and the
                    # (feature_x2, feature_y2) points are in the anchor bbox.
                    # We only need to update the shadow area.
                    if feature_x1 < anchor_x1:
                        start_x = int(0)
                    else:
                        start_x = int((feature_x1-anchor_x1)/anchor_res_x)

                    if feature_y1 < anchor_y1:
                        start_y = int(0)
                    else:
                        start_y = int((feature_y1-anchor_y1)/anchor_res_y)
                    if feature_x2 > anchor_x2:
                        end_x = int(28)
                    else:
                        end_x = int((feature_x2-anchor_x1)/anchor_res_x)
                    if feature_y2 > anchor_y2:
                        end_y = int(28)
                    else:
                        end_y = int((feature_y2-anchor_y1)/anchor_res_y)

                    for k in range(start_y, end_y):
                        for n in range(start_x, end_x):
                            img_x_idx = (anchor_x1 + n*anchor_res_x - feature_x1)/feature_res_x
                            if img_x_idx < 0:
                                img_x_idx = 0
                            if img_x_idx > 28:
                                img_x_idx = 28
                            img_y_idx = (anchor_y1 + k*anchor_res_y - feature_y1)/feature_res_y
                            if img_y_idx < 0:
                                img_y_idx = 0
                            if img_y_idx > 28:
                                img_y_idx = 28
                            anchor_mask[0, k, n] = image_mask_input[i, int(img_y_idx), int(img_x_idx)]
                            anchor_feature[0, k, n, :] = image_mrcnn_feature_input[i, int(img_y_idx), int(img_x_idx), :]
                            anchor_feature_bbox[0, :] = self._img_anchors[anchor_row, :]
                    anchor_row = anchor_row+1
                    if image_mask_shifted_final == []:
                        image_mask_shifted_final = anchor_mask
                        image_mrcnn_feature_shifted_final = anchor_feature
                        image_mrcnn_bbox_shifted_final = anchor_feature_bbox
                    else:
                        image_mask_shifted_final = np.concatenate((image_mask_shifted_final, anchor_mask), axis=0)
                        image_mrcnn_feature_shifted_final = np.concatenate((image_mrcnn_feature_shifted_final, anchor_feature), axis=0)
                        image_mrcnn_bbox_shifted_final = np.concatenate((image_mrcnn_bbox_shifted_final, anchor_feature_bbox), axis=0)
            # print('image_mask_shifted_final.shape = ', image_mask_shifted_final.shape)  # (270, 28, 28) = (M, 28, 28) M is the number of all candidates

            self._placeholder_inputs[self.PL_IMG_MASK_SHIFTED] = image_mask_shifted_final  # image_mask_shifted.shape =  (M, 28, 28), already shifted and cropped
            self._placeholder_inputs[self.PL_IMG_MRCNN_FEATURE_SHIFTED] = image_mrcnn_feature_shifted_final  # image_mrcnn_feature_shifted.shape =  M, 28, 28, 17), already shifted and cropped
            print('image_mrcnn_feature_shifted_final.shape = ', image_mrcnn_feature_shifted_final.shape)
            self._placeholder_inputs[self.PL_IMG_MRCNN_BBOX_SHIFTED] = image_mrcnn_bbox_shifted_final  # image_mrcnn_bbox_shifted.shape =  (M, 4), already shifted and cropped
            print('image_mask_input_final.shape = ', image_mask_input_final.shape)
            self._placeholder_inputs[self.PL_IMG_MASK_INPUT] = image_mask_input_final  # image_mask_input.shape =  (M, 28, 28), already tiled
            print('image_mrcnn_feature_input_final.shape = ', image_mrcnn_feature_input_final.shape)
            print('image_mrcnn_bbox_input_final.shape = ', image_mrcnn_bbox_input_final.shape)
            print('image_mrcnn_keypoints_input_final.shape = ', image_mrcnn_keypoints_input_final.shape)
            print('image_mrcnn_keypoints_norm_input_final.shape = ', image_mrcnn_keypoints_norm_input_final.shape)
            self._placeholder_inputs[self.PL_IMG_MRCNN_FEATURE_INPUT] = image_mrcnn_feature_input_final  # image_mrcnn_feature_input.shape =  M, 28, 28, 17), already tiled
            self._placeholder_inputs[self.PL_IMG_MRCNN_BBOX_INPUT] = image_mrcnn_bbox_input_final  # image_mrcnn_bbox_input.shape =  (M, 4), already tiled
            self._placeholder_inputs[self.PL_IMG_MRCNN_KEYPOINTS_INPUT] = image_mrcnn_keypoints_input_final  # image_mrcnn_keypoints_input.shape =  (M, 17, 2), already tiled, in full-size pixel value.
            self._placeholder_inputs[self.PL_IMG_MRCNN_KEYPOINTS_NORM_INPUT] = image_mrcnn_keypoints_norm_input_final  # image_mrcnn_keypoints_norm_input.shape =  (M, 17, 2), already tiled, in full-size pixel value.

            self._placeholder_inputs[self.PL_ORIENT_PRED] = orient_pred_final  # orient_pred.shape =  (M), M is the number of all candidates, already tiled.
            self._placeholder_inputs[self.PL_CROP_GROUP] = crop_group_final  # crop_group_final.shape =  (M), M is the number of all candidates, already tiled.

        resize_feature_time = time.time()
        print('resize_feature takes ', (resize_feature_time-read_orient_time), 's')
        # 0.6s~1.2s. Method No.1 is usually faster than Method No.2 if the
        # number of people is large.
        # this is a list to match the explicit shape for the placeholder
        self._placeholder_inputs[self.PL_IMG_IDX] = [int(sample_name)]

        # Fill in the rest
        # print('-=-=-=-=-=-=-= line876(rpn model) label_anchors = ', label_anchors)
        self._placeholder_inputs[self.PL_LABEL_ANCHORS] = label_anchors
        self._placeholder_inputs[self.PL_LABEL_BOXES_3D] = label_boxes_3d
        self._placeholder_inputs[self.PL_LABEL_BOXES_2D] = label_boxes_2d
        self._placeholder_inputs[self.PL_LABEL_BOXES_2D_NORM] = label_boxes_2d_norm
        self._placeholder_inputs[self.PL_LABEL_CLASSES] = label_classes
        self._placeholder_inputs[self.PL_LABEL_BOXES_QUATERNION] = label_boxes_quat

        # Sample Info
        # img_idx is a list to match the placeholder shape
        self._placeholder_inputs[self.PL_IMG_IDX] = [int(sample_name)]
        self._placeholder_inputs[self.PL_CALIB_P2] = stereo_calib_p2
        self._placeholder_inputs[self.PL_GROUND_PLANE] = ground_plane

        # Temporary sample info for debugging
        self.sample_info.clear()
        self.sample_info['sample_name'] = sample_name
        self.sample_info['rpn_mini_batch'] = anchors_info
        fill_rest_time = time.time()
        print('fill_rest takes ', (fill_rest_time-resize_feature_time), 's')

        # Create a feed_dict and fill it with input values
        feed_dict = dict()
        # print('image_mrcnn_bbox_input = ', image_mrcnn_bbox_input)
        if len(image_mrcnn_bbox_input) == 0 or np.array_equal(self._img_repeat_times, [0]):
            print('WARNING, No element!!!! !!!!!This is a hack !!!!!')
        # elif self._img_repeat_times == [1]:
        #     print('WARNING, img_repeat_times == [1] !!!!!This is a hack !!!!!')
        else:
            for key, value in self.placeholders.items():
                feed_dict[value] = self._placeholder_inputs[key]

        # print('---------------- pplp/core/models/rpn_panoptic_model.py end------------------')
        feed_dict_time = time.time()
        print('feed_dict takes ', (feed_dict_time-fill_rest_time), 's')
        end_create_feed_dict = time.time()
        print('create_feed_dict takes ', (end_create_feed_dict-start_create_feed_dict), 's')
        return feed_dict

    def _fill_anchor_pl_inputs(self,
                               anchors_info,
                               ground_plane,
                               image_shape,
                               stereo_calib_p2,
                               sample_name,
                               sample_augs):
        """
        Fills anchor placeholder inputs with corresponding data

        Args:
            anchors_info: anchor info from mini_batch_panoptic_utils
            ground_plane: ground plane coefficients
            image_shape: image shape (h, w), used for projecting anchors
            sample_name: name of the sample, e.g. "000001"
            sample_augs: list of sample augmentations
        """

        # print('^^^^^^^^^^ rpn_panoptic_model.py _fill_anchor_pl_inputs() ^^^^^^^^^^')
        # Lists for merging anchors info
        all_anchor_boxes_3d = []
        anchors_ious = []
        anchor_offsets = []
        anchor_classes = []

        # Create anchors for each class
        if len(self.dataset.classes) > 1:
            for class_idx in range(len(self.dataset.classes)):
                # Generate anchors for all classes
                grid_anchor_boxes_3d = self._anchor_generator.generate(
                    area_3d=self._area_extents,
                    anchor_3d_sizes=self._cluster_sizes[class_idx],
                    anchor_stride=self._anchor_strides[class_idx],
                    ground_plane=ground_plane)
                all_anchor_boxes_3d.append(grid_anchor_boxes_3d)
            all_anchor_boxes_3d = np.concatenate(all_anchor_boxes_3d)
        else:
            # Don't loop for a single class
            class_idx = 0
            grid_anchor_boxes_3d = self._anchor_generator.generate(
                area_3d=self._area_extents,
                anchor_3d_sizes=self._cluster_sizes[class_idx],
                anchor_stride=self._anchor_strides[class_idx],
                ground_plane=ground_plane)
            all_anchor_boxes_3d = grid_anchor_boxes_3d
        # print('self._area_extents = ', self._area_extents)
        # print('self._cluster_sizes[', class_idx, '] = ', self._cluster_sizes[class_idx])
        # print('self._anchor_strides[', class_idx, '] = ', self._anchor_strides[class_idx])
        # print('ground_plane = ', ground_plane)
        # print('all_anchor_boxes_3d[0] = ', all_anchor_boxes_3d[0])
        # print('all_anchor_boxes_3d.shape = ', all_anchor_boxes_3d.shape)

        # Filter empty anchors
        # Skip if anchors_info is []
        sample_has_labels = True
        # print('self._train_val_test = ', self._train_val_test)
        if self._train_val_test in ['train', 'val']:
            # Read in anchor info during training / validation
            if anchors_info:
                anchor_indices, anchors_ious, anchor_offsets, \
                    anchor_classes = anchors_info
                # print('anchor_indices = ', anchor_indices)
                # print('anchors_ious = ', anchors_ious)
                # print('anchor_offsets = ', anchor_offsets)
                # print('anchor_classes = ', anchor_classes)
                anchor_boxes_3d_to_use = all_anchor_boxes_3d[anchor_indices]
                # After this sellection, around 20 ~ 40 boxes are left
                # Now, anchor_boxes_3d_to_use is just a bunch of non-repeat anchors.
            else:
                train_cond = (self._train_val_test == "train" and
                              self._train_on_all_samples)
                eval_cond = (self._train_val_test == "val" and
                             self._eval_all_samples)
                if train_cond or eval_cond:
                    sample_has_labels = False
        else:
            sample_has_labels = False

        # print('sample_has_labels = ', sample_has_labels)
        if not sample_has_labels:
            # During testing, or validation with no anchor info, manually
            # filter empty anchors
            # TODO: share voxel_grid_2d with BEV generation if possible
            voxel_grid_2d = \
                self.dataset.panoptic_utils.create_sliced_voxel_grid_2d(
                    sample_name, self.dataset.bev_source,
                    image_shape=image_shape)

            # Convert to anchors and filter
            anchors_to_use = box_3d_panoptic_encoder.box_3d_to_anchor(
                all_anchor_boxes_3d)
            empty_filter = anchor_panoptic_filter.get_empty_anchor_filter_2d(
                anchors_to_use, voxel_grid_2d, density_threshold=1)

            anchor_boxes_3d_to_use = all_anchor_boxes_3d[empty_filter]

            # A very special case here:
            # Sometimes, no pedestrian detected by lidar, but some pedestrians can
            # be seen by camera.
            # In this case we also have groundtruth, but for each candidate anchor,
            # iou = 0.
            # Let's just return zeros for ious, offsets and classes. The training
            # and evaluation will still count in this sample.
            anchors_ious = np.zeros(anchor_boxes_3d_to_use.shape[0])
            anchor_offsets = np.zeros((anchor_boxes_3d_to_use.shape[0], 6))
            anchor_classes = np.zeros(anchor_boxes_3d_to_use.shape[0])

        # Convert lists to ndarrays
        anchor_boxes_3d_to_use = np.asarray(anchor_boxes_3d_to_use)
        anchors_ious = np.asarray(anchors_ious)
        anchor_offsets = np.asarray(anchor_offsets)
        anchor_classes = np.asarray(anchor_classes)

        # Flip anchors and centroid x offsets for augmented samples
        if panoptic_aug.AUG_FLIPPING in sample_augs:
            # print('panoptic_aug.AUG_FLIPPING = ', panoptic_aug.AUG_FLIPPING)
            anchor_boxes_3d_to_use = panoptic_aug.flip_boxes_3d(
                anchor_boxes_3d_to_use, flip_ry=False)
            if anchors_info:
                anchor_offsets[:, 0] = -anchor_offsets[:, 0]

        # Convert to anchors
        # box_3d [x, y, z, l, w, h, ry]
        # anchor form [x, y, z, dim_x, dim_y, dim_z]
        anchors_to_use = box_3d_panoptic_encoder.box_3d_to_anchor(
            anchor_boxes_3d_to_use)
        # num_anchors = len(anchors_to_use)

        # Project anchors into bev
        # bev_anchors is measured in meters
        # bev_anchors_norm is measured in percentage, i.e. from 0 to 1.
        # A normalized coordinate value of y is mapped to the image coordinate at y * (image_height - 1)
        bev_anchors, bev_anchors_norm = anchor_panoptic_projector.project_to_bev(
            anchors_to_use, self._bev_extents)
        # N x [x1, y1, x2, y2]. Origin is the top left corner
        # --------------------------------------------------------------
        # |                      |                                 |   |
        # |                    y1|                                 |   |
        # |                                 |----------|         y2|   |
        # |                                 |          |           |   |
        # |                                 |          |           |   |
        # |  ------------- x1 ------------- |----------|           |   |
        # |                                                            |
        # |                                                            |
        # | -----------------  x2  ---------------------               |
        # |                                                            |
        # |                                                            |
        # |                                                            |
        # |                                                            |
        # ------------------------ camera ------------------------------

        # Project box_3d anchors into image space
        # img_anchors are used both for shifted image features and the unshifted image features.
        img_anchors, img_anchors_norm = \
            anchor_panoptic_projector.project_to_image_space(
                anchors_to_use, stereo_calib_p2, image_shape)

        # Find correcsponding anchor cadidates w.r.t. anchors img_mrcnn_bbox
        # N x [x1, y1, x2, y2]. Origin is the top left corner, same as above
        # image_mrcnn_bbox_input: [batch, N, (y1, x1, y2, x2)] detection bounding boxes
        bev_anchors_norm_final = []
        bev_anchors_final = []
        img_anchors_norm_final = []
        img_anchors_final = []

        img_repeat_times_final = []
        anchors_to_use_final = []
        anchors_ious_final = []
        anchor_offsets_final = []
        anchor_classes_final = []
        mrcnn_bbox = self.image_mrcnn_bbox_input[:, [1, 0, 3, 2]]
        # This b-box is in pixels.
        # print('mrcnn_bbox = ', mrcnn_bbox)

        # Now pair the mrcnn bbox with bev anchor bboxes.
        # One mrcnn bbox may pair with more than one bev anchor bboxes. In this
        # case, we duplicate that mrcnn bboxes for sevral times until we make
        # enough pairs.
        # Accordingly, these parameters has to be tiled:
        # num_anchors, bev_anchors, bev_anchors_norm, img_anchors, img_anchors_norm, anchors_ious, anchor_offsets, anchor_classes
        # !!!!! Remember, img_anchors, img_anchors_norm comes from Mask RCNN!!!!

        # np.set_printoptions(threshold=np.nan)
        # print('anchor_boxes_3d_to_use.shape = ', anchor_boxes_3d_to_use.shape)  # (36, 7)
        # print('anchors_ious.shape = ', anchors_ious.shape)  # (36,)
        # print('anchor_offsets.shape = ', anchor_offsets.shape)  # (36, 6)
        # print('anchor_classes.shape = ', anchor_classes.shape)  # (36,)
        # print('bev_anchors.shape = ', bev_anchors.shape)  # (36, 4)
        # print('bev_anchors_norm.shape = ', bev_anchors_norm.shape)  # (36, 4)
        # print('img_anchors.shape = ', img_anchors.shape)  # (36, 4)
        # print('img_anchors_norm.shape = ', img_anchors_norm.shape)  # (36, 4)
        for row in mrcnn_bbox:
            ious = evaluation.two_d_iou(row,
                                        img_anchors)
            # print('!!!!! Warning !!!!! Only match with the best IoU')
            ious_filters = [x > 0.0 for x in ious]  # TODO: may need to change the threshold here.
            # ious_filters = np.argmax(ious)  # Only match with the best match here!.
            bev_anchors_norm_result = bev_anchors_norm[ious_filters]
            bev_anchors_result = bev_anchors[ious_filters]
            # print('row = ', row)
            # print('ious = ', ious)
            # print('anchors_to_use = ', anchors_to_use)
            # print('ious_filters = ', ious_filters)
            # print('bev_anchors_result = ', bev_anchors_result)
            # print('anchors_ious = ', anchors_ious)
            # print('anchor_offsets = ', anchor_offsets)
            # print('anchor_classes = ', anchor_classes)
            anchors_to_use_result = anchors_to_use[ious_filters]
            anchors_ious_final = np.append(anchors_ious_final, anchors_ious[ious_filters])  # (N,)
            anchor_offsets_result = anchor_offsets[ious_filters]  # (N, 6)
            anchor_classes_final = np.append(anchor_classes_final, anchor_classes[ious_filters])  # (N,)
            # print('Another hack for debugging!!!!!!!')
            # print('anchors_to_use[0] = ', anchors_to_use[0])
            ious_number = np.sum(ious_filters)  # In case ious_number = 1, we need to resize the arrray into 2d array.
            anchors_to_use_result = np.reshape(anchors_to_use_result, [ious_number, 6])
            bev_anchors_norm_result = np.reshape(bev_anchors_norm_result, [ious_number, 4])
            bev_anchors_result = np.reshape(bev_anchors_result, [ious_number, 4])
            anchor_offsets_result = np.reshape(anchor_offsets_result, [ious_number, 6])
            # print('bev_anchors_norm_result = ', bev_anchors_norm_result)

            img_repeat_times = int(ious_number)
            img_repeat_times_final = np.append(img_repeat_times_final, img_repeat_times)

            # If we want to use the same image feature for all the anchors: (Also need to change the strategy in function create_feed_dict())
            # image_shape(h, w)
            # Now, let's make the b-boxes matrix for BEV and image crops.
            if not self._img_feature_shifted:  # Then we repeat the same 2D crop boxes for 'img_repeat_times' times.
                row_norm = [row[0]/image_shape[1], row[1]/image_shape[0], row[2]/image_shape[1], row[3]/image_shape[0]]
                img_anchors_norm_result = np.tile(row_norm, (img_repeat_times, 1)) # This b-box is in norm, which is >0 and <1.
                img_anchors_result = np.tile(row, (img_repeat_times, 1)) # This b-box is in pixels.

            # If we want to use different image features for different the anchors: (Also need to change the strategy in function create_feed_dict())
            else:
                img_anchors_norm_result = img_anchors_norm[ious_filters]
                img_anchors_result = img_anchors[ious_filters]
                # print('img_anchors_norm_result = ', img_anchors_norm_result)
                #     In case ious_number = 1, we need to resize the arrray into 2d array.
                img_anchors_norm_result = np.reshape(img_anchors_norm_result, [ious_number, 4])
                img_anchors_result = np.reshape(img_anchors_result, [ious_number, 4])

            if bev_anchors_norm_final == []:
                bev_anchors_norm_final = bev_anchors_norm_result
                bev_anchors_final = bev_anchors_result
                img_anchors_norm_final = img_anchors_norm_result
                img_anchors_final = img_anchors_result
                anchors_to_use_final = anchors_to_use_result
                anchor_offsets_final = anchor_offsets_result
            else:
                bev_anchors_norm_final = np.concatenate((bev_anchors_norm_final, bev_anchors_norm_result), axis=0)
                bev_anchors_final = np.concatenate((bev_anchors_final, bev_anchors_result), axis=0)
                img_anchors_norm_final = np.concatenate((img_anchors_norm_final, img_anchors_norm_result), axis=0)
                img_anchors_final = np.concatenate((img_anchors_final, img_anchors_result), axis=0)
                anchors_to_use_final = np.concatenate((anchors_to_use_final, anchors_to_use_result), axis=0)
                anchor_offsets_final = np.concatenate((anchor_offsets_final, anchor_offsets_result), axis=0)
        # print('&&&&&&& useful ious = ', ious[ious_filters])
        # print('bev_anchors_norm_final = ', bev_anchors_norm_final)
        # print('img_anchors_final = ', img_anchors_final)
        # print('img_repeat_times_final = ', img_repeat_times_final)

        # Reorder into [y1, x1, y2, x2] for tf.crop_and_resize op. Only change the order of normed bboxes.
        self._bev_anchors_norm = bev_anchors_norm_final[:, [1, 0, 3, 2]]
        self._img_anchors_norm = img_anchors_norm_final[:, [1, 0, 3, 2]]  # Only used when self._img_feature_shifted = False
        self._bev_anchors = bev_anchors_final[:, [1, 0, 3, 2]]  # Only used when self._img_feature_shifted = False
        self._img_anchors = img_anchors_final[:, [1, 0, 3, 2]]  # Only used when self._img_feature_shifted = False
        self._img_repeat_times = img_repeat_times_final

        # Fill in placeholder inputs
        self._placeholder_inputs[self.PL_ANCHORS] = anchors_to_use_final
        num_anchors = len(anchors_to_use_final)

        # If we are in train/validation mode, and the anchor infos
        # are not empty, store them. Checking for just anchors_ious
        # to be non-empty should be enough.
        if self._train_val_test in ['train', 'val'] and \
                len(anchors_ious_final) > 0:
            self._placeholder_inputs[self.PL_ANCHOR_IOUS] = anchors_ious_final
            self._placeholder_inputs[self.PL_ANCHOR_OFFSETS] = anchor_offsets_final
            # mini_batch_mask = tf.Print(mini_batch_mask, ['line 694(rpn) : mini_batch_mask =', mini_batch_mask], summarize=100)
            self._placeholder_inputs[self.PL_ANCHOR_CLASSES] = anchor_classes_final
            # print('^^^^^^^^^^^^^^^^^^^^^^^ line 1233(rpn), anchors_ious_final = ', anchors_ious_final)
            # print('^^^^^^^^^^^^^^^^^^^^^^^ line 1234(rpn), anchor_offsets_final = ', anchor_offsets_final)
            # print('^^^^^^^^^^^^^^^^^^^^^^^ line 1235(rpn), anchor_classes_final = ', anchor_classes_final)

        # During test, or val when there is no anchor info
        elif self._train_val_test in ['test'] or \
                len(anchors_ious_final) == 0:
            # During testing, or validation with no gt, fill these in with 0s
            self._placeholder_inputs[self.PL_ANCHOR_IOUS] = \
                np.zeros(num_anchors)
            self._placeholder_inputs[self.PL_ANCHOR_OFFSETS] = \
                np.zeros([num_anchors, 6])
            self._placeholder_inputs[self.PL_ANCHOR_CLASSES] = \
                np.zeros(num_anchors)
        else:
            raise ValueError('Got run mode {}, and non-empty anchor info'.
                             format(self._train_val_test))

        self._placeholder_inputs[self.PL_BEV_ANCHORS] = bev_anchors_final  # shape = Nx4
        self._placeholder_inputs[self.PL_BEV_ANCHORS_NORM] = \
            self._bev_anchors_norm
        self._placeholder_inputs[self.PL_IMG_ANCHORS] = img_anchors_final  # shape = Nx4, tiled
        self._placeholder_inputs[self.PL_IMG_ANCHORS_NORM] = \
            self._img_anchors_norm  # Only used when self._img_feature_shifted = False
        self._placeholder_inputs[self.PL_IMG_REPEAT_TIMES] = \
            self._img_repeat_times
        # print('^^^^^^^^^^ rpn_pplp lin1643: self._img_repeat_times = ', self._img_repeat_times)

    def loss(self, prediction_dict):

        # these should include mini-batch values only
        objectness_gt = prediction_dict[self.PRED_MB_OBJECTNESS_GT]
        # objectness_gt = tf.Print(objectness_gt, ['^^^^^^^^^^^^^^^^^^^^^line 1258(pplp) : objectness_gt =', objectness_gt], summarize=1000)
        offsets_gt = prediction_dict[self.PRED_MB_OFFSETS_GT]
        # offsets_gt = tf.Print(offsets_gt, ['^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^line 1125(pplp) : offsets_gt =', offsets_gt], summarize=1000)

        # Predictions
        with tf.variable_scope('rpn_prediction_mini_batch'):
            objectness = prediction_dict[self.PRED_MB_OBJECTNESS]
            offsets = prediction_dict[self.PRED_MB_OFFSETS]
            # offsets = tf.Print(offsets, ['^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^line 1204(pplp) : offsets =', offsets], summarize=1000)

        with tf.variable_scope('rpn_losses'):
            with tf.variable_scope('objectness'):
                cls_loss = losses.WeightedSoftmaxLoss()
                cls_loss_weight = self._config.loss_config.cls_loss_weight
                objectness_loss = cls_loss(objectness,
                                           objectness_gt,
                                           weight=cls_loss_weight)

                with tf.variable_scope('obj_norm'):
                    # normalize by the number of anchor mini-batches
                    objectness_loss = objectness_loss / tf.cast(
                        tf.shape(objectness_gt)[0], dtype=tf.float32)
                    tf.summary.scalar('objectness', objectness_loss)

            with tf.variable_scope('regression'):
                reg_loss = losses.WeightedSmoothL1Loss()
                reg_loss_weight = self._config.loss_config.reg_loss_weight
                anchorwise_localization_loss = reg_loss(offsets,
                                                        offsets_gt,
                                                        weight=reg_loss_weight)
                masked_localization_loss = \
                    anchorwise_localization_loss * objectness_gt[:, 1]
                localization_loss = tf.reduce_sum(masked_localization_loss)

                with tf.variable_scope('reg_norm'):
                    # normalize by the number of positive objects
                    num_positives = tf.reduce_sum(objectness_gt[:, 1])  # If objectness_gt = [], error will occur.
                    # Assert the condition `num_positives > 0`
                    with tf.control_dependencies(
                            [tf.assert_positive(num_positives)]):
                        localization_loss = localization_loss / num_positives
                        tf.summary.scalar('regression', localization_loss)

            with tf.variable_scope('total_loss'):
                total_loss = objectness_loss + localization_loss

        loss_dict = {
            self.LOSS_RPN_OBJECTNESS: objectness_loss,
            self.LOSS_RPN_REGRESSION: localization_loss,
        }

        return loss_dict, total_loss

    def create_path_drop_masks(self,
                               p_img,
                               p_bev,
                               random_values):
        """Determines global path drop decision based on given probabilities.

        Args:
            p_img: A tensor of float32, probability of keeping image branch
            p_bev: A tensor of float32, probability of keeping bev branch
            random_values: A tensor of float32 of shape [3], the results
                of coin flips, values should range from 0.0 - 1.0.

        Returns:
            final_img_mask: A constant tensor mask containing either one or zero
                depending on the final coin flip probability.
            final_bev_mask: A constant tensor mask containing either one or zero
                depending on the final coin flip probability.
        """

        def keep_branch(): return tf.constant(1.0)

        def kill_branch(): return tf.constant(0.0)

        # The logic works as follows:
        # We have flipped 3 coins, first determines the chance of keeping
        # the image branch, second determines keeping bev branch, the third
        # makes the final decision in the case where both branches were killed
        # off, otherwise the initial img and bev chances are kept.

        img_chances = tf.case([(tf.less(random_values[0], p_img),
                                keep_branch)], default=kill_branch)

        bev_chances = tf.case([(tf.less(random_values[1], p_bev),
                                keep_branch)], default=kill_branch)

        # Decision to determine whether both branches were killed off
        third_flip = tf.logical_or(tf.cast(img_chances, dtype=tf.bool),
                                   tf.cast(bev_chances, dtype=tf.bool))
        third_flip = tf.cast(third_flip, dtype=tf.float32)

        # Make a second choice, for the third case
        # Here we use a 50/50 chance to keep either image or bev
        # If its greater than 0.5, keep the image
        img_second_flip = tf.case([(tf.greater(random_values[2], 0.5),
                                    keep_branch)],
                                  default=kill_branch)
        # If its less than or equal to 0.5, keep bev
        bev_second_flip = tf.case([(tf.less_equal(random_values[2], 0.5),
                                    keep_branch)],
                                  default=kill_branch)

        # Use lambda since this returns another condition and it needs to
        # be callable
        final_img_mask = tf.case([(tf.equal(third_flip, 1),
                                   lambda: img_chances)],
                                 default=lambda: img_second_flip)

        final_bev_mask = tf.case([(tf.equal(third_flip, 1),
                                   lambda: bev_chances)],
                                 default=lambda: bev_second_flip)

        return final_img_mask, final_bev_mask
