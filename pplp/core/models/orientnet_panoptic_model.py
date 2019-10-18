import numpy as np

import tensorflow as tf

from pplp.builders import orientnet_layers_builder
from pplp.builders import orientnet_loss_builder

from pplp.core import constants

from pplp.core import model
from pplp.core import orientation_encoder

from transforms3d.quaternions import quat2mat, mat2quat
from transforms3d.euler import euler2mat, mat2euler


class OrientModel(model.DetectionModel):
    ##############################
    # Keys for Predictions
    ##############################
    # inputs
    PL_IMG_INPUT = 'img_input_pl'
    PL_IMG_IDX = 'img_index_pl'
    PL_IMG_MASK_28X28_INPUT = 'img_mask_input_pl'
    PL_IMG_MRCNN_BBOX_INPUT = 'img_mrcnn_bbox_input_pl'

    # groundtruths
    PL_LABEL_BOXES_QUATERNION = 'label_boxes_quat_pl'
    PL_LABEL_BOXES_3D = 'label_boxes_3d_pl'
    PL_LABEL_VALID_MASK = 'label_valid_mask_pl'
    PRED_ORIENTATIONS_GT = 'pred_orientations_gt'
    PRED_ORIENTATIONS_QUATS_GT = 'pred_orient_quats_gt'
    PRED_VALID_MASK = 'pred_valid_mask'
    PRED_ANGLE_VECTORS = 'pred_angle_vectors'
    PRED_ANGLES = 'pred_angles'
    PRED_BOXES_3D = 'pred_boxes_3d'  # It is just a copy of LABEL_BOXES_3D, for the convenience of verification.

    ##############################
    # Keys for Loss
    ##############################
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
        super(OrientModel, self).__init__(model_config)

        if train_val_test not in ["train", "val", "test"]:
            raise ValueError('Invalid train_val_test value,'
                             'should be one of ["train", "val", "test"]')
        self._train_val_test = train_val_test

        self._is_training = (self._train_val_test == 'train')

        # Input config
        input_config = self._config.input_config
        self._bev_pixel_size = np.asarray([input_config.bev_dims_h,
                                           input_config.bev_dims_w])
        self._bev_depth = input_config.bev_depth

        self._img_pixel_size = np.asarray([input_config.img_dims_h,
                                           input_config.img_dims_w])
        self._img_depth = input_config.img_depth

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

        # Dataset config
        self._num_final_classes = self.dataset.num_classes + 1

        # AVOD config
        avod_config = self._config.avod_config
        self._path_drop_probabilities = self._config.path_drop_probabilities
        self._box_rep = avod_config.avod_box_representation

        if self._box_rep not in ['box_3d', 'box_8c', 'box_8co',
                                 'box_4c', 'box_4ca']:
            raise ValueError('Invalid box representation', self._box_rep)

    def _add_placeholder(self, dtype, shape, name):
        placeholder = tf.placeholder(dtype, shape, name)
        self.placeholders[name] = placeholder
        return placeholder

    def _set_up_input_pls(self, testing=False):
        """Sets up input placeholders by adding them to self._placeholders.
        Keys are defined as self.PL_*.
        """

        with tf.variable_scope('img_input'):  # We need it for summary only
            # Take variable size input images
            img_input_placeholder = self._add_placeholder(
                tf.float32,
                [None, None, self._img_depth],
                self.PL_IMG_INPUT)

            self._img_input_batches = tf.expand_dims(
                img_input_placeholder, axis=0)

        # ------------- For feature maps directly from Mask RCNN: -------------
        # We need this block even if self._img_feature_shifted = True
        with tf.variable_scope('img_mrcnn_mask_input'):
            # Placeholder for image masks or full_masks
            # full_masks: (dtype=uint8), [batch, height, width, N] Instance masks
            # masks: array(dtype=float32), [N, 28, 28]
            self._img_mrcnn_mask_input_pl = self._add_placeholder(
                tf.float32,
                [None, 28, 28],
                self.PL_IMG_MASK_28X28_INPUT)

            self.img_mrcnn_mask_input_tile_17_pl = tf.tile(
                tf.reshape(self._img_mrcnn_mask_input_pl, [-1, 28, 28, 1]),
                [1, 1, 1, 17],
                name=None
            )
            # Now tf.shape(self.img_mrcnn_mask_input_tile_17_pl) = [N, 28, 28, 17]

        with tf.variable_scope('img_mrcnn_bbox_input'):
            # Placeholder for image keypoint features:
            # bbox: [batch, N, (y1, x1, y2, x2)] detection bounding boxes
            # This b-box is in pixels.
            self._img_mrcnn_bbox_input_pl = self._add_placeholder(
                tf.float32,
                [None, 4],
                self.PL_IMG_MRCNN_BBOX_INPUT)

            y1 = self._img_mrcnn_bbox_input_pl[:, 0]/tf.cast(tf.shape(img_input_placeholder)[0], tf.float32)
            x1 = self._img_mrcnn_bbox_input_pl[:, 1]/tf.cast(tf.shape(img_input_placeholder)[1], tf.float32)
            y2 = self._img_mrcnn_bbox_input_pl[:, 2]/tf.cast(tf.shape(img_input_placeholder)[0], tf.float32)
            x2 = self._img_mrcnn_bbox_input_pl[:, 3]/tf.cast(tf.shape(img_input_placeholder)[1], tf.float32)
            normed_boxes = tf.stack([y1, x1, y2, x2], axis=1)
            # Crop the whole image into boxes, and resize each box into 64x64x3
            img_croped_input_64x64x3 = tf.image.crop_and_resize(
                self._img_input_batches,
                normed_boxes,
                tf.zeros([tf.shape(self._img_mrcnn_bbox_input_pl)[0]], tf.int32),
                [64, 64])

            # img_mrcnn_mask_input_28x28x3 = tf.tile(
            #     tf.reshape(self._img_mrcnn_mask_input_pl, [-1, 28, 28, 1]),
            #     [1, 1, 1, 3],
            #     name=None
            # )
            #
            # Make croped mask for 64x64 input
            img_mrcnn_mask_64x64 = tf.image.resize_images(
                tf.reshape(self._img_mrcnn_mask_input_pl, [-1, 28, 28, 1]),  # shape=(?, 28, 28，1)
                [64, 64]
                # method=ResizeMethod.BILINEAR,
                # align_corners=True
                # preserve_aspect_ratio=False
            )
            img_mrcnn_mask_64x64x3 = tf.tile(
                tf.reshape(img_mrcnn_mask_64x64, [-1, 64, 64, 1]),
                [1, 1, 1, 3],
                name=None
            )
            # Set non-mask values to zeros
            self.img_croped_masked_input = tf.multiply(img_croped_input_64x64x3,  tf.cast(img_mrcnn_mask_64x64x3, dtype=tf.float32))

        if not testing:
            with tf.variable_scope('pl_labels'):
                self.boxes_3d_gt = self._add_placeholder(tf.float32, [None, 7],
                                                         self.PL_LABEL_BOXES_3D)
                self.quats_gt = self._add_placeholder(tf.float32, [None, 4],
                                                      self.PL_LABEL_BOXES_QUATERNION)

                self.valid_mask = self._add_placeholder(tf.bool, [None],
                                                        self.PL_LABEL_VALID_MASK)

    def build(self, testing=False):
        # Setup input placeholders
        self._set_up_input_pls(testing=testing)

        rgb_img_crop = self.img_croped_masked_input  # shape=(?, 64, 64, 3)
        # rgb_img_crop = tf.Print(rgb_img_crop, ['line 204(orient) : tf.shape(rgb_img_crop) =', tf.shape(rgb_img_crop)], summarize=100)

        img_input_batches = self._img_input_batches  # shape=(?, 1920, 1080, 3)

        # Fully connected layers (Box Predictor)
        avod_layers_config = self.model_config.layers_config.avod_config
        # img_rois = tf.Print(img_rois, ['line 259(pplp) : tf.shape(img_rois) =', tf.shape(img_rois)], summarize=100)
        # bev_rois = tf.Print(bev_rois, ['line 260(pplp) : tf.shape(bev_rois) =', tf.shape(bev_rois)], summarize=100)
        fc_layers_type = avod_layers_config.WhichOneof('fc_layers')
        if fc_layers_type == 'resnet_fc_layers':
            fc_output_layers = \
                orientnet_layers_builder.build(
                    layers_config=avod_layers_config,
                    is_training=self._is_training,
                    rgb_img_crop=rgb_img_crop)
        else:
            raise NotImplementedError(
                'Unsupport fc_layers_type !!!!', self._box_rep)

        # This may be None
        all_angle_vectors = \
            fc_output_layers.get(orientnet_layers_builder.KEY_ANGLE_VECTORS)  # Not normed.

        ######################################################
        # Subsample mini_batch for the loss function
        ######################################################
        if not testing:
            if self._box_rep in ['box_3d', 'box_4ca']:
                orientations_gt = self.boxes_3d_gt[:, 6]
                orient_quats_gt = self.quats_gt
                valid_mask = self.valid_mask
            else:
                raise NotImplementedError('Ground truth tensors not implemented')

            # Encode anchor offsets
            with tf.variable_scope('avod_encode_mb_anchors'):
                if self._box_rep in ['box_4ca']:
                    # Gather corresponding ground truth orientation
                    mb_orientations_gt = orientations_gt
                    mb_orient_quats_gt = orient_quats_gt
                    mb_valid_mask = valid_mask

                else:
                    raise NotImplementedError(
                        'Anchor encoding not implemented for', self._box_rep)

        ######################################################
        # Final Predictions
        ######################################################
        # Get orientations from angle vectors
        avod_layers_config = self.model_config.layers_config.avod_config
        fc_layers_type = avod_layers_config.WhichOneof('fc_layers')
        if all_angle_vectors is not None:
            if fc_layers_type == 'resnet_fc_layers':
                with tf.variable_scope('pplp_orientation'):
                    # all_angle_vectors = tf.Print(all_angle_vectors, ['line 611(pplp) : all_angle_vectors =', all_angle_vectors], summarize=1000)
                    all_orientations = \
                        orientation_encoder.tf_angle_quats_to_orientation(
                            all_angle_vectors)
                    # all_orientations = tf.Print(all_orientations, ['line 615(pplp) : all_orientations =', all_orientations], summarize=1000)
            else:
                with tf.variable_scope('pplp_orientation'):
                    all_orientations = \
                        orientation_encoder.tf_angle_vector_to_orientation(
                            all_angle_vectors)

        prediction_dict = dict()

        if not testing:
            prediction_dict[self.PRED_BOXES_3D] = \
                self.boxes_3d_gt

        if self._box_rep == 'box_4ca':
            if self._train_val_test in ['train', 'val']:
                prediction_dict[self.PRED_ORIENTATIONS_GT] = \
                    mb_orientations_gt
                prediction_dict[self.PRED_ORIENTATIONS_QUATS_GT] = \
                    mb_orient_quats_gt
                prediction_dict[self.PRED_VALID_MASK] = \
                    mb_valid_mask
            prediction_dict[self.PRED_ANGLE_VECTORS] = all_angle_vectors  # Not normed.
            prediction_dict[self.PRED_ANGLES] = all_orientations
            # Curently, all angle_vectors and orientations are correspond to pedestrians at the image center.
            # If you want to edit this, please remember to change the code in `orientnet_panoptic_evaluator.py`
        else:
            raise NotImplementedError('Prediction dict not implemented for',
                                      self._box_rep)

        return prediction_dict

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

    def _read_orient_from_file(self, classes_name, sample_name):
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

        sub_str = 'orient'
        results = {}

        file_name = self._make_file_path(classes_name,
                                         sub_str,
                                         sample_name)
        print('self._read_orient_from_file :: file_name = ', file_name)
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
                    samples = self.dataset.next_batch(batch_size=1, shuffle=True, OrientNet=True)

                else:  # self._train_val_test == "val"
                    # Load samples in order for validation
                    print('Load samples in order for validation')
                    samples = self.dataset.next_batch(batch_size=1,
                                                      shuffle=True,  # Default shuffle=False
                                                      OrientNet=True)

                # Check if thers is any pedestrian detected by maskrcnn
                if not samples:
                    print('No pedestrian detected by MaskRCNN, get the next batch!')
                    continue
                # Only handle one sample at a time for now
                sample = samples[0]
                anchors_info = sample.get(constants.KEY_ANCHORS_INFO)

                # When training, if the mini batch is empty, go to the next
                # sample. Otherwise carry on with the found valid sample.
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
                # print('load_samples():', sample_index)
                # In testing mode, we don't need to read the orientation groundtruth, so OrientNet=False
                samples = self.dataset.load_samples([sample_index], OrientNet=False)
            else:
                # print('testing mode, next_batch()')
                samples = self.dataset.next_batch(batch_size=1, shuffle=False, OrientNet=False)

            # Only handle one sample at a time for now
            sample = samples[0]
            anchors_info = sample.get(constants.KEY_ANCHORS_INFO)

        sample_name = sample.get(constants.KEY_SAMPLE_NAME)
        sample_augs = sample.get(constants.KEY_SAMPLE_AUGS)

        # Network input data
        image_input = sample.get(constants.KEY_IMAGE_INPUT)  # shape =  (1080, 1920, 3)
        image_mask_input = sample.get(constants.KEY_IMAGE_MASK_INPUT)  # shape =  (?, 28, 28)
        image_mrcnn_bbox_input = sample.get(constants.KEY_IMAGE_MRCNN_BBOX_INPUT)  # This b-box is in pixels, shape =  (?, 4)
        # Image shape (h, w)
        image_shape = [image_input.shape[0], image_input.shape[1]]

        has_meaningful_loss = False
        label_boxes_3d = []
        label_boxes_quat = []
        valid_mask = []
        if self._train_val_test in ["train", "val"]:
            # get orientation ground truth from .npy files whose result has the same order as the MaskRCNN detection.
            classes_name = 'Pedestrian'
            MaskRCNN_label_gt = self._read_orient_from_file(classes_name, sample_name)
            # print('MaskRCNN_label_gt = ', MaskRCNN_label_gt)
            label_boxes_3d_gt = MaskRCNN_label_gt.item().get('boxes_3d')
            # print('label_boxes_3d_gt = ', label_boxes_3d_gt)
            # [x1, y1, x2, y2, height, width, length, x,y,z, ry]
            num_maskrcnn_result = image_mask_input.shape[0]

            # Get ground truth data from Panoptic dataset label files.
            # Now we should filter out all the useless ground truth.
            # In label_boxes_3d_gt, the number of rows >= num_maskrcnn_result,
            # but the first num_maskrcnn_result rows are always correspond to the
            # image_mask_input and they are ordered in the same sequence.
            # We should generate groundtruths for all the MaskRCNN results. But not
            # all the MaskRCNN results are valid. We have to build a valid_mask
            # to remember all the valid results so that the invalid results will
            # not be used in Loss calculation.
            invalid_num = 0
            has_meaningful_loss = True
            valid_mask = [True]*num_maskrcnn_result
            for i in range(num_maskrcnn_result):
                if np.isnan(label_boxes_3d_gt[i, 1]):
                    valid_mask[i] = False
                    invalid_num += 1
            if num_maskrcnn_result == invalid_num:
                has_meaningful_loss = False
            # print('image_mask_input.shape = ', image_mask_input.shape)
            # print('image_mrcnn_bbox_input.shape = ', image_mrcnn_bbox_input.shape)
            # print('valid_mask = ', valid_mask)

            label_boxes_2d_norm = label_boxes_3d_gt[0:num_maskrcnn_result, 0:4]
            label_boxes_2d_norm[:, 0] = label_boxes_2d_norm[:, 0]/self._img_pixel_size[1].astype(np.float32)
            label_boxes_2d_norm[:, 1] = label_boxes_2d_norm[:, 1]/self._img_pixel_size[0].astype(np.float32)
            label_boxes_2d_norm[:, 2] = label_boxes_2d_norm[:, 2]/self._img_pixel_size[1].astype(np.float32)
            label_boxes_2d_norm[:, 3] = label_boxes_2d_norm[:, 3]/self._img_pixel_size[0].astype(np.float32)
            # print('label_boxes_2d_norm = ', label_boxes_2d_norm)
            # We only need orientation from box_3d
            # label_boxes_3d = sample.get(constants.KEY_LABEL_BOXES_3D)
            label_boxes_3d = np.zeros(label_boxes_3d_gt[0:num_maskrcnn_result, 4:11].shape)
            label_boxes_3d[:, 0:3] = label_boxes_3d_gt[0:num_maskrcnn_result, 7:10]
            label_boxes_3d[:, 3] = label_boxes_3d_gt[0:num_maskrcnn_result, 6]
            label_boxes_3d[:, 4] = label_boxes_3d_gt[0:num_maskrcnn_result, 5]
            label_boxes_3d[:, 5] = label_boxes_3d_gt[0:num_maskrcnn_result, 4]
            label_boxes_3d[:, 6] = label_boxes_3d_gt[0:num_maskrcnn_result, 10]
            # print('label_boxes_3d = ', label_boxes_3d)
            # box_3d [x, y, z, l, w, h, ry]
            # Transform orientation to quaternion format (w, x, y, z)
            label_boxes_quat = []
            # print('label_boxes_3d = ', label_boxes_3d)
            for j in range(label_boxes_3d.shape[0]):
                # use translation to get apparant orientation of object in the
                # camera view.
                if not np.isnan(label_boxes_3d[j, 1]):
                    pitch_angle = np.arctan2(label_boxes_3d[j][0], label_boxes_3d[j][2]+label_boxes_3d[j][5]/2);
                    roll_angle = -np.arctan2(label_boxes_3d[j][1], label_boxes_3d[j][2]+label_boxes_3d[j][5]/2);
                    # When the camera coordinate is defined as:
                    # x points to the ground; y points down to the floor;
                    # z shoot out from the camera.
                    # Then angle definition should be:
                    # (make sure the definition is the same in all orientation-related files!!)
                    # pitch_angle = np.arctan2(label_boxes_3d[j][0], label_boxes_3d[j][2]);
                    # roll_angle = np.arctan2(label_boxes_3d[j][1]-label_boxes_3d[j][5]/2, label_boxes_3d[j][2]);
                    # print('For people #', j, ': pitch_angle=', pitch_angle, '; roll_angle=', roll_angle, '; yaw_angle=', label_boxes_3d[j][6])
                    rot = euler2mat(roll_angle, pitch_angle, 0, axes='rxyz')
                    R = euler2mat(0, label_boxes_3d[j][6], 0, axes='rxyz')
                    R = np.dot(np.linalg.inv(rot), R)
                    quat = mat2quat(R)  # (w,x,y,z)
                else:
                    quat = [np.nan, np.nan, np.nan, np.nan]
                # print('quat = ', quat)
                if label_boxes_quat == []:
                    label_boxes_quat = np.array([quat])
                else:
                    label_boxes_quat = np.concatenate((label_boxes_quat, [quat]), axis=0)
                    # label_boxes_quat = label_boxes_quat.copy().astype(np.float32)
            # print('label_boxes_quat = ', label_boxes_quat)

        # Fill the placeholders for anchor information
        self._placeholder_inputs[self.PL_IMG_INPUT] = image_input  # image_input.shape =  (1080, 1920, 3)
        self._placeholder_inputs[self.PL_IMG_MRCNN_BBOX_INPUT] = image_mrcnn_bbox_input # This b-box is in pixels
        self._placeholder_inputs[self.PL_IMG_MASK_28X28_INPUT] = image_mask_input

        # Fill in the groundtruth info
        self._placeholder_inputs[self.PL_LABEL_BOXES_3D] = label_boxes_3d
        self._placeholder_inputs[self.PL_LABEL_BOXES_QUATERNION] = label_boxes_quat
        self._placeholder_inputs[self.PL_LABEL_VALID_MASK] = valid_mask

        # Sample Info
        # img_idx is a list to match the placeholder shape
        self._placeholder_inputs[self.PL_IMG_IDX] = [int(sample_name)]
        # Temporary sample info for debugging
        self.sample_info.clear()
        self.sample_info['sample_name'] = sample_name
        self.sample_info['rpn_mini_batch'] = anchors_info

        # Create a feed_dict and fill it with input values
        feed_dict = dict()
        # print('image_mrcnn_bbox_input = ', image_mrcnn_bbox_input)
        if len(image_mrcnn_bbox_input) == 0:
            print('WARNING, No element!!!! !!!!!This is a hack !!!!!')
        # elif self._img_repeat_times == [1]:
        #     print('WARNING, img_repeat_times == [1] !!!!!This is a hack !!!!!')
        else:
            for key, value in self.placeholders.items():
                feed_dict[value] = self._placeholder_inputs[key]

        # print('---------------- pplp/core/models/rpn_panoptic_model.py end------------------')
        return feed_dict, has_meaningful_loss

    def loss(self, prediction_dict):
        # Note: The loss should be using mini-batch values only
        loss_dict = {}
        losses_output = orientnet_loss_builder.build(self, prediction_dict)

        ang_loss_norm = losses_output.get(
            orientnet_loss_builder.KEY_ANG_LOSS_NORM)
        if ang_loss_norm is not None:
            loss_dict.update({self.LOSS_FINAL_ORIENTATION: ang_loss_norm})

        with tf.variable_scope('ang_loss_norm'):
            total_loss = ang_loss_norm

        return loss_dict, total_loss
