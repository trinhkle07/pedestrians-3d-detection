# import cv2
import numpy as np
import os

from PIL import Image

from wavedata.tools.obj_detection import obj_panoptic_utils
from wavedata.tools.obj_detection import evaluation

from pplp.core import box_3d_panoptic_encoder, anchor_panoptic_projector
from pplp.core import anchor_encoder
from pplp.core import anchor_filter

from pplp.core.anchor_generators import grid_anchor_3d_generator


# import for MRCNN
# Add this block for ROS python conflict
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    sys.path.remove('$HOME/segway_kinetic_ws/devel/lib/python2.7/dist-packages')
except ValueError:
    pass

import cv2
# import random
# import math
# import numpy as np
# import skimage.io
# import matplotlib
# import matplotlib.pyplot as plt

from pplp.core.maskrcnn import coco
from pplp.core.maskrcnn import maskrcnn_utils
from pplp.core.models import maskrcnn_model as modellib
from pplp.core.maskrcnn import visualize
# from pplp.core.models.maskrcnn_model import log


class MiniBatchPreprocessor(object):
    def __init__(self,
                 dataset,
                 mini_batch_dir,
                 anchor_strides,
                 density_threshold,
                 neg_iou_3d_range,
                 pos_iou_3d_range):
        """Preprocesses anchors and saves info to files for RPN training

        Args:
            dataset: Dataset object
            mini_batch_dir: directory to save the info
            anchor_strides: anchor strides for generating anchors (per class)
            density_threshold: minimum number of points required to keep an
                anchor
            neg_iou_3d_range: 3D iou range for an anchor to be negative
            pos_iou_3d_range: 3D iou range for an anchor to be positive
        """

        self._dataset = dataset
        self.mini_batch_panoptic_utils = self._dataset.panoptic_utils.mini_batch_panoptic_utils

        self._mini_batch_dir = mini_batch_dir

        self._area_extents = self._dataset.panoptic_utils.area_extents
        self._anchor_strides = anchor_strides

        self._density_threshold = density_threshold
        self._negative_iou_range = neg_iou_3d_range
        self._positive_iou_range = pos_iou_3d_range

    def _calculate_anchors_info(self,
                                all_anchor_boxes_3d,
                                empty_anchor_filter,
                                gt_labels):
        """Calculates the list of anchor information in the format:
            N x 8 [anchor_indices, max_gt_iou(2d or 3d), (6 x offsets), class_index]
                max_gt_out - highest 3D iou with any ground truth box
                offsets - encoded offsets [dx, dy, dz, d_dimx, d_dimy, d_dimz]
                class_index - the anchor's class as an index
                    (e.g. 0 or 1, for "Background" or "Car")

        Args:
            all_anchor_boxes_3d: list of anchors in box_3d format
                N x [x, y, z, l, w, h, ry]
            empty_anchor_filter: boolean mask of which anchors are non empty
            gt_labels: list of Object Label data format containing ground truth
                labels to generate positives/negatives from.

        Returns:
            list of anchor info
        """
        # Check for ground truth objects
        if len(gt_labels) == 0:
            raise Warning("No valid ground truth label to generate anchors.")

        panoptic_utils = self._dataset.panoptic_utils

        # Filter empty anchors
        anchor_indices = np.where(empty_anchor_filter)[0]
        anchor_boxes_3d = all_anchor_boxes_3d[empty_anchor_filter]

        # Convert anchor_boxes_3d to anchor format
        anchors = box_3d_panoptic_encoder.box_3d_to_anchor(anchor_boxes_3d)

        # Convert gt to boxes_3d -> anchors -> iou format
        # print('mini_batch_panoptic_preprocessor.py :')
        # print('gt_labels = ', gt_labels)
        gt_boxes_3d = np.asarray(
            [box_3d_panoptic_encoder.object_label_to_box_3d(gt_obj)
             for gt_obj in gt_labels])
        gt_anchors = box_3d_panoptic_encoder.box_3d_to_anchor(gt_boxes_3d,
                                                     ortho_rotate=True)

        rpn_iou_type = self.mini_batch_panoptic_utils.rpn_iou_type
        if rpn_iou_type == '2d':
            # Convert anchors to 2d iou format
            anchors_for_2d_iou, _ = np.asarray(anchor_panoptic_projector.project_to_bev(
                anchors, panoptic_utils.bev_extents))

            gt_boxes_for_2d_iou, _ = anchor_panoptic_projector.project_to_bev(
                gt_anchors, panoptic_utils.bev_extents)

        elif rpn_iou_type == '3d':
            # Convert anchors to 3d iou format for calculation
            anchors_for_3d_iou = box_3d_panoptic_encoder.box_3d_to_3d_iou_format(
                anchor_boxes_3d)

            gt_boxes_for_3d_iou = \
                box_3d_panoptic_encoder.box_3d_to_3d_iou_format(gt_boxes_3d)
        else:
            raise ValueError('Invalid rpn_iou_type {}', rpn_iou_type)

        # Initialize sample and offset lists
        num_anchors = len(anchor_boxes_3d)
        all_info = np.zeros((num_anchors,
                             self.mini_batch_panoptic_utils.col_length))

        # Update anchor indices
        all_info[:, self.mini_batch_panoptic_utils.col_anchor_indices] = anchor_indices

        # For each of the labels, generate samples
        # print('gt_boxes_for_2d_iou = ', gt_boxes_for_2d_iou)
        # print('anchors_for_2d_iou = ', anchors_for_2d_iou)
        for gt_idx in range(len(gt_labels)):

            gt_obj = gt_labels[gt_idx]
            gt_box_3d = gt_boxes_3d[gt_idx]

            # Get 2D or 3D IoU for every anchor
            # Default setting 2D
            if self.mini_batch_panoptic_utils.rpn_iou_type == '2d':
                gt_box_for_2d_iou = gt_boxes_for_2d_iou[gt_idx]
                ious = evaluation.two_d_iou(gt_box_for_2d_iou,
                                            anchors_for_2d_iou)
            elif self.mini_batch_panoptic_utils.rpn_iou_type == '3d':
                gt_box_for_3d_iou = gt_boxes_for_3d_iou[gt_idx]
                ious = evaluation.three_d_iou(gt_box_for_3d_iou,
                                              anchors_for_3d_iou)

            # Only update indices with a higher iou than before
            update_indices = np.greater(
                ious, all_info[:, self.mini_batch_panoptic_utils.col_ious])

            # Get ious to update
            ious_to_update = ious[update_indices]

            # Calculate offsets, use 3D iou to get highest iou
            anchors_to_update = anchors[update_indices]
            gt_anchor = box_3d_panoptic_encoder.box_3d_to_anchor(gt_box_3d,
                                                        ortho_rotate=True)
            offsets = anchor_encoder.anchor_to_offset(anchors_to_update,
                                                      gt_anchor)

            # Convert gt type to index
            class_idx = panoptic_utils.class_str_to_index(gt_obj.type)

            # Update anchors info (indices already updated)
            # [index, iou, (offsets), class_index]
            all_info[update_indices,
                     self.mini_batch_panoptic_utils.col_ious] = ious_to_update

            all_info[update_indices,
                     self.mini_batch_panoptic_utils.col_offsets_lo:
                     self.mini_batch_panoptic_utils.col_offsets_hi] = offsets
            all_info[update_indices,
                     self.mini_batch_panoptic_utils.col_class_idx] = class_idx

        return all_info

    def preprocess(self, indices):
        """Preprocesses anchor info and saves info to files

        Args:
            indices (int array): sample indices to process.
                If None, processes all samples
        """
        # Get anchor stride for class
        anchor_strides = self._anchor_strides

        dataset = self._dataset
        dataset_utils = self._dataset.panoptic_utils
        classes_name = dataset.classes_name

        # Make folder if it doesn't exist yet
        output_dir = self.mini_batch_panoptic_utils.get_file_path(classes_name,
                                                         anchor_strides,
                                                         sample_name=None)
        os.makedirs(output_dir, exist_ok=True)

        # Get clusters for class
        all_clusters_sizes, _ = dataset.get_cluster_info()

        anchor_generator = grid_anchor_3d_generator.GridAnchor3dGenerator()

        # Load indices of data_split
        all_samples = dataset.sample_list

        if indices is None:
            indices = np.arange(len(all_samples))
        num_samples = len(indices)
        print('indices = ', indices)

        # For each image in the dataset, save info on the anchors
        for sample_idx in indices:
            # Get image name for given cluster
            sample_name = all_samples[sample_idx].name
            img_idx = int(sample_name)

            # Check for existing files and skip to the next
            if self._check_for_existing(classes_name, anchor_strides,
                                        sample_name):
                print("{} / {}: Sample already preprocessed".format(
                    sample_idx + 1, num_samples, sample_name))
                continue

            # Get ground truth and filter based on difficulty
            ground_truth_list = obj_panoptic_utils.read_labels(dataset.label_dir,
                                                      img_idx)

            # Filter objects to dataset classes
            # print('mini_batch_panoptic_preprocessor.py : ')
            # print('ground_truth_list = ', ground_truth_list)
            # If no valid ground truth, skip this image
            if not ground_truth_list:
                print("{} / {} No {}s for sample {} "
                      "(Ground Truth Filter)".format(
                          sample_idx + 1, num_samples,
                          classes_name, sample_name))

                # Output an empty file and move on to the next image.
                self._save_to_file(classes_name, anchor_strides, sample_name)
                continue

            filtered_gt_list = dataset_utils.filter_labels(ground_truth_list)
            filtered_gt_list = np.asarray(filtered_gt_list)

            # Filtering by class has no valid ground truth, skip this image
            if len(filtered_gt_list) == 0:
                print("{} / {} No {}s for sample {} "
                      "(Ground Truth Filter)".format(
                          sample_idx + 1, num_samples,
                          classes_name, sample_name))

                # Output an empty file and move on to the next image.
                self._save_to_file(classes_name, anchor_strides, sample_name)
                continue

            # Get ground plane
            ground_plane = obj_panoptic_utils.get_road_plane(img_idx,
                                                    dataset.planes_dir)

            image = Image.open(dataset.get_rgb_image_path(sample_name))
            image_shape = [image.size[1], image.size[0]]

            # Generate sliced 2D voxel grid for filtering
            # print('******** Generate sliced 2D voxel grid for filtering *********')
            print('sample_name = ', sample_name)
            # print('dataset.bev_source = ', dataset.bev_source)
            # print('image_shape = ', image_shape)

            # If run with density filter:
            # vx_grid_2d = dataset_utils.create_sliced_voxel_grid_2d(
            #     sample_name,
            #     source=dataset.bev_source,
            #     image_shape=image_shape)

            # If run with occupancy filter:
            point_cloud = dataset_utils.get_point_cloud('lidar', img_idx, image_shape)
            bev_maps = dataset_utils.create_bev_maps(point_cloud, ground_plane)

            # List for merging all anchors
            all_anchor_boxes_3d = []

            # Create anchors for each class
            # print('mini_batch_panoptic_preprocessor.py : Create anchors for each class')
            # print('mini_batch_panoptic_preprocessor.py : dataset.classes = ', dataset.classes)
            for class_idx in range(len(dataset.classes)):
                # Generate anchors for all classes
                # print('class_idx = ', class_idx)
                # print('len(dataset.classes) = ', len(dataset.classes))
                # print('self._area_extents = ', self._area_extents)
                # print('all_clusters_sizes = ', all_clusters_sizes)
                # print('self._anchor_strides = ', self._anchor_strides)
                # print('ground_plane = ', ground_plane)
                grid_anchor_boxes_3d = anchor_generator.generate(
                    area_3d=self._area_extents,
                    anchor_3d_sizes=all_clusters_sizes[class_idx],
                    anchor_stride=self._anchor_strides[class_idx],
                    ground_plane=ground_plane)

                all_anchor_boxes_3d.extend(grid_anchor_boxes_3d)

            # Filter empty anchors
            all_anchor_boxes_3d = np.asarray(all_anchor_boxes_3d)
            # For panoptic dataset, we have 14*16*2(degree 0 and pi) = 448 anchor_boxes_3d here!
            np.set_printoptions(threshold=np.nan)
            # print('all_anchor_boxes_3d = ', all_anchor_boxes_3d)
            anchors = box_3d_panoptic_encoder.box_3d_to_anchor(all_anchor_boxes_3d)

            # Use density filter as anchor filters:
            # Here, we created 448 boolean maska for all 488 anchors,
            # but only those whose density>1 is counted.
            # Usually the final Numberis between 20~40.

            # empty_anchor_filter = anchor_filter.get_empty_anchor_filter_2d(
            #     anchors, vx_grid_2d, self._density_threshold)

            # Use occupancy filter as anchor filters:
            # print('bev_maps[\'occupancy_maps\'] = ', bev_maps['occupancy_maps'])

            empty_anchor_filter = anchor_filter.get_empty_anchor_filter_occupancy(
                anchors, bev_maps['occupancy_maps'], self._area_extents)

            # Calculate anchor info
            anchors_info = self._calculate_anchors_info(
                all_anchor_boxes_3d, empty_anchor_filter, filtered_gt_list)
            # anchors_info : N x 9 [anchor_indices, max_gt_iou(2d or 3d), (6 x offsets), class_index]
            #     max_gt_out - highest 3D iou with any ground truth box
            #     offsets - encoded offsets [dx, dy, dz, d_dimx, d_dimy, d_dimz]
            #     class_index - the anchor's class as an index
            #         (e.g. 0 or 1, for "Background" or "Pedestrian")
            # for example, anchors_info[23] = [ 3.69000000e+02  1.23000000e-01 -6.01920353e-01  5.23366792e-03
            #                 5.50028119e-01 -1.05079512e-01  1.73129711e-02  5.26857404e-01
            #                 1.00000000e+00]

            # print('empty_anchor_filter = ', empty_anchor_filter)

            # In our dataset, anchor_ious is the second row of anchors_info
            anchor_ious = anchors_info[:, self.mini_batch_panoptic_utils.col_ious]
            # print('anchors_info[:, 0] = ', anchors_info[:, 0])
            # print('anchor_ious = ', anchor_ious)
            # There will always be corner cases. For example, for image 500100003527.jpg
            # There are 2 pedestrians on the image, but the Panoptic dataset only
            # provide groundtruth skeletons for an occluded person. In this case,
            # even though we have 1 groundtruth label, the anchor_ious still equals
            # to all zeros! So we better save an empty file for this image.
            if np.max(anchor_ious) == 0:
                print("{} / {} . Anchor_ious are all zeros for sample {} "
                      "(Save an empty file for this minibatch)".format(
                          sample_idx + 1, num_samples, sample_name))

                # Output an empty file and move on to the next image.
                self._save_to_file(classes_name, anchor_strides, sample_name)
                continue

            valid_iou_indices = np.where(anchor_ious > 0.0)[0]

            print("{} / {}:"
                  "{:>6} anchors, "
                  "{:>6} iou > 0.0, "
                  "for {:>3} {}(s) for sample {}".format(
                      sample_idx + 1, num_samples,
                      len(anchors_info),
                      len(valid_iou_indices),
                      len(filtered_gt_list), classes_name, sample_name
                  ))

            # Save anchors info
            # under "anchors_info", we only saved those anchors that has density>1
            self._save_to_file(classes_name, anchor_strides,
                               sample_name, anchors_info)

    def preprocess_mrcnn(self, indices):
        """Preprocesses MRCNN result to files

        Args:
            indices (int array): sample indices to process.
                If None, processes all samples
        """
        # Get anchor stride for class
        dataset = self._dataset
        dataset_utils = self._dataset.panoptic_utils
        classes_name = dataset.classes_name

        # Make folder if it doesn't exist yet
        sub_str = 'mrcnn'
        output_dir = self.mini_batch_panoptic_utils.make_file_path(
            classes_name,
            sub_str,
            sample_name=None)
        os.makedirs(output_dir, exist_ok=True)

        # Get clusters for class
        all_clusters_sizes, _ = dataset.get_cluster_info()

        # Load indices of data_split
        all_samples = dataset.sample_list

        if indices is None:
            indices = np.arange(len(all_samples))
        num_samples = len(indices)

        # Initialize MskRCNN Model:
        # Root directory of the project
        ROOT_DIR = os.getcwd()

        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "mylogs")

        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco_humanpose.h5")
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            maskrcnn_utils.download_trained_weights(COCO_MODEL_PATH)

        class InferenceConfig(coco.CocoConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            KEYPOINT_MASK_POOL_SIZE = 7

        inference_config = InferenceConfig()

        # Recreate the model in inference mode
        model = modellib.MaskRCNN(mode="inference",
                                  config=inference_config,
                                  model_dir=MODEL_DIR)

        # Get path to saved weights
        # Either set a specific path or find last trained weights
        # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
        # model_path = model.find_last()[1]
        model_path = os.path.join(ROOT_DIR, "mask_rcnn_coco_humanpose.h5")
        # Load trained weights (fill in path to trained weights here)
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)

        # For each image in the dataset, save info on the anchors
        for sample_idx in indices:
            print('########## loop starts ###################')
            # Get image name for given cluster
            sample_name = all_samples[sample_idx].name
            img_idx = int(sample_name)
            print('img_idx = ', img_idx)

            # Check for existing files and skip to the next
            if self._check_for_mrcnn_existing(classes_name, sub_str, sample_name):
                print("{} / {}: Sample already preprocessed".format(
                    sample_idx + 1, num_samples, sample_name))
                continue

            # Get ground truth and filter based on difficulty
            ground_truth_list = obj_panoptic_utils.read_labels(dataset.label_dir,
                                                               img_idx)
            # print('ground_truth_list = ', ground_truth_list)
            # If no valid ground truth, skip this image
            if not ground_truth_list:
                print("{} / {} No {}s for sample {} "
                      "(Ground Truth Filter)".format(
                          sample_idx + 1, num_samples,
                          classes_name, sample_name))

                # Output an empty file and move on to the next image.
                self._save_mrcnn_to_file(classes_name, sub_str, sample_name)
                continue

            # Filter objects to dataset classes
            filtered_gt_list = dataset_utils.filter_labels(ground_truth_list)
            filtered_gt_list = np.asarray(filtered_gt_list)
            # print('filtered_gt_list = ', filtered_gt_list)

            # Filtering by class has no valid ground truth, skip this image
            if len(filtered_gt_list) == 0:
                print("{} / {} No {}s for sample {} "
                      "(Ground Truth Filter)".format(
                          sample_idx + 1, num_samples,
                          classes_name, sample_name))

                # Output an empty file and move on to the next image.
                self._save_mrcnn_to_file(classes_name, sub_str, sample_name)
                continue

            # Get RGB image
            image = cv2.imread(dataset.get_rgb_image_path(sample_name))
            print('Reading image: ', dataset.get_rgb_image_path(sample_name))
            # BGR->RGB
            image = image[:, :, ::-1]
            # Run detection
            mrcnn_results = model.detect_keypoint_and_feature_map([image], verbose=0)
            # print('mrcnn_results = ', mrcnn_results)
            if len(mrcnn_results) == 0:
                print('No people detected on the image!')
                # Output an empty file and move on to the next image.
                self._save_mrcnn_to_file(classes_name, sub_str, sample_name)
            else:
                print('There are ', len(mrcnn_results['rois']), ' people on the image.')

                # Save Image MaskRCNN info
                self._save_mrcnn_to_file(classes_name, sub_str,
                                         sample_name, mrcnn_results)

                self._visualize_mrcnn_to_file(image, classes_name, sub_str, sample_name,
                                              mrcnn_results)

    def load_mrcnn(self, img_idx):
        """Load MRCNN result from .npy files

        Args:
            sample_idx (int): sample index to process.
        """

        # Read every info of current sample_idx
        # print('########## load_mrcnn starts ###################')
        # Get image name for given cluster
        # print('img_idx = ', img_idx)
        sample_name = str(img_idx)

        # Get anchor stride for class
        dataset = self._dataset
        classes_name = dataset.classes_name

        # Make folder if it doesn't exist yet
        sub_str = 'mrcnn'

        # print('classes_name = ', classes_name)
        # print('sub_str = ', sub_str)
        # print('sample_name = ', sample_name)
        # Check for non-existing files and ERROR if cannot find.
        if not self._check_for_mrcnn_existing(classes_name, sub_str,
                                              sample_name):
            raise Exception("Sample {} Not Found!".format(sample_name))

        # Output an empty file and move on to the next image.
        mrcnn_result = self._read_mrcnn_from_file(classes_name, sample_name)
        # print('########## load_mrcnn starts end ###################')
        return mrcnn_result

    def load_orientnet(self, img_idx):
        """Load OrientNet result from .npy files

        Args:
            sample_idx (int): sample index to process.
        """

        # Read every info of current sample_idx
        # print('########## load_orientnet starts ###################')
        # Get image name for given cluster
        # print('img_idx = ', img_idx)
        sample_name = str(img_idx)

        # Get anchor stride for class
        dataset = self._dataset
        classes_name = dataset.classes_name

        # Make folder if it doesn't exist yet
        sub_str = 'orient'

        # print('classes_name = ', classes_name)
        # print('sub_str = ', sub_str)
        # print('sample_name = ', sample_name)
        # Check for non-existing files and ERROR if cannot find.
        if not self._check_for_mrcnn_existing(classes_name, sub_str,
                                              sample_name):
            raise Exception("Sample {} Not Found!".format(sample_name))

        # Output an empty file and move on to the next image.
        orientnet_result = self._read_orient_from_file(classes_name, sample_name)
        # print('########## load_mrcnn starts end ###################')
        return orientnet_result

    def _check_for_existing(self, classes_name, anchor_strides, sample_name):
        """
        Checks if a mini batch file exists already

        Args:
            classes_name (str): classes name, e.g. 'Car', 'Pedestrian',
                'Cyclist', 'People'
            anchor_strides: anchor strides
            sample_name (str): sample name from dataset, e.g. '000123'

        Returns:
            True if the anchors info file already exists
        """

        file_name = self.mini_batch_panoptic_utils.get_file_path(
            classes_name,
            anchor_strides,
            sample_name)
        if os.path.exists(file_name):
            return True

        return False

    def _check_for_mrcnn_existing(self, classes_name, sub_str, sample_name, subsub_str=None):
        """
        Checks if a mini batch file exists already

        Args:
            classes_name (str): classes name, e.g. 'Car', 'Pedestrian',
                'Cyclist', 'People'
            sub_str: a name for folder subname
            sample_name (str): sample name from dataset, e.g. '000123'

        Returns:
            True if the anchors info file already exists
        """

        file_name = self.mini_batch_panoptic_utils.make_file_path(classes_name,
                                                                  sub_str,
                                                                  sample_name,
                                                                  subsub_str= subsub_str)
        if os.path.exists(file_name):
            return True

        print('Cannot find: ', file_name)
        return False

    def _save_to_file(self, classes_name, anchor_strides, sample_name,
                      anchors_info=np.array([])):
        """
        Saves the anchors info matrix to a file

        Args:
            classes_name (str): classes name, e.g. 'Car', 'Pedestrian',
                'Cyclist', 'People'
            anchor_strides: anchor strides
            sample_name (str): name of sample, e.g. '000123'
            anchors_info: ndarray of anchor info of shape (N, 8)
                N x [index, iou, (6 x offsets), class_index], defaults to
                an empty array
        """

        file_name = self.mini_batch_panoptic_utils.get_file_path(classes_name,
                                                                 anchor_strides,
                                                                 sample_name)

        # Save to npy file
        anchors_info = np.asarray(anchors_info, dtype=np.float32)
        np.save(file_name, anchors_info)

    def _save_mrcnn_to_file(self, classes_name, sub_str, sample_name,
                            mrcnn_results=None):
        """
        Saves the MRCNN info matrix to a file

        Args:
            classes_name (str): classes name, e.g. 'Car', 'Pedestrian',
                'Cyclist', 'People'
            anchor_strides: anchor strides
            sample_name (str): name of sample, e.g. '000123'
            mrcnn_results: To Do
        """
        if mrcnn_results:
            # print('mrcnn_results = ', mrcnn_results)
            # Save msakrcnn_result
            file_name = self.mini_batch_panoptic_utils.make_file_path(classes_name,
                                                                      sub_str,
                                                                      sample_name)
            print('_save_mrcnn_to_file :: file_name = ', file_name)
            np.save(file_name, mrcnn_results)

# np.hstack(array1,array2) vstack concatenate
# result['rois'] = array
        else:
            results = {}
            file_name = self.mini_batch_panoptic_utils.make_file_path(classes_name,
                                                                      sub_str,
                                                                      sample_name)
            print('_save_mrcnn_to_file : results empty : file_name = ', file_name)
            # Save to npy file
            np.save(file_name, results)

    def _visualize_mrcnn_to_file(self, image, classes_name, sub_str, sample_name,
                                 mrcnn_results):
        """
        Saves the MRCNN info matrix to a file

        Args:
            classes_name (str): classes name, e.g. 'Car', 'Pedestrian',
                'Cyclist', 'People'
            anchor_strides: anchor strides
            sample_name (str): name of sample, e.g. '000123'
            mrcnn_results: To Do
        """

        class InferenceConfig(coco.CocoConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            KEYPOINT_MASK_POOL_SIZE = 7

        inference_config = InferenceConfig()

        mrcnn_class_names = ['BG', 'person']

        file_name = self.mini_batch_panoptic_utils.make_image_path(classes_name,
                                                                   sub_str,
                                                                   sample_name)

        print('file_name = ', file_name)
        # visualize.save_keypoints(image, mrcnn_results['rois'], mrcnn_results['keypoints'], mrcnn_results['class_ids'], mrcnn_class_names, filename=file_name, skeleton=inference_config.LIMBS, save_picture=True)
        # visualize.display_keypoints(image, mrcnn_results['rois'],mrcnn_results['keypoints'],mrcnn_results['class_ids'], mrcnn_class_names,skeleton = inference_config.LIMBS)
        visualize.save_image_keypoint_and_mask(image, mrcnn_results['rois'], mrcnn_results['keypoints'], mrcnn_results['full_masks'], mrcnn_results['class_ids'], mrcnn_class_names,
                                               filename=file_name, skeleton=inference_config.LIMBS, save_picture=True)

    def _read_mrcnn_from_file(self, classes_name, sample_name):
        """
        Reads the MRCNN info matrix from a file

        Args:
            classes_name (str): classes name, e.g. 'Car', 'Pedestrian',
                'Cyclist', 'People'
            sample_name (str): name of sample, e.g. '000123'

        Returns:
            mrcnn_results: {'scores': array(dtype=float32),
                            'features': array(dtype=float32),
                            'keypoints': array(),
                            'class_ids': array(dtype=int32),
                            'masks': array(dtype=float32),
                            'rois': array(dtype=int32),
                            'full_masks': array(dtype=uint8)}
        """

        # print('=============== _read_mrcnn_from_file =================')
        sub_str = 'mrcnn'
        results = {}

        file_name = self.mini_batch_panoptic_utils.make_file_path(classes_name,
                                                                  sub_str,
                                                                  sample_name)
        # print('_read_mrcnn_from_file :: file_name = ', file_name)
        # Load from npy file
        results = np.load(file_name)
        # print('=============== _read_mrcnn_from_file end =================')
        return results

    def _read_orient_from_file(self, classes_name, sample_name):
        """
        Reads the OrientNet info from a file

        Args:
            classes_name (str): classes name, e.g. 'Car', 'Pedestrian',
                'Cyclist', 'People'
            sample_name (str): name of sample, e.g. '000123'

        Returns:
            mrcnn_results: {'boxes_3d': array(dtype=float32)}
        """

        sub_str = 'orient'
        results = {}

        file_name = self.mini_batch_panoptic_utils.make_file_path(classes_name,
                                                                  sub_str,
                                                                  sample_name)
        print('_read_orient_from_file :: file_name = ', file_name)
        # Load from npy file
        results = np.load(file_name)
        # print('=============== _read_mrcnn_from_file end =================')
        return results
