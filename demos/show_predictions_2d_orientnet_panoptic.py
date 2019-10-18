import os
import time

# Add this block for ROS python conflict
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    sys.path.remove('$HOME/segway_kinetic_ws/devel/lib/python2.7/dist-packages')
except ValueError:
    pass
import cv2

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects

from wavedata.tools.core import calib_panoptic_utils
from wavedata.tools.obj_detection import obj_panoptic_utils
from wavedata.tools.obj_detection import evaluation
from wavedata.tools.visualization import vis_panoptic_utils

import pplp
from pplp.builders.dataset_panoptic_builder import DatasetBuilder
from pplp.core import box_3d_panoptic_encoder
from pplp.core import box_3d_panoptic_projector
from pplp.core import anchor_panoptic_projector

from google.protobuf import text_format

BOX_COLOUR_SCHEME = {
    'Car': '#00FF00',           # Green
    'Pedestrian': '#00FFFF',    # Teal
    'Cyclist': '#FFFF00'        # Yellow
}


def main():
    """This demo shows OrientNet predictions on 2D image space.
    Given certain thresholds for predictions, it selects and draws the
    groundtruth bounding boxes on the image sample. It goes through the entire
    prediction samples for the given dataset split.

    """
    dataset_config = DatasetBuilder.copy_config(DatasetBuilder.PANOPTIC_VAL)

    ##############################
    # Options
    ##############################
    dataset_config = DatasetBuilder.merge_defaults(dataset_config)
    dataset_config.data_split = 'val'

    # fig_size = (10, 6.1)
    fig_size = (12, 23)  # The size of final picture as a whole.

    rpn_score_threshold = 0.00
    orientnet_score_threshold = 0.30

    gt_classes = ['Pedestrian']

    # Overwrite this to select a specific checkpoint
    print('!!!Please make sure your settings are all correct!!!!!')
    global_step = 298261  # None
    checkpoint_name = 'orientation_pedestrian_panoptic'

    # Drawing Toggles
    draw_proposals_separate = False
    draw_overlaid = False  # To draw both proposal and predcition bounding boxes
    draw_predictions_separate = True

    # Show orientation for both GT and proposals/predictions
    draw_orientations_on_prop = False  # Set it to false would be OK, since all the orietations of proposals are poiting to the 0 angle.
    draw_orientations_on_pred = True

    # Draw 2D bounding boxes
    draw_projected_2d_boxes = True

    # Draw BEV bounding boxes
    draw_bev_map = False

    # Draw pointclouds
    draw_point_cloud = True
    point_cloud_source = 'lidar'
    slices_config = \
        """
        slices {
            height_lo: -5 # -0.2
            height_hi: 2 # 2.3
            num_slices: 1 # 5
        }
        """

    print('slices_config = ', slices_config)

    text_format.Merge(slices_config,
                      dataset_config.panoptic_utils_config.bev_generator)

    # Save images for samples with no detections
    save_empty_images = False

    draw_proposals_bev = True
    draw_proposals_2d_box = False
    draw_proposals_3d_box = True
    draw_proposals_score = True
    draw_proposals_iou = True

    draw_prediction_score = True
    draw_prediction_iou = True
    ##############################
    # End of Options
    ##############################

    # Get the dataset
    dataset = DatasetBuilder.build_panoptic_dataset(dataset_config,
                                                    use_defaults=False)

    # Setup Paths
    predictions_dir = pplp.root_dir() + \
        '/data/outputs/' + checkpoint_name + '/predictions'

    proposals_and_scores_dir = predictions_dir + \
        '/proposals_and_scores/' + dataset.data_split

    predictions_and_scores_dir = predictions_dir + \
        '/final_predictions_and_scores/' + dataset.data_split

    # Output images directories
    output_dir_base = predictions_dir + '/images_2d'

    # Get checkpoint step
    steps = os.listdir(proposals_and_scores_dir)
    steps.sort(key=int)
    print('Available steps: {}'.format(steps))

    # Use latest checkpoint if no index provided
    if global_step is None:
        global_step = steps[-1]

    if draw_proposals_separate:
        prop_out_dir = output_dir_base + '/proposals/{}/{}/{}'.format(
            dataset.data_split, global_step, rpn_score_threshold)

        if not os.path.exists(prop_out_dir):
            os.makedirs(prop_out_dir)

        print('Proposal images saved to:', prop_out_dir)

    if draw_overlaid:
        overlaid_out_dir = output_dir_base + '/overlaid/{}/{}/{}'.format(
            dataset.data_split, global_step, orientnet_score_threshold)

        if not os.path.exists(overlaid_out_dir):
            os.makedirs(overlaid_out_dir)

        print('Overlaid images saved to:', overlaid_out_dir)

    if draw_predictions_separate:
        pred_out_dir = output_dir_base + '/predictions/{}/{}/{}'.format(
            dataset.data_split, global_step,
            orientnet_score_threshold)

        if not os.path.exists(pred_out_dir):
            os.makedirs(pred_out_dir)

        print('Prediction images saved to:', pred_out_dir)

    # Rolling average array of times for time estimation
    avg_time_arr_length = 10
    last_times = np.repeat(time.time(), avg_time_arr_length) + \
        np.arange(avg_time_arr_length)

    # for sample_idx in [100]:
    #     print('Hack the number!!!!!')
    for sample_idx in range(dataset.num_samples):
        print('\nStart sample #', sample_idx+1)
        # Estimate time remaining with 5 slowest times
        start_time = time.time()
        last_times = np.roll(last_times, -1)
        last_times[-1] = start_time
        avg_time = np.mean(np.sort(np.diff(last_times))[-5:])
        samples_remaining = dataset.num_samples - sample_idx
        est_time_left = avg_time * samples_remaining

        # Print progress and time remaining estimate
        sys.stdout.write('\rSaving {} / {}, Avg Time: {:.3f}s, '
                         'Time Remaining: {:.2f}s \n'. format(
                             sample_idx + 1,
                             dataset.num_samples,
                             avg_time,
                             est_time_left))
        sys.stdout.flush()

        sample_name = dataset.sample_names[sample_idx]
        img_idx = int(sample_name)

        ##############################
        # Proposals
        ##############################
        if draw_proposals_separate or draw_overlaid:
            # Load proposals from files
            proposals_file_path = proposals_and_scores_dir + \
                "/{}/{}.txt".format(global_step, sample_name)
            print('proposals_file_path = ', proposals_file_path)
            if not os.path.exists(proposals_file_path):
                print(proposals_file_path, 'does not exist!')
                print('Sample {}: No proposals, skipping'.format(sample_name))
                continue
            print('Sample {}: Drawing proposals'.format(sample_name))

            proposals_and_scores = np.loadtxt(proposals_file_path)

            # change 1D array in to 2D array even if it has only one row.
            if len(proposals_and_scores.shape) == 1:
                proposals_and_scores.shape = (1, -1)

            # proposals_and_scores, 1~7th colunms are the boxes_3d,
            # the 8th colunm is the score.
            proposal_boxes_3d = proposals_and_scores[:, 0:7]
            proposal_scores = proposals_and_scores[:, 7]

            # Apply score mask to proposals
            print('rpn_score_threshold = ', rpn_score_threshold)
            score_mask = proposal_scores >= rpn_score_threshold
            proposal_boxes_3d = proposal_boxes_3d[score_mask]
            proposal_scores = proposal_scores[score_mask]
            print('There are ', len(proposal_scores), 'proposals left. ')

            proposal_objs = \
                [box_3d_panoptic_encoder.box_3d_to_object_label(proposal,
                                                                obj_type='Proposal')
                 for proposal in proposal_boxes_3d]

        ##############################
        # Predictions
        ##############################
        if draw_predictions_separate or draw_overlaid:
            predictions_file_path = predictions_and_scores_dir + \
                "/{}/{}.txt".format(global_step,
                                    sample_name)
            if not os.path.exists(predictions_file_path):
                print('predictions_file_path NOT EXIST: ', predictions_file_path)
                continue

            # Load predictions from files
            predictions_and_scores = np.loadtxt(
                predictions_and_scores_dir +
                "/{}/{}.txt".format(global_step,
                                    sample_name))

            # change 1D array in to 2D array even if it has only one row.
            if len(predictions_and_scores.shape) == 1:
                predictions_and_scores.shape = (1, -1)

            # print('predictions_and_scores = ', predictions_and_scores)
            prediction_boxes_3d = predictions_and_scores[:, 0:7]
            prediction_scores = predictions_and_scores[:, 7]
            # print('prediction_scores = ', prediction_scores)
            prediction_class_indices = predictions_and_scores[:, 8]

            # process predictions only if we have any predictions left after
            # masking
            if len(prediction_boxes_3d) > 0:

                # Apply score mask
                avod_score_mask = prediction_scores >= orientnet_score_threshold
                prediction_boxes_3d = prediction_boxes_3d[avod_score_mask]
                print('orientnet_score_threshold = ', orientnet_score_threshold)
                print('There are ', len(prediction_boxes_3d), ' predictions left.')
                prediction_scores = prediction_scores[avod_score_mask]
                prediction_class_indices = \
                    prediction_class_indices[avod_score_mask]

                # # Swap l, w for predictions where w > l
                # swapped_indices = \
                #     prediction_boxes_3d[:, 4] > prediction_boxes_3d[:, 3]
                # prediction_boxes_3d = np.copy(prediction_boxes_3d)
                # prediction_boxes_3d[swapped_indices, 3] = \
                #     prediction_boxes_3d[swapped_indices, 4]
                # prediction_boxes_3d[swapped_indices, 4] = \
                #     prediction_boxes_3d[swapped_indices, 3]

        ##############################
        # Ground Truth
        ##############################

        # Get ground truth labels
        if dataset.has_labels:
            print('dataset.label_dir = ', dataset.label_dir)
            print('img_idx = ', img_idx)
            gt_objects = obj_panoptic_utils.read_labels(dataset.label_dir, img_idx)
            # for obj in gt_objects:
            #     print('obj.x1 = ', obj.x1)
        else:
            gt_objects = []

        # Filter objects to desired difficulty
        filtered_gt_objs = dataset.panoptic_utils.filter_labels(
            gt_objects, classes=gt_classes)

        # if sample_idx == 100:
        #     for obj in filtered_gt_objs:
        #         if obj.t[0]>1:
        #             # print('obj.x1 = ', obj.x1)
        #             # print('obj.y1 = ', obj.y1)
        #             # print('obj.x2 = ', obj.x2)
        #             # print('obj.y2 = ', obj.y2)
        #             print('obj.t = ', obj.t)
        #             print('obj.w = ', obj.w)
        #             print('obj.h = ', obj.h)
        #             print('obj.l = ', obj.l)
        #     # print('filtered_gt_objs.x1 = ', filtered_gt_objs.x1)
        #     # print('filtered_gt_objs.x2 = ', filtered_gt_objs.x2)
        #     # print('filtered_gt_objs.y1 = ', filtered_gt_objs.y1)
        #     # print('filtered_gt_objs.y2 = ', filtered_gt_objs.y2)
        boxes2d, _, _ = obj_panoptic_utils.build_bbs_from_objects(
            filtered_gt_objs, class_needed=gt_classes)

        image_path = dataset.get_rgb_image_path(sample_name)
        image = Image.open(image_path)
        image_size = image.size

        # Read the stereo calibration matrix for visualization
        stereo_calib = calib_panoptic_utils.read_calibration(dataset.calib_dir,
                                                    img_idx)
        calib_p2 = stereo_calib.HD_11
        distortion = stereo_calib.Kd_11
        ##############################
        # Reformat and prepare to draw
        ##############################
        # To get the BEV occupancy map, we need to find the ground plane first.
        panoptic_utils = dataset.panoptic_utils
        ground_plane = panoptic_utils.get_ground_plane(sample_name)
        image_shape = [image.size[1], image.size[0]]
        point_cloud = panoptic_utils.get_point_cloud('lidar', img_idx, image_shape)
        bev_maps = panoptic_utils.create_bev_maps(point_cloud, ground_plane)
        bev_img = np.array(bev_maps['occupancy_maps'], dtype=np.int)  # Remember, the original occupancy grid format is int.
        bev_img = np.resize(bev_img, (bev_img.shape[1], bev_img.shape[2]))  # [height, width]
        if not draw_bev_map:
            bev_img = np.zeros((bev_img.shape[1], bev_img.shape[2]), dtype=np.float)
        if draw_proposals_separate or draw_overlaid:
            proposals_as_anchors = box_3d_panoptic_encoder.box_3d_to_anchor(
                proposal_boxes_3d)

            proposal_boxes, _ = anchor_panoptic_projector.project_to_image_space(
                    proposals_as_anchors, calib_p2, image_size, distortion=distortion)

            num_of_proposals = proposal_boxes_3d.shape[0]

            prop_fig, prop_bev_axes, prop_2d_axes, prop_3d_axes = \
                vis_panoptic_utils.visualization(dataset.rgb_image_dir,
                                        img_idx,
                                        bev_img,
                                        display=False,
                                        fig_size=fig_size)

            draw_proposals(filtered_gt_objs,
                           calib_p2,
                           num_of_proposals,
                           proposal_objs,
                           proposal_scores,
                           proposal_boxes,
                           prop_2d_axes,
                           prop_3d_axes,
                           prop_bev_axes,
                           panoptic_utils.area_extents,
                           bev_img.shape,
                           draw_proposals_bev,
                           draw_proposals_2d_box,
                           draw_proposals_3d_box,
                           draw_proposals_score,
                           draw_proposals_iou,
                           draw_orientations_on_prop,
                           distortion=distortion)
            if draw_point_cloud:
                # First,get pointclouds. Now pointclouds are in camera coordinates.
                panoptic_utils = dataset.panoptic_utils
                image_shape = [image_size[1], image_size[0]]
                point_cloud = panoptic_utils.get_point_cloud(
                    point_cloud_source, img_idx, image_shape)
                # print('point_cloud =', point_cloud)
                # Now point_cloud is a 4XN array, in Lidar frame, but only
                # includes those points that can be seen on the image

                # Filter the useful pointclouds from all points
                # In order to do that, we need to find the ground plane first.
                ground_plane = panoptic_utils.get_ground_plane(sample_name)
                filtered_points = panoptic_utils.filter_bev_points(point_cloud, ground_plane)
                # if len(filtered_points) > 0:
                #     print('point_cloud =', point_cloud)
                #     print('filtered_points =', filtered_points)

                # Now, filtered_points is transposed, so filtered_points should
                # be Nx4

                # Project the filtered pointclouds on 2D image. Now filtered
                # pointclouds are already in camera coordinates.
                point_2d = obj_panoptic_utils.project_points_on_2D_image(img_idx,
                                                                         dataset.calib_dir,
                                                                         image_size,
                                                                         filtered_points)
                draw_points(prop_2d_axes, point_2d, 'red', pt_size=4)

                # TODO: Project the filtered pointclouds on BEV image. Now filtered
                # pointclouds are already in camera coordinates.
                # point_bev = obj_panoptic_utils.project_points_on_BEV_image(img_idx,
                #                                                 dataset.calib_dir,
                #                                                 image_size,
                #                                                 filtered_points)
                # draw_points(prop_bev_axes, point_bev, 'red', pt_size=4)

            if draw_proposals_separate:
                # Save just the proposals
                filename = prop_out_dir + '/' + sample_name + '.jpg'
                print('Draw proposals_separate: ', filename)
                # Now add the legends
                # prop_bev_axes.legend(loc='best', shadow=True, fontsize=20)
                # prop_2d_axes.legend(loc='best', shadow=True, fontsize=20)
                # prop_3d_axes.legend(loc='upper right', shadow=True, fontsize=20)
                plt.savefig(filename)

                if not draw_overlaid:
                    plt.close(prop_fig)

        if draw_overlaid or draw_predictions_separate:
            # print('prediction_boxes_3d = ', prediction_boxes_3d)
            if len(prediction_boxes_3d) > 0:
                # Project the 3D box predictions to image space
                image_filter = []
                final_boxes_2d = []
                for i in range(len(prediction_boxes_3d)):
                    box_3d = prediction_boxes_3d[i, 0:7]
                    img_box = box_3d_panoptic_projector.project_to_image_space(
                        box_3d, calib_p2,
                        truncate=True, image_size=image_size,
                        discard_before_truncation=False, distortion=distortion)
                    if img_box is not None:
                        image_filter.append(True)
                        final_boxes_2d.append(img_box)
                    else:
                        image_filter.append(False)
                final_boxes_2d = np.asarray(final_boxes_2d)
                final_prediction_boxes_3d = prediction_boxes_3d[image_filter]
                final_scores = prediction_scores[image_filter]
                final_class_indices = prediction_class_indices[image_filter]

                num_of_predictions = final_boxes_2d.shape[0]

                # Convert to objs
                final_prediction_objs = \
                    [box_3d_panoptic_encoder.box_3d_to_object_label(
                        prediction, obj_type='Prediction')
                        for prediction in final_prediction_boxes_3d]
                for (obj, score) in zip(final_prediction_objs, final_scores):
                    obj.score = score
            else:
                if save_empty_images:
                    pred_fig, pred_bev_axes, pred_2d_axes, pred_3d_axes = \
                        vis_panoptic_utils.visualization(dataset.rgb_image_dir,
                                                         img_idx,
                                                         display=False,
                                                         fig_size=fig_size)
                    filename = pred_out_dir + '/' + sample_name + '.jpg'
                    plt.savefig(filename)
                    print('Draw empty_images: ', filename)
                    plt.close(pred_fig)
                continue

            if draw_overlaid:
                # Overlay prediction boxes on image
                draw_predictions(filtered_gt_objs,
                                 calib_p2,
                                 num_of_predictions,
                                 final_prediction_objs,
                                 final_class_indices,
                                 final_boxes_2d,
                                 prop_2d_axes,
                                 prop_3d_axes,
                                 prop_bev_axes,
                                 panoptic_utils.area_extents,
                                 bev_img.shape,
                                 draw_prediction_score,
                                 draw_prediction_iou,
                                 gt_classes,
                                 draw_orientations_on_pred,
                                 distortion=distortion)
                filename = overlaid_out_dir + '/' + sample_name + '.jpg'
                # Now add the legends
                # prop_bev_axes.legend(loc='best', shadow=True, fontsize=20)
                # prop_2d_axes.legend(loc='best', shadow=True, fontsize=20)
                # prop_3d_axes.legend(loc='upper right', shadow=True, fontsize=20)
                plt.savefig(filename)
                print('Draw overlaid: ', filename)
                plt.close(prop_fig)

            if draw_predictions_separate:
                # Now only draw prediction boxes on images
                # on a new figure handler
                if draw_projected_2d_boxes:
                    pred_fig, pred_bev_axes, pred_2d_axes, pred_3d_axes = \
                        vis_panoptic_utils.visualization(dataset.rgb_image_dir,
                                                         img_idx,
                                                         bev_img,
                                                         display=False,
                                                         fig_size=fig_size)

                    draw_predictions(filtered_gt_objs,
                                     calib_p2,
                                     num_of_predictions,
                                     final_prediction_objs,
                                     final_class_indices,
                                     final_boxes_2d,
                                     pred_2d_axes,
                                     pred_3d_axes,
                                     pred_bev_axes,
                                     panoptic_utils.area_extents,
                                     bev_img.shape,
                                     draw_prediction_score,
                                     draw_prediction_iou,
                                     gt_classes,
                                     draw_orientations_on_pred,
                                     distortion=distortion)
                    # Now add the legends
                    # pred_bev_axes.legend(loc='best', shadow=True, fontsize=20)
                    # pred_2d_axes.legend(loc='best', shadow=True, fontsize=20)
                    # pred_3d_axes.legend(loc='best', shadow=True, fontsize=20)
                else:
                    pred_fig, pred_3d_axes = \
                        vis_panoptic_utils.visualize_single_plot(
                            dataset.rgb_image_dir, img_idx, display=False)

                    draw_3d_predictions(filtered_gt_objs,
                                        calib_p2,
                                        num_of_predictions,
                                        final_prediction_objs,
                                        final_class_indices,
                                        final_boxes_2d,
                                        pred_3d_axes,
                                        draw_prediction_score,
                                        draw_prediction_iou,
                                        gt_classes,
                                        draw_orientations_on_pred,
                                        distortion=distortion)
                    # Now add the legends
                    # pred_3d_axes.legend(loc='upper right', shadow=True, fontsize=20)
                filename = pred_out_dir + '/' + sample_name + '.jpg'
                plt.savefig(filename)
                print('Draw predictions_separate: ', filename)
                plt.close(pred_fig)

    print('\nDone')


def draw_proposals(filtered_gt_objs,
                   p_matrix,
                   num_of_proposals,
                   proposal_objs,
                   proposal_scores,
                   proposal_boxes,
                   prop_2d_axes,
                   prop_3d_axes,
                   prop_bev_axes,
                   area_extents,
                   image_shape,
                   draw_bev,
                   draw_2d_box,
                   draw_3d_box,
                   draw_score,
                   draw_iou,
                   draw_orientations_on_prop,
                   distortion=None):
    # Draw filtered ground truth boxes
    gt_boxes = []
    tmp_num = 0
    for obj in filtered_gt_objs:
        # Draw groundtruth 2D boxes on Bird's Eye View image
        # print('obj.t = ', obj.t)
        vis_panoptic_utils.draw_box_bev(
            prop_bev_axes, obj, area_extents, image_shape,
            show_orientation=True,
            color_table=['r', 'y', 'r', 'w'], line_width=6,
            double_line=False,
            label='Groundtruth' if tmp_num == 0 else "")

        # Draw 2D boxes on camera image
        vis_panoptic_utils.draw_box_2d(
            prop_2d_axes, obj, test_mode=True, color_tm='r', label='Groundtruth' if tmp_num == 0 else "")

        # Draw 3D boxes
        # if obj.t[0]>1:
        #     print ('draw_proposals')
        #     print('obj.t = ', obj.t)
        #     print('obj.w = ', obj.w)
        #     print('obj.h = ', obj.h)
        #     print('obj.l = ', obj.l)
        vis_panoptic_utils.draw_box_3d(prop_3d_axes, obj, p_matrix,
                                       show_orientation=draw_orientations_on_prop,
                                       color_table=['r', 'y', 'r', 'w'],
                                       line_width=6,
                                       double_line=False,
                                       distortion=distortion,
                                       label='Groundtruth' if tmp_num == 0 else "")
        if draw_iou:
            gt_box_2d = [obj.x1, obj.y1, obj.x2, obj.y2]
            # print('gt_box_2d = ', gt_box_2d); [632.0063460656979, 0.0, 1257.3118986504794, 1080.0]
            gt_boxes.append(gt_box_2d)

        tmp_num = tmp_num + 1

    if gt_boxes:
        # the two_2 eval function expects np.array
        gt_boxes = np.asarray(gt_boxes)

    # Overlay proposal boxes on images
    for anchor_idx in range(num_of_proposals):
        obj_label = proposal_objs[anchor_idx]

        # Draw proposal 2D boxes on Bird's Eye View image
        if draw_bev:
            vis_panoptic_utils.draw_box_bev(
                prop_bev_axes, obj_label, area_extents, image_shape,
                show_orientation=False,  # Don't have to draw orietation for proposals because all proposals are parallel to the axis.
                color_table=['cornflowerblue', 'y', 'r', 'w'], line_width=5,
                double_line=False, label='proposals' if anchor_idx == 0 else "")

        # Draw 2D boxes (can't use obj_label since 2D corners are not
        # filled in)
        if draw_2d_box:
            rgb_box_2d = proposal_boxes[anchor_idx]

            box_x1 = rgb_box_2d[0]
            box_y1 = rgb_box_2d[1]
            box_w = rgb_box_2d[2] - box_x1
            box_h = rgb_box_2d[3] - box_y1

            rect = patches.Rectangle((box_x1, box_y1),
                                     box_w, box_h,
                                     linewidth=5,
                                     edgecolor='cornflowerblue',
                                     facecolor='none',
                                     label='proposals' if anchor_idx == 0 else "")

            prop_2d_axes.add_patch(rect)

            if draw_score or draw_iou:
                box_x2 = rgb_box_2d[2]
                box_y2 = rgb_box_2d[3]

                pred_box_2d = [box_x1,
                               box_y1,
                               box_x2,
                               box_y2]

                info_text_x = (box_x1 + box_x2) / 2
                info_text_y = box_y1

                draw_proposals_info(prop_2d_axes,
                                    info_text_x,
                                    info_text_y,
                                    proposal_scores[anchor_idx],
                                    pred_box_2d,
                                    gt_boxes,
                                    draw_score,
                                    draw_iou)

        # Draw 3D boxes
        if draw_3d_box:
            vis_panoptic_utils.draw_box_3d(prop_3d_axes, obj_label, p_matrix,
                                  show_orientation=draw_orientations_on_prop,
                                  color_table=[
                                      'cornflowerblue', 'y', 'r', 'w'],
                                  line_width=5,
                                  double_line=False,
                                  distortion=distortion,
                                  label='proposals' if anchor_idx == 0 else "")



def draw_predictions(filtered_gt_objs,
                     p_matrix,
                     predictions_to_show,
                     prediction_objs,
                     prediction_class,
                     final_boxes,
                     pred_2d_axes,
                     pred_3d_axes,
                     prop_bev_axes,
                     area_extents,
                     image_shape,
                     draw_score,
                     draw_iou,
                     gt_classes,
                     draw_orientations_on_pred,
                     distortion=None):
    # Draw filtered ground truth boxes
    gt_boxes = []
    tmp_num = 0
    for obj in filtered_gt_objs:
        # Draw groundtruth 2D boxes on Bird's Eye View image
        vis_panoptic_utils.draw_box_bev(
            prop_bev_axes, obj, area_extents, image_shape,
            show_orientation=True,
            color_table=['r', 'y', 'r', 'w'], line_width=6,
            double_line=False,
            label='Groundtruth' if tmp_num == 0 else "")

        # Draw 2D boxes
        vis_panoptic_utils.draw_box_2d(
            pred_2d_axes, obj, test_mode=True, color_tm='r',
            label='Groundtruth' if tmp_num == 0 else "")

        # Draw 3D boxes
        vis_panoptic_utils.draw_box_3d(pred_3d_axes, obj, p_matrix,
                              show_orientation=draw_orientations_on_pred,
                              color_table=['r', 'y', 'r', 'w'],
                              line_width=6,
                              double_line=False,
                              distortion=distortion,
                              label='Groundtruth' if tmp_num == 0 else "")
        if draw_iou:
            gt_box_2d = [obj.x1, obj.y1, obj.x2, obj.y2]
            gt_boxes.append(gt_box_2d)
        tmp_num = tmp_num + 1

    if gt_boxes:
        # the two_2 eval function expects np.array
        gt_boxes = np.asarray(gt_boxes)

    for pred_idx in range(predictions_to_show):
        pred_obj = prediction_objs[pred_idx]
        pred_class_idx = prediction_class[pred_idx]

        rgb_box_2d = final_boxes[pred_idx]

        box_x1 = rgb_box_2d[0]
        box_y1 = rgb_box_2d[1]
        box_w = rgb_box_2d[2] - box_x1
        box_h = rgb_box_2d[3] - box_y1

        box_cls = gt_classes[int(pred_class_idx)]
        rect = patches.Rectangle((box_x1, box_y1),
                                 box_w, box_h,
                                 linewidth=5,
                                 edgecolor=BOX_COLOUR_SCHEME[box_cls],
                                 facecolor='none',
                                 label='predictions' if pred_idx == 0 else "")

        pred_2d_axes.add_patch(rect)

        # Draw predictions 2D boxes on Bird's Eye View image
        vis_panoptic_utils.draw_box_bev(
            prop_bev_axes, pred_obj, area_extents, image_shape,
            show_orientation=True,
            color_table=[BOX_COLOUR_SCHEME[box_cls], 'y', 'r', 'w'], line_width=5,
            double_line=False,
            label='predictions' if pred_idx == 0 else "")

        # Draw 3D boxes
        vis_panoptic_utils.draw_box_3d(pred_3d_axes, pred_obj, p_matrix,
                                       show_orientation=draw_orientations_on_pred,
                                       color_table=['#00FF00', 'y', 'r', 'w'],
                                       line_width=5,
                                       double_line=False,
                                       box_color=BOX_COLOUR_SCHEME[box_cls],
                                       distortion=distortion,
                                       label='predictions' if pred_idx == 0 else "")

        if draw_score or draw_iou:
            box_x2 = rgb_box_2d[2]
            box_y2 = rgb_box_2d[3]

            pred_box_2d = [box_x1,
                           box_y1,
                           box_x2,
                           box_y2]

            info_text_x = (box_x1 + box_x2) / 2
            info_text_y = box_y1

            draw_prediction_info(pred_2d_axes,
                                 info_text_x,
                                 info_text_y,
                                 pred_obj,
                                 pred_class_idx,
                                 pred_box_2d,
                                 gt_boxes,
                                 draw_score,
                                 draw_iou,
                                 gt_classes)


def draw_points(ax, filtered_2d_pts, color_tm, pt_size):
    ax.plot(filtered_2d_pts[0, :], filtered_2d_pts[1, :], linestyle="None",
            marker='.', markersize=pt_size, color=color_tm, label='LiDAR points')


def draw_3d_predictions(filtered_gt_objs,
                        p_matrix,
                        predictions_to_show,
                        prediction_objs,
                        prediction_class,
                        final_boxes,
                        pred_3d_axes,
                        draw_score,
                        draw_iou,
                        gt_classes,
                        draw_orientations_on_pred,
                        distortion=None):
    # Draw filtered ground truth boxes
    gt_boxes = []
    for obj in filtered_gt_objs:
        # Draw 3D boxes
        vis_panoptic_utils.draw_box_3d(pred_3d_axes, obj, p_matrix,
                                       show_orientation=draw_orientations_on_pred,
                                       color_table=['r', 'y', 'r', 'w'],
                                       line_width=5,
                                       double_line=False,
                                       distortion=distortion)
        if draw_iou:
            gt_box_2d = [obj.x1, obj.y1, obj.x2, obj.y2]
            gt_boxes.append(gt_box_2d)

    if gt_boxes:
        # the two_2 eval function expects np.array
        gt_boxes = np.asarray(gt_boxes)

    for pred_idx in range(predictions_to_show):
        pred_obj = prediction_objs[pred_idx]
        pred_class_idx = prediction_class[pred_idx]

        rgb_box_2d = final_boxes[pred_idx]

        box_x1 = rgb_box_2d[0]
        box_y1 = rgb_box_2d[1]

        # Draw 3D boxes
        box_cls = gt_classes[int(pred_class_idx)]
        vis_panoptic_utils.draw_box_3d(pred_3d_axes, pred_obj, p_matrix,
                                       show_orientation=draw_orientations_on_pred,
                                       color_table=['#00FF00', 'y', 'r', 'w'],
                                       line_width=5,
                                       double_line=False,
                                       box_color=BOX_COLOUR_SCHEME[box_cls],
                                       distortion=distortion)

        if draw_score or draw_iou:
            box_x2 = rgb_box_2d[2]
            box_y2 = rgb_box_2d[3]

            pred_box_2d = [box_x1,
                           box_y1,
                           box_x2,
                           box_y2]

            info_text_x = (box_x1 + box_x2) / 2
            info_text_y = box_y1
            draw_prediction_info(pred_3d_axes,
                                 info_text_x,
                                 info_text_y,
                                 pred_obj,
                                 pred_class_idx,
                                 pred_box_2d,
                                 gt_boxes,
                                 draw_score,
                                 draw_iou,
                                 gt_classes)


def draw_prediction_info(ax, x, y,
                         pred_obj,
                         pred_class_idx,
                         pred_box_2d,
                         ground_truth,
                         draw_score,
                         draw_iou,
                         gt_classes):

    label = ""

    if draw_score:
        label += "sc:{:.2f}".format(pred_obj.score)

    if draw_iou and len(ground_truth) > 0:
        if draw_score:
            label += ', '
        iou = evaluation.two_d_iou(pred_box_2d, ground_truth)
        label += "iou:{:.3f}".format(max(iou))

    box_cls = gt_classes[int(pred_class_idx)]

    ax.text(x, y + 54,
            gt_classes[int(pred_class_idx)] + '\n' + label,
            verticalalignment='bottom',
            horizontalalignment='center',
            color=BOX_COLOUR_SCHEME[box_cls],
            fontsize=10,
            fontweight='bold',
            path_effects=[
                patheffects.withStroke(linewidth=5,
                                       foreground='black')])


def draw_proposals_info(ax, x, y,
                        proposal_scores,
                        proposal_box_2d,
                        ground_truth,
                        draw_score,
                        draw_iou):

    label = ""

    # print('proposal_scores = ', proposal_scores)
    if draw_score:
        label += "sc:{:.2f}".format(proposal_scores)

    # print('proposal_box_2d = ', proposal_box_2d)
    # print('ground_truth = ', ground_truth)
    if draw_iou and len(ground_truth) > 0:
        if draw_score:
            label += ', '
        iou = evaluation.two_d_iou(proposal_box_2d, ground_truth)
        label += "iou:{:.3f}".format(max(iou))

    ax.text(x, y + 150,
            label,
            verticalalignment='bottom',
            horizontalalignment='center',
            color='cornflowerblue',
            fontsize=10,
            fontweight='bold',
            path_effects=[
                patheffects.withStroke(linewidth=5,
                                       foreground='black')])


def get_point_cloud_in_image_fov(point_cloud, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    """Filter lidar points, keep those in image FOV"""
    pts_2d = calib.project_velo_to_image(point_cloud)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
        (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (point_cloud[:, 0] > clip_distance)
    imgfov_point_cloud = point_cloud[fov_inds, :]
    if return_more:
        return imgfov_point_cloud, pts_2d, fov_inds
    else:
        return imgfov_point_cloud


def draw_point_cloud_on_image(point_cloud, img, calib, img_width, img_height):
    """Project LiDAR points to image"""
    imgfov_point_cloud, pts_2d, fov_inds = get_point_cloud_in_image_fov(point_cloud,
        calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds,:]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_point_cloud)

    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3]*255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        color = cmap[int(640.0/depth), :]
        cv2.circle(img, (int(np.round(imgfov_pts_2d[i, 0])),
            int(np.round(imgfov_pts_2d[i, 1]))),
            2, color=tuple(color), thickness=-1)
    Image.fromarray(img).show()
    return img


def get_point_filtered(point_cloud, extents, ground_plane=None, offset_dist=2.0):
    """
    Creates a point filter using the 3D extents and ground plane

    :param point_cloud: Point cloud in the form [[x,...],[y,...],[z,...]]
    :param extents: 3D area in the form
        [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
    :param ground_plane: Optional, coefficients of the ground plane
        (a, b, c, d)
    :param offset_dist: If ground_plane is provided, removes points above
        this offset from the ground_plane
    :return: A binary mask for points within the extents and offset plane
    """

    point_cloud = np.asarray(point_cloud)

    # Filter points within certain xyz range
    x_extents = extents[0]
    y_extents = extents[1]
    z_extents = extents[2]
    print('x_extents =', x_extents)
    print('y_extents =', y_extents)
    print('z_extents =', z_extents)
    extents_filter = (point_cloud[0] > x_extents[0]) & \
                     (point_cloud[0] < x_extents[1]) & \
                     (point_cloud[1] > y_extents[0]) & \
                     (point_cloud[1] < y_extents[1]) & \
                     (point_cloud[2] > z_extents[0]) & \
                     (point_cloud[2] < z_extents[1])

    if ground_plane is not None:
        ground_plane = np.array(ground_plane)

        # Calculate filter using ground plane
        ones_col = np.ones(point_cloud.shape[1])
        padded_points = np.vstack([point_cloud, ones_col])

        offset_plane = ground_plane + [0, 0, 0, -offset_dist]

        # Create plane filter
        dot_prod = np.dot(offset_plane, padded_points)
        plane_filter = dot_prod < 0

        # Combine the two filters
        point_filter = np.logical_and(extents_filter, plane_filter)
    else:
        # Only use the extents for filtering
        point_filter = extents_filter
        print('point_filter = ', point_filter)

    return point_filter


if __name__ == '__main__':
    main()
