import sys
import datetime
import subprocess
from distutils import dir_util

import numpy as np
import os
from PIL import Image
import tensorflow as tf

from wavedata.tools.core import calib_panoptic_utils

import pplp
from pplp.core import box_3d_panoptic_projector
from pplp.core import summary_utils


def save_predictions_in_panoptic_format(model,
                                     checkpoint_name,
                                     data_split,
                                     score_threshold,
                                     global_step):
    """ Converts a set of network predictions into text files required for
    PANOPTIC evaluation.
    """

    dataset = model.dataset
    # Round this because protobuf encodes default values as full decimal
    score_threshold = round(score_threshold, 3)

    # Get available prediction folders
    predictions_root_dir = pplp.root_dir() + '/data/outputs/' + \
        checkpoint_name + '/predictions'

    final_predictions_root_dir = predictions_root_dir + \
        '/final_predictions_and_scores/' + dataset.data_split

    final_predictions_dir = final_predictions_root_dir + \
        '/' + str(global_step)

    # 3D prediction directories
    panoptic_predictions_3d_dir = predictions_root_dir + \
        '/panoptic_pplp_eval/' + \
        str(score_threshold) + '/' + \
        str(global_step) + '/data'

    if not os.path.exists(panoptic_predictions_3d_dir):
        os.makedirs(panoptic_predictions_3d_dir)

    # Do conversion
    num_samples = dataset.num_samples
    num_valid_samples = 0

    print('\nGlobal step:', global_step)
    print('Converting detections from:', final_predictions_dir)

    print('3D Detections being saved to:', panoptic_predictions_3d_dir)

    for sample_idx in range(num_samples):
        # Print progress
        sys.stdout.write('\rConverting {} / {}'.format(
            sample_idx + 1, num_samples))
        sys.stdout.flush()

        sample_name = dataset.sample_names[sample_idx]

        prediction_file = sample_name + '.txt'

        panoptic_predictions_3d_file_path = panoptic_predictions_3d_dir + \
            '/' + prediction_file

        predictions_file_path = final_predictions_dir + \
            '/' + prediction_file

        # If no predictions, skip to next file
        if not os.path.exists(predictions_file_path):
            np.savetxt(panoptic_predictions_3d_file_path, [])
            continue

        all_predictions = np.loadtxt(predictions_file_path)

        # # Swap l, w for predictions where w > l
        # swapped_indices = all_predictions[:, 4] > all_predictions[:, 3]
        # fixed_predictions = np.copy(all_predictions)
        # fixed_predictions[swapped_indices, 3] = all_predictions[
        #     swapped_indices, 4]
        # fixed_predictions[swapped_indices, 4] = all_predictions[
        #     swapped_indices, 3]

        all_predictions = np.array(all_predictions)

        # change 1D array in to 2D array even if it has only one row.
        if len(all_predictions.shape) == 1:
            all_predictions.shape = (1, -1)

        score_filter = all_predictions[:, 7] >= score_threshold
        all_predictions = all_predictions[score_filter]
        # If no predictions, skip to next file
        if len(all_predictions) == 0:
            np.savetxt(panoptic_predictions_3d_file_path, [])
            continue

        # Project to image space
        sample_name = prediction_file.split('.')[0]
        img_idx = int(sample_name)

        # Load image for truncation
        image = Image.open(dataset.get_rgb_image_path(sample_name))

        stereo_calib_p2 = calib_panoptic_utils.read_calibration(dataset.calib_dir,
                                                       img_idx).HD_11

        boxes = []
        image_filter = []
        for i in range(len(all_predictions)):
            box_3d = all_predictions[i, 0:7]
            img_box = box_3d_panoptic_projector.project_to_image_space(
                box_3d, stereo_calib_p2,
                truncate=True, image_size=image.size)

            # Skip invalid boxes (outside image space)
            if img_box is None:
                image_filter.append(False)
                print('**ERROR img_box = ', img_box)
                continue

            image_filter.append(True)
            boxes.append(img_box)

        boxes = np.asarray(boxes)
        all_predictions = all_predictions[image_filter]

        # If no predictions, skip to next file
        if len(boxes) == 0:
            np.savetxt(panoptic_predictions_3d_file_path, [])
            continue

        num_valid_samples += 1

        # To keep each value in its appropriate position, an array of zeros
        # (N, 16) is allocated but only values [4:16] are used
        panoptic_predictions = np.zeros([len(boxes), 16])

        # Get object types
        all_pred_classes = all_predictions[:, 8].astype(np.int32)
        obj_types = [dataset.classes[class_idx]
                     for class_idx in all_pred_classes]

        # Truncation and Occlusion are always empty (see below)

        # Alpha (Not computed)
        panoptic_predictions[:, 3] = -10 * np.ones((len(panoptic_predictions)),
                                                dtype=np.int32)

        # 2D predictions
        panoptic_predictions[:, 4:8] = boxes[:, 0:4]

        # 3D predictions
        # (l, w, h)
        panoptic_predictions[:, 8] = all_predictions[:, 5]
        panoptic_predictions[:, 9] = all_predictions[:, 4]
        panoptic_predictions[:, 10] = all_predictions[:, 3]
        # (x, y, z)
        panoptic_predictions[:, 11:14] = all_predictions[:, 0:3]
        # (ry, score)
        panoptic_predictions[:, 14:16] = all_predictions[:, 6:8]

        # Round detections to 3 decimal places
        panoptic_predictions = np.round(panoptic_predictions, 3)

        # Empty Truncation, Occlusion
        panoptic_empty_1 = -1 * np.ones((len(panoptic_predictions), 2),
                                     dtype=np.int32)

        # Stack 3D predictions text
        panoptic_text_3d = np.column_stack([obj_types,
                                         panoptic_empty_1,
                                         panoptic_predictions[:, 3:16]])

        # Save to text files
        np.savetxt(panoptic_predictions_3d_file_path, panoptic_text_3d,
                   newline='\r\n', fmt='%s')

    print('\nNum valid:', num_valid_samples)
    print('Num samples:', num_samples)


def set_up_summary_writer(model_config,
                          sess):
    """ Helper function to set up log directories and summary
        handlers.
    Args:
        model_config: Model protobuf configuration
        sess : A tensorflow session
    """

    paths_config = model_config.paths_config

    logdir = paths_config.logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logdir = logdir + '/eval'

    datetime_str = str(datetime.datetime.now())
    summary_writer = tf.summary.FileWriter(logdir + '/' + datetime_str,
                                           sess.graph)

    global_summaries = set([])
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summary_merged = summary_utils.summaries_to_keep(summaries,
                                                     global_summaries,
                                                     histograms=False,
                                                     input_imgs=False,
                                                     input_bevs=False)

    return summary_writer, summary_merged


def strip_checkpoint_id(checkpoint_dir):
    """Helper function to return the checkpoint index number.

    Args:
        checkpoint_dir: Path directory of the checkpoints

    Returns:
        checkpoint_id: An int representing the checkpoint index
    """

    checkpoint_name = checkpoint_dir.split('/')[-1]
    return int(checkpoint_name.split('-')[-1])


def print_inference_time_statistics(total_feed_dict_time,
                                    total_inference_time):

    # Print feed_dict time stats
    total_feed_dict_time = np.asarray(total_feed_dict_time)
    print('Feed dict time:')
    print('Min: ', np.round(np.min(total_feed_dict_time), 5))
    print('Max: ', np.round(np.max(total_feed_dict_time), 5))
    print('Mean: ', np.round(np.mean(total_feed_dict_time), 5))
    print('Median: ', np.round(np.median(total_feed_dict_time), 5))

    # Print inference time stats
    total_inference_time = np.asarray(total_inference_time)
    print('Inference time:')
    print('Min: ', np.round(np.min(total_inference_time), 5))
    print('Max: ', np.round(np.max(total_inference_time), 5))
    print('Mean: ', np.round(np.mean(total_inference_time), 5))
    print('Median: ', np.round(np.median(total_inference_time), 5))


def copy_panoptic_native_code(checkpoint_name):
    """Copies and compiles panoptic native code.

    It also creates neccessary directories for storing the results
    of the panoptic native evaluation code.
    """

    pplp_root_dir = pplp.root_dir()
    panoptic_native_code_copy = pplp_root_dir + '/data/outputs/' + \
        checkpoint_name + '/predictions/panoptic_pplp_eval/'

    # Only copy if the code has not been already copied over
    if not os.path.exists(panoptic_native_code_copy):

        os.makedirs(panoptic_native_code_copy)
        original_panoptic_native_code = pplp.top_dir() + \
            '/scripts/offline_eval/panoptic_pplp_eval/'

        predictions_dir = pplp_root_dir + '/data/outputs/' + \
            checkpoint_name + '/predictions/'
        # create dir for it first
        dir_util.copy_tree(original_panoptic_native_code,
                           panoptic_native_code_copy)
        # run the script to compile the c++ code
        script_folder = predictions_dir + \
            '/panoptic_pplp_eval/'
        make_script = script_folder + 'run_make.sh'
        subprocess.call([make_script, script_folder])

    # Set up the results folders if they don't exist
    results_dir = pplp.top_dir() + '/scripts/offline_eval/results'
    results_05_dir = pplp.top_dir() + '/scripts/offline_eval/results_05_iou'
    results_orient_05_dir = pplp.top_dir() + '/scripts/offline_eval/results_orient_05_iou'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(results_05_dir):
        os.makedirs(results_05_dir)
    if not os.path.exists(results_orient_05_dir):
        os.makedirs(results_orient_05_dir)


def run_panoptic_native_script(checkpoint_name, score_threshold, global_step):
    """Runs the panoptic native code script."""

    eval_script_dir = pplp.root_dir() + '/data/outputs/' + \
        checkpoint_name + '/predictions'
    make_script = eval_script_dir + \
        '/panoptic_pplp_eval/run_eval.sh'
    script_folder = eval_script_dir + \
        '/panoptic_pplp_eval/'

    results_dir = pplp.top_dir() + '/scripts/offline_eval/results/'

    # Round this because protobuf encodes default values as full decimal
    score_threshold = round(score_threshold, 3)
    print('--------------- run_panoptic_native_script(checkpoint_name, score_threshold, global_step) -----------')
    print('make_script = ', make_script)
    print('script_folder = ', script_folder)
    print('score_threshold = ', score_threshold)
    print('global_step = ', global_step)
    print('checkpoint_name = ', checkpoint_name)
    print('results_dir = ', results_dir)
    subprocess.call([make_script, script_folder,
                     str(score_threshold),
                     str(global_step),
                     str(checkpoint_name),
                     str(results_dir)])


def run_panoptic_native_script_with_05_iou(checkpoint_name, score_threshold,
                                        global_step):
    """Runs the panoptic native code script."""

    eval_script_dir = pplp.root_dir() + '/data/outputs/' + \
        checkpoint_name + '/predictions'
    make_script = eval_script_dir + \
        '/panoptic_pplp_eval/run_eval_05_iou.sh'
    script_folder = eval_script_dir + \
        '/panoptic_pplp_eval/'

    results_dir = pplp.top_dir() + '/scripts/offline_eval/results_05_iou/'

    print('--------------- run_panoptic_native_script_with_05_iou(checkpoint_name, score_threshold, global_step) -----------')
    print('make_script = ', make_script)
    print('script_folder = ', script_folder)
    print('score_threshold = ', score_threshold)
    print('global_step = ', global_step)
    print('checkpoint_name = ', checkpoint_name)
    print('results_dir = ', results_dir)

    # Round this because protobuf encodes default values as full decimal
    score_threshold = round(score_threshold, 3)

    subprocess.call([make_script, script_folder,
                     str(score_threshold),
                     str(global_step),
                     str(checkpoint_name),
                     str(results_dir)])


def run_panoptic_orient_script_with_05_iou(checkpoint_name, score_threshold,
                                           global_step):
    """Runs the panoptic native code script."""

    eval_script_dir = pplp.root_dir() + '/data/outputs/' + \
        checkpoint_name + '/predictions'
    make_script = eval_script_dir + \
        '/panoptic_pplp_eval/run_eval_orient_05_iou.sh'
    script_folder = eval_script_dir + \
        '/panoptic_pplp_eval/'

    results_dir = pplp.top_dir() + '/scripts/offline_eval/results_orient_05_iou/'

    print('--------------- run_panoptic_orient_script_with_05_iou(checkpoint_name, score_threshold, global_step) -----------')
    print('make_script = ', make_script)
    print('script_folder = ', script_folder)
    print('score_threshold = ', score_threshold)
    print('global_step = ', global_step)
    print('checkpoint_name = ', checkpoint_name)
    print('results_dir = ', results_dir)

    # Round this because protobuf encodes default values as full decimal
    score_threshold = round(score_threshold, 3)

    subprocess.call([make_script, script_folder,
                     str(score_threshold),
                     str(global_step),
                     str(checkpoint_name),
                     str(results_dir)])
