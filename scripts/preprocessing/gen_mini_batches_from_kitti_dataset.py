import os

import numpy as np

# Add this block for ROS python conflict
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    sys.path.remove('$HOME/segway_kinetic_ws/devel/lib/python2.7/dist-packages')
except ValueError:
    pass

import pplp
from pplp.builders.dataset_builder import DatasetBuilder


def do_preprocessing(dataset, indices):

    mini_batch_utils = dataset.kitti_utils.mini_batch_utils

    print("Generating mini batches in {}".format(
        mini_batch_utils.mini_batch_dir))

    # Generate all mini-batches, this can take a long time
    mini_batch_utils.preprocess_mrcnn_mini_batches(indices)
    mini_batch_utils.preprocess_rpn_mini_batches(indices)

    print("Mini batches generated")


def split_indices(dataset, num_children):
    """Splits indices between children

    Args:
        dataset: Dataset object
        num_children: Number of children to split samples between

    Returns:
        indices_split: A list of evenly split indices
    """

    all_indices = np.arange(dataset.num_samples)

    # Pad indices to divide evenly
    length_padding = (-len(all_indices)) % num_children
    padded_indices = np.concatenate((all_indices,
                                     np.zeros(length_padding,
                                              dtype=np.int32)))

    # Split and trim last set of indices to original length
    indices_split = np.split(padded_indices, num_children)
    indices_split[-1] = np.trim_zeros(indices_split[-1])

    return indices_split


def split_work(all_child_pids, dataset, indices_split, num_children):
    """Spawns children to do work

    Args:
        all_child_pids: List of child pids are appended here, the parent
            process should use this list to wait for all children to finish
        dataset: Dataset object
        indices_split: List of indices to split between children
        num_children: Number of children
    """

    for child_idx in range(num_children):
        new_pid = os.fork()
        if new_pid:
            all_child_pids.append(new_pid)
        else:
            indices = indices_split[child_idx]
            print('child', dataset.classes,
                  indices_split[child_idx][0],
                  indices_split[child_idx][-1])
            do_preprocessing(dataset, indices)
            os._exit(0)


def main():
    """Generates anchors info which is used for mini batch sampling.

    Processing on 'Cars' can be split into multiple processes, see the Options
    section for configuration.

    Args:
        dataset: KittiDataset (optional)
            If dataset is provided, only generate info for that dataset.
            If no dataset provided, generates info for all 3 classes.
    """

    kitti_dataset_config_path = pplp.root_dir() + \
        '/configs/mb_preprocessing/rpn_pedestrians.config'

    ##############################
    # Options
    ##############################
    # Serial vs parallel processing
    in_parallel = False

    process_ped = True

    # Number of child processes to fork, samples will
    #  be divided evenly amongst the processes (in_parallel must be True)
    num_ped_children = 8

    ##############################
    # Dataset setup
    ##############################

    if process_ped:
        ped_dataset = DatasetBuilder.load_dataset_from_config(
            kitti_dataset_config_path)

    ##############################
    # Serial Processing
    ##############################
    if not in_parallel:
        if process_ped:
            do_preprocessing(ped_dataset, None)

        print('All Done (Serial)')

    ##############################
    # Parallel Processing
    ##############################
    else:

        # List of all child pids to wait on
        all_child_pids = []

        # Pedestrians
        if process_ped:
            ped_indices_split = split_indices(ped_dataset, num_ped_children)
            split_work(
                all_child_pids,
                ped_dataset,
                ped_indices_split,
                num_ped_children)

        # Wait to child processes to finish
        print('num children:', len(all_child_pids))
        for i, child_pid in enumerate(all_child_pids):
            os.waitpid(child_pid, 0)

        print('All Done (Parallel)')


if __name__ == '__main__':
    main()
