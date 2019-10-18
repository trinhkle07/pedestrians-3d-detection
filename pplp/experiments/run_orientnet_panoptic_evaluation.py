"""OrientNet evaluator.

This runs the OrientNet evaluator.
"""

import argparse
import os
# Add this block for ROS python conflict
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    sys.path.remove('$HOME/segway_kinetic_ws/devel/lib/python2.7/dist-packages')
except ValueError:
    pass

import tensorflow as tf

import pplp
import pplp.builders.config_builder_panoptic_util as config_builder
from pplp.builders.dataset_panoptic_builder import DatasetBuilder
from pplp.core.models.orientnet_panoptic_model import OrientModel
from pplp.core.orientnet_panoptic_evaluator import Evaluator


def evaluate(model_config, eval_config, dataset_config):

    # Parse eval config
    eval_mode = eval_config.eval_mode
    if eval_mode not in ['val', 'test']:
        raise ValueError('Evaluation mode can only be set to `val` or `test`')
    evaluate_repeatedly = eval_config.evaluate_repeatedly

    # Parse dataset config
    data_split = dataset_config.data_split
    if data_split.startswith('train'):
        dataset_config.data_split_dir = 'training'
        dataset_config.has_labels = True

    elif data_split.startswith('val'):
        dataset_config.data_split_dir = 'training'

        # Don't load labels for val split when running in test mode
        if eval_mode == 'val':
            dataset_config.has_labels = True
        elif eval_mode == 'test':
            dataset_config.has_labels = False

    elif data_split.startswith('test'):
        dataset_config.data_split_dir = 'testing'
        dataset_config.has_labels = False

    else:
        raise ValueError('Invalid data split', data_split)

    # Convert to object to overwrite repeated fields
    dataset_config = config_builder.proto_to_obj(dataset_config)

    # Remove augmentation during evaluation
    dataset_config.aug_list = []

    # Build the dataset object
    dataset = DatasetBuilder.build_panoptic_dataset(dataset_config,
                                                    use_defaults=False)

    # Setup the model
    model_name = model_config.model_name

    # Convert to object to overwrite repeated fields
    model_config = config_builder.proto_to_obj(model_config)

    # Switch path drop off during evaluation
    model_config.path_drop_probabilities = [1.0, 1.0]

    with tf.Graph().as_default():
        if model_name == 'orient_model':
            model = OrientModel(model_config, train_val_test=eval_mode,
                                dataset=dataset)
        else:
            raise ValueError('Invalid model name {}'.format(model_name))

        model_evaluator = Evaluator(model,
                                    dataset_config,
                                    eval_config)

        if evaluate_repeatedly:
            model_evaluator.repeated_checkpoint_run()
        else:
            model_evaluator.run_latest_checkpoints()


def main(_):
    parser = argparse.ArgumentParser()

    default_pipeline_config_path = pplp.root_dir() + \
        '/configs/orientation_pedestrian_panoptic.config'

    parser.add_argument('--pipeline_config',
                        type=str,
                        dest='pipeline_config_path',
                        default=default_pipeline_config_path,
                        help='Path to the pipeline config')

    parser.add_argument('--data_split',
                        type=str,
                        dest='data_split',
                        default='val',
                        help='Data split for evaluation')

    parser.add_argument('--device',
                        type=str,
                        dest='device',
                        default='0',
                        help='CUDA device id')

    args = parser.parse_args()

    # Parse pipeline config
    model_config, _, eval_config, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            args.pipeline_config_path,
            is_training=False)

    # Overwrite data split
    dataset_config.data_split = args.data_split

    # Overwrite eval_config if data_split is 'test'
    if args.data_split.startswith('test'):
        eval_config.eval_mode = 'test'

    # Set CUDA device id
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    print('model_config = ', model_config)
    print('eval_config = ', eval_config)
    print('dataset_config = ', dataset_config)
    evaluate(model_config, eval_config, dataset_config)


if __name__ == '__main__':
    tf.app.run()
