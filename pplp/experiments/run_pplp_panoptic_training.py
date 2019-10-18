"""Detection model trainer.

This runs the DetectionModel trainer.
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
from pplp.core.models.pplp_panoptic_model import PPLPModel
from pplp.core.models.rpn_pplp_panoptic_model import RpnModel
from pplp.core import trainer

tf.logging.set_verbosity(tf.logging.ERROR)


def train(model_config, train_config, dataset_config):

    dataset = DatasetBuilder.build_panoptic_dataset(dataset_config,
                                                    use_defaults=False)

    train_val_test = 'train'
    model_name = model_config.model_name

    with tf.Graph().as_default():
        if model_name == 'rpn_model':
            model = RpnModel(model_config,
                             train_val_test=train_val_test,
                             dataset=dataset)
        elif model_name == 'pplp_model':
            model = PPLPModel(model_config,
                              train_val_test=train_val_test,
                              dataset=dataset)
        else:
            raise ValueError('Invalid model_name')

        trainer.train(model, train_config)


def main(_):
    parser = argparse.ArgumentParser()

    # Defaults
    default_pipeline_config_path = pplp.root_dir() + \
        '/configs/pplp_pedestrian_panoptic.config'
    default_data_split = 'train'
    default_device = '1'

    parser.add_argument('--pipeline_config',
                        type=str,
                        dest='pipeline_config_path',
                        default=default_pipeline_config_path,
                        help='Path to the pipeline config')

    parser.add_argument('--data_split',
                        type=str,
                        dest='data_split',
                        default=default_data_split,
                        help='Data split for training')

    parser.add_argument('--device',
                        type=str,
                        dest='device',
                        default=default_device,
                        help='CUDA device id')

    args = parser.parse_args()

    # Parse pipeline config
    model_config, train_config, _, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            args.pipeline_config_path, is_training=True)

    # Overwrite data split
    dataset_config.data_split = args.data_split

    # Set CUDA device id
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    train(model_config, train_config, dataset_config)


if __name__ == '__main__':
    tf.app.run()
