import tensorflow as tf

from pplp.core.pplp_fc_layers import orientnet_resnet_layers

KEY_ANGLE_VECTORS = 'angle_vectors'


def build(layers_config, is_training, rgb_img_crop=None):
    """Builds second stage fully connected layers

    Args:
        layers_config: Configuration object
        is_training (bool): Whether the network is training or evaluating
        rgb_img_crop: The crop of each pedestrian in rgb image.

    Returns:
        fc_output_layers: Output layer dictionary
    """
    with tf.variable_scope('box_predictor'):
        fc_layers_type = layers_config.WhichOneof('fc_layers')

        if fc_layers_type == 'resnet_fc_layers':
            fc_layers_config = layers_config.resnet_fc_layers

            # Here, angle_vectors is the angle_quaternions
            angle_vectors = \
                orientnet_resnet_layers.build(
                    fc_layers_config=fc_layers_config,
                    is_training=is_training,
                    rgb_img_crop=rgb_img_crop)

        else:
            raise ValueError('Invalid fc layers config')

    fc_output_layers = dict()
    if fc_layers_type == 'resnet_fc_layers':
        fc_output_layers[KEY_ANGLE_VECTORS] = angle_vectors
    else:
        raise ValueError('Invalid fc_layers_type')

    return fc_output_layers
