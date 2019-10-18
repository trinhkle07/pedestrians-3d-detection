import tensorflow as tf

from pplp.core.pplp_fc_layers import ops


def build(fc_layers_config,
          is_training,
          rgb_img_crop):
    """Builds fusion layers

    Args:
        fc_layers_config: Fully connected layers config object
        is_training: Whether the network is training or evaluating
        rgb_img_crop: The crop of each pedestrian in rgb image.)

    Returns:
        angle_quaternions: Output angle vectors (or None)
    """

    # Parse configs
    fusion_type = fc_layers_config.fusion_type
    norm_type = fc_layers_config.norm_type
    keep_prob = fc_layers_config.keep_prob

    if fusion_type == 'early':
        angle_quaternions = \
            _early_fusion_fc_layers(keep_prob=keep_prob,
                                    is_training=is_training,
                                    rgb_img_crop=rgb_img_crop,
                                    norm_type=norm_type)
    else:
        print('fusion_type = ', fusion_type)
        raise ValueError('Invalid fusion type {}'.format(fusion_type))

    return angle_quaternions


# ResNet18 architecture
def _quat_resnet(tensor_in, norm_type, is_training, keep_prob):

    with tf.variable_scope('Quat_Net', reuse=tf.AUTO_REUSE):
        # tensor_in = rpn_fusion_out
        # vp_mask = tf.expand_dims(vp_mask, -1)
        # Output (bs, 32, 32, 64)
        conv1 = ops.conv2d('conv1', tensor_in, 7, 64, stride=2, norm=norm_type, is_training=is_training, act=None)
        # net.quat_net['conv1'] = conv1
        # Output (bs, 16, 16, 64)
        pool1 = tf.layers.max_pooling2d(conv1, 3, 2, padding='same', name='pool1')
        # net.quat_net['pool1'] = pool1

        # Output (bs, 16, 16, 64)
        conv2_1a = ops.conv2d('conv2_1a', pool1, 3, 64, stride=1, norm=norm_type, is_training=is_training)
        # net.quat_net['conv2_1a'] = conv2_1a
        conv2_2a = ops.conv2d('conv2_2a', conv2_1a, 3, 64, stride=1, norm=norm_type, is_training=is_training)
        # net.quat_net['conv2_2a'] = conv2_2a
        res_2a = tf.add_n([conv2_2a, pool1], name='res_2a')

        conv2_1b = ops.conv2d('conv2_1b', res_2a, 3, 64, stride=1, norm=norm_type, is_training=is_training)
        # net.quat_net['conv2_1b'] = conv2_1b
        conv2_2b = ops.conv2d('conv2_2b', conv2_1b, 3, 64, stride=1, norm=norm_type, is_training=is_training)
        # net.quat_net['conv2_2b'] = conv2_2b
        res_2b = tf.add_n([conv2_2b, res_2a], name='res_2b')

        # Output (bs, 8, 8, 128)
        conv3_1a = ops.conv2d('conv3_1a', res_2b, 3, 128, stride=2, norm=norm_type, is_training=is_training)
        # net.quat_net['conv3_1a'] = conv3_1a
        conv3_2a = ops.conv2d('conv3_2a', conv3_1a, 3, 128, stride=1, norm=norm_type, is_training=is_training)
        # net.quat_net['conv3_2a'] = conv3_2a
        res_2b_skip = ops.conv2d('res_2b_skip', res_2b, 1, 128, stride=2, norm=norm_type, is_training=is_training)
        res_3a = tf.add_n([conv3_2a, res_2b_skip], name='res_3a')

        conv3_1b = ops.conv2d('conv3_1b', res_3a, 3, 128, stride=1, norm=norm_type, is_training=is_training)
        # net.quat_net['conv3_1b'] = conv3_1b
        conv3_2b = ops.conv2d('conv3_2b', conv3_1b, 3, 128, stride=1, norm=norm_type, is_training=is_training)
        # net.quat_net['conv3_2b'] = conv3_2b
        res_3b = tf.add_n([conv3_2b, res_3a], name='res_3b')

        # Output (bs, 4, 4, 256)
        conv4_1a = ops.conv2d('conv4_1a', res_3b, 3, 256, stride=2, norm=norm_type, is_training=is_training)
        # net.quat_net['conv4_1a'] = conv4_1a
        conv4_2a = ops.conv2d('conv4_2a', conv4_1a, 3, 256, stride=1, norm=norm_type, is_training=is_training)
        # net.quat_net['conv4_2a'] = conv4_2a
        res_3b_skip = ops.conv2d('res_3b_skip', res_3b, 1, 256, stride=2, norm=norm_type, is_training=is_training)
        res_4a = tf.add_n([conv4_2a, res_3b_skip], name='res_4a')

        conv4_1b = ops.conv2d('conv4_1b', res_4a, 3, 256, stride=1, norm=norm_type, is_training=is_training)
        # net.quat_net['conv4_1b'] = conv4_1b
        conv4_2b = ops.conv2d('conv4_2b', conv4_1b, 3, 256, stride=1, norm=norm_type, is_training=is_training)
        # net.quat_net['conv4_2b'] = conv4_2b
        res_4b = tf.add_n([conv4_2b, res_4a], name='res_4b')

        # Output (bs, 2, 2, 512)
        conv5_1a = ops.conv2d('con5_1a', res_4b, 3, 512, stride=2, norm=norm_type, is_training=is_training)
        # net.quat_net['con5_1a'] = conv5_1a
        conv5_2a = ops.conv2d('con5_2a', conv5_1a, 3, 512, stride=1, norm=norm_type, is_training=is_training)
        # net.quat_net['con5_2a'] = conv5_2a
        res_4b_skip = ops.conv2d('res_4b_skip', res_4b, 1, 512, stride=2, norm=norm_type, is_training=is_training)
        res_5a = tf.add_n([conv5_2a, res_4b_skip], name='res_5a')

        conv5_1b = ops.conv2d('conv5_1b', res_5a, 3, 512, stride=1, norm=norm_type, is_training=is_training)
        # net.quat_net['conv5_1b'] = conv5_1b
        conv5_2b = ops.conv2d('conv5_2b', conv5_1b, 3, 512, stride=1, norm=norm_type, is_training=is_training)
        # net.quat_net['conv5_2b'] = conv5_2b
        res_5b = tf.add_n([conv5_2b, res_5a], name='res_5b')
        res_5b = ops.dropout(res_5b, keep_prob)  # res_5b.shape is (?x2x2x512)

        # Output (bs, 4*num_classes)
        num_classes = 1
        fc1 = ops.fully_connected('fc1', res_5b, 512)  # fc1.shape is (?x512)
        # net.quat_net['fc1'] = fc1
        fc2 = ops.fully_connected('fc2', fc1, 4*num_classes)
        # net.quat_net['fc2'] = fc2
        # out = tf.tanh(fc2)
        out = fc2
        # net.quat_net['out'] = out

    return out


def _early_fusion_fc_layers(keep_prob, is_training, rgb_img_crop, norm_type):

    # Build the layers for Orientation (Quaternion)
    crop_quats = _quat_resnet(rgb_img_crop, norm_type, is_training, keep_prob)

    return crop_quats
