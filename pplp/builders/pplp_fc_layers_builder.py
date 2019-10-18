import tensorflow as tf

from pplp.core.pplp_fc_layers import basic_fc_layers
from pplp.core.pplp_fc_layers import fusion_fc_layers
from pplp.core.pplp_fc_layers import fusion_fc_angle_cl_layers
from pplp.core.pplp_fc_layers import separate_fc_layers

KEY_CLS_LOGITS = 'classification_logits'
KEY_OFFSETS = 'offsets'

KEY_ENDPOINTS = 'end_points'


def build(layers_config,
          input_rois, input_weights,
          num_final_classes, box_rep,
          top_anchors, ground_plane,
          is_training, img_keypoints=None, rgb_img_crop=None):
    """Builds second stage fully connected layers

    Args:
        layers_config: Configuration object
        input_rois: List of input ROI feature maps
        input_weights: List of weights for each input e.g. [1.0, 1.0]
        num_final_classes: Number of output classes, including 'Background'
        box_rep: Box representation (e.g. 'box_3d', 'box_8c', etc.)
        top_anchors: Top proposal anchors, to include location information
        ground_plane: Ground plane coefficients
        is_training (bool): Whether the network is training or evaluating

    Returns:
        fc_output_layers: Output layer dictionary
    """

    # Default all output layers to None
    cls_logits = offsets = end_points = None

    with tf.variable_scope('box_predictor') as sc:
        end_points_collection = sc.name + '_end_points'

        fc_layers_type = layers_config.WhichOneof('fc_layers')

        if fc_layers_type == 'basic_fc_layers':
            fc_layers_config = layers_config.basic_fc_layers

            cls_logits, offsets, end_points = \
                basic_fc_layers.build(
                    fc_layers_config=fc_layers_config,
                    input_rois=input_rois,
                    input_weights=input_weights,
                    num_final_classes=num_final_classes,
                    box_rep=box_rep,

                    is_training=is_training,
                    end_points_collection=end_points_collection)

        elif fc_layers_type == 'fusion_fc_layers':
            fc_layers_config = layers_config.fusion_fc_layers

            cls_logits, offsets, end_points = \
                fusion_fc_layers.build(
                    fc_layers_config=fc_layers_config,
                    input_rois=input_rois,
                    input_weights=input_weights,
                    num_final_classes=num_final_classes,
                    box_rep=box_rep,

                    is_training=is_training,
                    end_points_collection=end_points_collection)

        elif fc_layers_type == 'fusion_fc_angle_cls_layers':
            fc_layers_config = layers_config.fusion_fc_angle_cls_layers

            cls_logits, offsets, angle_logits, end_points = \
                fusion_fc_angle_cl_layers.build(
                    fc_layers_config=fc_layers_config,
                    input_rois=input_rois,
                    input_weights=input_weights,
                    num_final_classes=num_final_classes,
                    box_rep=box_rep,
                    is_training=is_training,
                    end_points_collection=end_points_collection)

        elif fc_layers_type == 'separate_fc_layers':
            fc_layers_config = layers_config.separate_fc_layers

            cls_logits, offsets, end_points = \
                separate_fc_layers.build(
                    fc_layers_config=fc_layers_config,
                    input_rois=input_rois,
                    input_weights=input_weights,
                    num_final_classes=num_final_classes,
                    box_rep=box_rep,
                    is_training=is_training,
                    end_points_collection=end_points_collection,
                    img_keypoints=img_keypoints)

        else:
            raise ValueError('Invalid fc layers config')

    # # Histogram summaries
    # with tf.variable_scope('histograms_avod'):
    #     for fc_layer in end_points:
    #         tf.summary.histogram(fc_layer, end_points[fc_layer])

    fc_output_layers = dict()
    fc_output_layers[KEY_CLS_LOGITS] = cls_logits
    fc_output_layers[KEY_OFFSETS] = offsets
    fc_output_layers[KEY_ENDPOINTS] = end_points

    return fc_output_layers
