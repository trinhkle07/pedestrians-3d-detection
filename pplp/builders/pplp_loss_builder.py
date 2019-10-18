import tensorflow as tf

from pplp.core import losses
from pplp.core import orientation_encoder
from pplp.core.pplp_fc_layers import ops

KEY_CLASSIFICATION_LOSS = 'classification_loss'
KEY_REGRESSION_LOSS = 'regression_loss'
KEY_PPLP_LOSS = 'pplp_loss'
KEY_OFFSET_LOSS_NORM = 'offset_loss_norm'
KEY_ANG_LOSS_NORM = 'ang_loss_norm'


def build(model, prediction_dict):
    """Builds the loss for a variety of box representations

    Args:
        model: network model
        prediction_dict: prediction dictionary

    Returns:
        losses_output: loss dictionary
    """

    avod_box_rep = model._config.avod_config.avod_box_representation
    fc_layers_type = model._config.layers_config.avod_config.WhichOneof('fc_layers')

    if avod_box_rep in ['box_3d', 'box_4ca']:
        if fc_layers_type == 'fusion_fc_layers':
            # Boxes with classification and offset output
            losses_output = _build_cls_off_loss(model, prediction_dict)

    elif avod_box_rep in ['box_8c', 'box_8co', 'box_4c']:
        losses_output = _build_cls_off_loss(model, prediction_dict)

    else:
        raise ValueError('Invalid box representation', avod_box_rep)

    return losses_output


def _get_cls_loss(model, cls_logits, cls_gt):
    """Calculates cross entropy loss for classification

    Args:
        model: network model
        cls_logits: predicted classification logits
        cls_gt: ground truth one-hot classification vector

    Returns:
        cls_loss: cross-entropy classification loss
    """

    # Cross-entropy loss for classification
    weighted_softmax_classification_loss = \
        losses.WeightedSoftmaxLoss()
    cls_loss_weight = model._config.loss_config.cls_loss_weight
    cls_loss = weighted_softmax_classification_loss(
        cls_logits, cls_gt, weight=cls_loss_weight)

    # Normalize by the size of the minibatch
    with tf.variable_scope('cls_norm'):
        cls_loss = cls_loss / tf.cast(
            tf.shape(cls_gt)[0], dtype=tf.float32)

    # Add summary scalar during training
    if model._train_val_test == 'train':
        tf.summary.scalar('classification', cls_loss)

    return cls_loss


def _get_positive_mask(positive_selection, cls_softmax, cls_gt):
    """Gets the positive mask based on the ground truth box classifications

    Args:
        positive_selection: positive selection method
            (e.g. 'corr_cls', 'not_bkg')
        cls_softmax: prediction classification softmax scores
        cls_gt: ground truth classification one-hot vector

    Returns:
        positive_mask: positive mask
    """

    # Get argmax for predicted class
    classification_argmax = tf.argmax(cls_softmax, axis=1)

    # Get the ground truth class indices back from one_hot vector
    class_indices_gt = tf.argmax(cls_gt, axis=1)
    # class_indices_gt = tf.Print(class_indices_gt, ['^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^line 88(pplp loss) : class_indices_gt =', class_indices_gt], summarize=1000)

    # Mask for which predictions are not background
    not_background_mask = tf.greater(class_indices_gt, 0)

    # Combine the masks
    if positive_selection == 'corr_cls':
        # Which prediction classifications match ground truth
        correct_classifications_mask = tf.equal(
            classification_argmax, class_indices_gt)
        positive_mask = tf.logical_and(
            correct_classifications_mask, not_background_mask)
    elif positive_selection == 'not_bkg':
        positive_mask = not_background_mask
    else:
        raise ValueError('Invalid positive selection', positive_selection)

    return positive_mask


def _get_off_loss(model, offsets, offsets_gt,
                  cls_softmax, cls_gt):
    """Calculates the smooth L1 combined offset and angle loss, normalized by
        the number of positives

    Args:
        model: network model
        offsets: prediction offsets
        offsets_gt: ground truth offsets
        cls_softmax: prediction classification softmax scores
        cls_gt: classification ground truth one-hot vector

    Returns:
        final_reg_loss: combined offset and angle vector loss
        offset_loss_norm: normalized offset loss
    """

    # weighted_smooth_l1_localization_loss = losses.WeightedSmoothL1Loss()
    weighted_smooth_l1_localization_loss_2output = losses.WeightedSmoothL1Loss_2ouput()
    # weighted_softmax_loss = losses.WeightedSoftmaxLoss()

    reg_loss_weight = model._config.loss_config.reg_loss_weight

    # anchorwise_localization_loss = weighted_smooth_l1_localization_loss(  # shape=(?,), ? is the number of anchor candidates.
    #     offsets, offsets_gt, weight=reg_loss_weight)
    anchorwise_localization_loss, elementwise_localization_loss = weighted_smooth_l1_localization_loss_2output(  # shape=(?,), ? is the number of anchor candidates.
        offsets, offsets_gt, weight=reg_loss_weight)

    # cls_gt = tf.Print(cls_gt, ['^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^line 144(pplp loss) : model._positive_selection =', model._positive_selection], summarize=1000)
    # cls_softmax = tf.Print(cls_softmax, ['^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^line 145(pplp loss) : cls_softmax =', cls_softmax], summarize=1000)
    # cls_gt = tf.Print(cls_gt, ['^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^line 146(pplp loss) : cls_gt =', cls_gt], summarize=1000)
    positive_mask = _get_positive_mask(model._positive_selection,
                                       cls_softmax, cls_gt)
    # positive_mask = tf.Print(positive_mask, ['^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^line 149(pplp loss) : positive_mask =', positive_mask], summarize=1000)

    # Cast to float to get number of positives
    pos_classification_floats = tf.cast(
        positive_mask, tf.float32)

    # Apply mask to only keep regression loss for positive predictions
    # anchorwise_localization_loss = tf.Print(anchorwise_localization_loss, ['^^^^^^^^^^^^^^line 151(pplp loss) : anchorwise_localization_loss.shape =', tf.shape(anchorwise_localization_loss)], summarize=1000)
    pos_localization_loss = tf.reduce_sum(tf.boolean_mask(
        anchorwise_localization_loss, positive_mask))
    # pos_localization_loss = tf.Print(pos_localization_loss, ['^^^^^^^^^^^^^^line 154(pplp loss) : pos_localization_loss.shape =', tf.shape(pos_localization_loss)], summarize=1000)

    # elementwise_localization_loss = tf.Print(elementwise_localization_loss, ['^^^^^^^^^^^^^^line 155(pplp loss) : elementwise_localization_loss.shape =', tf.shape(elementwise_localization_loss)], summarize=1000)
    valid_elementwise_localization_loss = tf.boolean_mask(
        elementwise_localization_loss, positive_mask, axis=0)
    # valid_elementwise_localization_loss = tf.Print(valid_elementwise_localization_loss, ['^^^^^^^^^^^^^^line 157(pplp loss) : valid_elementwise_localization_loss.shape =', tf.shape(valid_elementwise_localization_loss)], summarize=1000)
    pos_localization_loss_elementwise = tf.reduce_sum(valid_elementwise_localization_loss, axis=0)
    # pos_localization_loss_elementwise = tf.reduce_sum(tf.boolean_mask(
    #     elementwise_localization_loss, positive_mask, axis=0), axis=1)
    # pos_localization_loss_elementwise = tf.Print(pos_localization_loss_elementwise, ['^^^^^^^^^^^^^^line 156(pplp loss) : pos_localization_loss_elementwise =', pos_localization_loss_elementwise], summarize=1000)
    # pos_localization_loss_elementwise = tf.Print(pos_localization_loss_elementwise, ['^^^^^^^^^^^^^^line 162(pplp loss) : pos_localization_loss_elementwise.shape =', tf.shape(pos_localization_loss_elementwise)], summarize=1000)

    # Combine regression losses
    combined_reg_loss = pos_localization_loss

    with tf.variable_scope('reg_norm'):
        # Normalize by the number of positive/desired classes
        # only if we have any positives
        num_positives = tf.reduce_sum(pos_classification_floats)
        pos_div_cond = tf.not_equal(num_positives, 0)

        offset_loss_norm = tf.cond(
            pos_div_cond,
            lambda: pos_localization_loss / num_positives,
            lambda: tf.constant(0.0))

        offset_loss_norm_elewise = tf.cond(
            pos_div_cond,
            lambda: tf.divide(pos_localization_loss_elementwise, num_positives),
            lambda: tf.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        offset_x1, offset_y1, offset_x2, offset_y2, offset_x3, offset_y3, offset_x4, offset_y4, offset_h1, offset_h2 = tf.split(offset_loss_norm_elewise, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], axis=0)

        final_reg_loss = tf.cond(
            pos_div_cond,
            lambda: combined_reg_loss / num_positives,
            lambda: tf.constant(0.0))

    # Add summary scalars
    if model._train_val_test == 'train':
        tf.summary.scalar('localization', offset_loss_norm)
        tf.summary.scalar('offset_x1', offset_x1[0])
        tf.summary.scalar('offset_y1', offset_y1[0])
        tf.summary.scalar('offset_x2', offset_x2[0])
        tf.summary.scalar('offset_y2', offset_y2[0])
        tf.summary.scalar('offset_x3', offset_x3[0])
        tf.summary.scalar('offset_y3', offset_y3[0])
        tf.summary.scalar('offset_x4', offset_x4[0])
        tf.summary.scalar('offset_y4', offset_y4[0])
        tf.summary.scalar('offset_h1', offset_h1[0])
        tf.summary.scalar('offset_h2', offset_h2[0])
        tf.summary.scalar('regression_total', final_reg_loss)

        tf.summary.scalar('mb_num_positives', num_positives)

    return final_reg_loss, offset_loss_norm


def _build_cls_off_loss(model, prediction_dict):
    """Builds classification, offset, and angle vector losses.

    Args:
        model: network model
        prediction_dict: prediction dictionary

    Returns:
        losses_output: losses dictionary
    """

    # Minibatch Predictions
    mb_cls_logits = prediction_dict[model.PRED_MB_CLASSIFICATION_LOGITS]
    mb_cls_softmax = prediction_dict[model.PRED_MB_CLASSIFICATION_SOFTMAX]
    mb_offsets = prediction_dict[model.PRED_MB_OFFSETS]

    # Ground Truth
    mb_cls_gt = prediction_dict[model.PRED_MB_CLASSIFICATIONS_GT]
    mb_offsets_gt = prediction_dict[model.PRED_MB_OFFSETS_GT]

    # Losses
    with tf.variable_scope('pplp_losses'):
        with tf.variable_scope('classification'):
            # mb_cls_logits = tf.Print(mb_cls_logits, ['line 305(pplp loss) : mb_cls_logits =', mb_cls_logits], summarize=1000)
            cls_loss = _get_cls_loss(model, mb_cls_logits, mb_cls_gt)

        with tf.variable_scope('regression'):
            final_reg_loss, offset_loss_norm = _get_off_loss(
                model, mb_offsets, mb_offsets_gt,
                mb_cls_softmax, mb_cls_gt)

        with tf.variable_scope('pplp_loss'):
            pplp_loss = cls_loss + final_reg_loss
            # pplp_loss = tf.Print(pplp_loss, ['line 320(pplp loss) : pplp_loss =', pplp_loss], summarize=1000)
            tf.summary.scalar('pplp_loss', pplp_loss)

    # Loss dictionary
    losses_output = dict()

    losses_output[KEY_CLASSIFICATION_LOSS] = cls_loss
    losses_output[KEY_REGRESSION_LOSS] = final_reg_loss
    losses_output[KEY_PPLP_LOSS] = pplp_loss

    # Separate losses for plotting
    losses_output[KEY_OFFSET_LOSS_NORM] = offset_loss_norm

    return losses_output
