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
        if fc_layers_type == 'fusion_fc_angle_cls_layers':
            # Boxes with angle logits output
            losses_output = _build_cls_off_ang_logits_loss(model, prediction_dict)
        elif fc_layers_type == 'resnet_fc_layers':
            # Boxes with angle quaternion output
            losses_output = _build_ang_quat_loss(model, prediction_dict)
        else:
            # Boxes with angle vector output
            losses_output = _build_cls_off_ang_loss(model, prediction_dict)

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


def _get_off_ang_loss(model, offsets, offsets_gt,
                      angle_vectors, angle_vectors_gt,
                      cls_softmax, cls_gt):
    """Calculates the smooth L1 combined offset and angle loss, normalized by
        the number of positives

    Args:
        model: network model
        offsets: prediction offsets
        offsets_gt: ground truth offsets
        angle_vectors: prediction angle vectors
        angle_vectors_gt: ground truth angle vectors
        cls_softmax: prediction classification softmax scores
        cls_gt: classification ground truth one-hot vector

    Returns:
        final_reg_loss: combined offset and angle vector loss
        offset_loss_norm: normalized offset loss
        ang_loss_norm: normalized angle vector loss
    """

    weighted_smooth_l1_localization_loss = losses.WeightedSmoothL1Loss()

    reg_loss_weight = model._config.loss_config.reg_loss_weight
    ang_loss_weight = model._config.loss_config.ang_loss_weight

    anchorwise_localization_loss = weighted_smooth_l1_localization_loss(
        offsets, offsets_gt, weight=reg_loss_weight)
    anchorwise_orientation_loss = weighted_smooth_l1_localization_loss(
        angle_vectors, angle_vectors_gt, weight=ang_loss_weight)

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
    pos_localization_loss = tf.reduce_sum(tf.boolean_mask(
        anchorwise_localization_loss, positive_mask))
    pos_orientation_loss = tf.reduce_sum(tf.boolean_mask(
        anchorwise_orientation_loss, positive_mask))

    # Combine regression losses
    combined_reg_loss = pos_localization_loss + pos_orientation_loss

    with tf.variable_scope('reg_norm'):
        # Normalize by the number of positive/desired classes
        # only if we have any positives
        num_positives = tf.reduce_sum(pos_classification_floats)
        pos_div_cond = tf.not_equal(num_positives, 0)

        offset_loss_norm = tf.cond(
            pos_div_cond,
            lambda: pos_localization_loss / num_positives,
            lambda: tf.constant(0.0))

        ang_loss_norm = tf.cond(
            pos_div_cond,
            lambda: pos_orientation_loss / num_positives,
            lambda: tf.constant(0.0))

        final_reg_loss = tf.cond(
            pos_div_cond,
            lambda: combined_reg_loss / num_positives,
            lambda: tf.constant(0.0))

    # Add summary scalars
    if model._train_val_test == 'train':
        tf.summary.scalar('localization', offset_loss_norm)
        tf.summary.scalar('orientation', ang_loss_norm)
        tf.summary.scalar('regression_total', final_reg_loss)

        tf.summary.scalar('mb_num_positives', num_positives)

    return final_reg_loss, offset_loss_norm, ang_loss_norm


def _get_off_ang_logits_loss(model, offsets, offsets_gt,
                             angle_logits, angle_logits_gt,
                             cls_softmax, cls_gt):
    """Calculates the smooth L1 combined offset and angle loss, normalized by
        the number of positives

    Args:
        model: network model
        offsets: prediction offsets
        offsets_gt: ground truth offsets
        angle_logits: prediction angle logits            shape=(?,16)
        angle_logits_gt: ground truth angle logits       shape=(?,16)
        cls_softmax: prediction classification softmax scores
        cls_gt: classification ground truth one-hot vector

    Returns:
        final_reg_loss: combined offset and angle vector loss
        offset_loss_norm: normalized offset loss
        ang_loss_norm: normalized angle vector loss
    """

    weighted_smooth_l1_localization_loss = losses.WeightedSmoothL1Loss()
    weighted_softmax_loss = losses.WeightedSoftmaxLoss()

    reg_loss_weight = model._config.loss_config.reg_loss_weight
    ang_loss_weight = model._config.loss_config.ang_loss_weight

    anchorwise_localization_loss = weighted_smooth_l1_localization_loss(  # shape=(?,)
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
    pos_localization_loss = tf.reduce_sum(tf.boolean_mask(
        anchorwise_localization_loss, positive_mask))

    # Apply mask to angle_logits before we calculate final loss
    pos_angle_logits = tf.boolean_mask(
        angle_logits, positive_mask)
    pos_angle_logits_gt = tf.boolean_mask(
        angle_logits_gt, positive_mask)

    # pos_angle_logits = tf.Print(pos_angle_logits, ['^^^^^^^^^^^^^^^line 247(pplp loss) : pos_angle_logits =', pos_angle_logits], summarize=1000)
    # pos_angle_logits_gt = tf.Print(pos_angle_logits_gt, ['^^^^^^^^^^^^^^^line 248(pplp loss) : pos_angle_logits_gt =', pos_angle_logits_gt], summarize=1000)
    pos_orientation_loss = weighted_softmax_loss(
        pos_angle_logits, pos_angle_logits_gt, weight=ang_loss_weight)
    # pos_orientation_loss = tf.Print(pos_orientation_loss, ['^^^^^^^^^^^^^^^line 251(pplp loss) : pos_orientation_loss =', pos_orientation_loss], summarize=1000)

    # Combine regression losses
    combined_reg_loss = pos_localization_loss + pos_orientation_loss

    with tf.variable_scope('reg_norm'):
        # Normalize by the number of positive/desired classes
        # only if we have any positives
        num_positives = tf.reduce_sum(pos_classification_floats)
        pos_div_cond = tf.not_equal(num_positives, 0)

        offset_loss_norm = tf.cond(
            pos_div_cond,
            lambda: pos_localization_loss / num_positives,
            lambda: tf.constant(0.0))

        ang_loss_norm = tf.cond(
            pos_div_cond,
            lambda: pos_orientation_loss / tf.cast(tf.shape(pos_angle_logits_gt)[0], dtype=tf.float32),
            lambda: tf.constant(0.0))

        final_reg_loss = tf.cond(
            pos_div_cond,
            lambda: combined_reg_loss / num_positives,
            lambda: tf.constant(0.0))

    # Add summary scalars
    if model._train_val_test == 'train':
        tf.summary.scalar('localization', offset_loss_norm)
        tf.summary.scalar('orientation', ang_loss_norm)
        tf.summary.scalar('regression_total', final_reg_loss)

        tf.summary.scalar('mb_num_positives', num_positives)

    return final_reg_loss, offset_loss_norm, ang_loss_norm


def _get_off_ang_quat_loss(model, angle_quats, angle_quats_gt):
    """Calculates the smooth L1 combined offset and angle loss, normalized by
        the number of positives

    Args:
        model: network model
        angle_quats: prediction angle quaternions            shape=(?,4),  where ? is the number of all candidate anchors
        angle_quats_gt: ground truth angle quaternions       shape=(?,4),  where ? is the number of all candidate anchors

    Returns:
        ang_loss_norm: normalized angle vector loss
    """
    ang_loss_weight = model._config.loss_config.ang_loss_weight

    norm = tf.sqrt(tf.reduce_sum(tf.square(angle_quats), axis=1, keepdims=True))
    angle_quats = angle_quats / norm
    # angle_quats = tf.Print(angle_quats, ['line 305(orientnet loss) : angle_quats =', angle_quats], summarize=1000)
    # angle_quats_gt = tf.Print(angle_quats_gt, ['line 306(orientnet loss) : angle_quats_gt =', angle_quats_gt], summarize=1000)
    pos_orientation_loss = ops.loss_log_quaternion(angle_quats, angle_quats_gt, tf.shape(angle_quats)[0], use_logging=False)
    pos_orientation_loss = tf.multiply(pos_orientation_loss, ang_loss_weight)
    # pos_orientation_loss is already the mean loss!
    # pos_orientation_loss = tf.Print(pos_orientation_loss, ['line 307(orientnet loss) : pos_orientation_loss =', pos_orientation_loss], summarize=1000)

    with tf.variable_scope('reg_norm'):
        # Normalize by the number of positive/desired classes
        # only if we have any positives
        num_angles = tf.shape(angle_quats_gt)[0]
        pos_div_cond = tf.not_equal(num_angles, 0)

        ang_loss_norm = tf.cond(
            pos_div_cond,
            lambda: pos_orientation_loss,  # pos_orientation_loss is already the mean loss!
            lambda: tf.constant(0.0))

    # Add summary scalars
    if model._train_val_test == 'train':
        tf.summary.scalar('orientation', ang_loss_norm)
        # ang_loss_norm = tf.Print(ang_loss_norm, ['line 323(orientnet loss) : ang_loss_norm =', ang_loss_norm], summarize=1000)
    return ang_loss_norm


def _build_cls_off_ang_loss(model, prediction_dict):
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
    mb_angle_vectors = prediction_dict[model.PRED_MB_ANGLE_VECTORS]

    # Ground Truth
    mb_cls_gt = prediction_dict[model.PRED_MB_CLASSIFICATIONS_GT]
    mb_offsets_gt = prediction_dict[model.PRED_MB_OFFSETS_GT]
    mb_orientations_gt = prediction_dict[model.PRED_MB_ORIENTATIONS_GT]

    # mb_cls_logits = tf.Print(mb_cls_logits, ['line 284(pplp) : mb_cls_logits =', mb_cls_logits], summarize=100)
    # mb_cls_softmax = tf.Print(mb_cls_softmax, ['line 285(pplp) : mb_cls_softmax =', mb_cls_softmax], summarize=100)
    # mb_offsets = tf.Print(mb_offsets, ['line 286(pplp) : mb_offsets =', mb_offsets], summarize=100)
    # mb_angle_vectors = tf.Print(mb_angle_vectors, ['line 287(pplp) : mb_angle_vectors =', mb_angle_vectors], summarize=100)
    # mb_cls_gt = tf.Print(mb_cls_gt, ['line 288(pplp) : mb_cls_gt =', mb_cls_gt], summarize=100)
    # mb_offsets_gt = tf.Print(mb_offsets_gt, ['line 289(pplp) : mb_offsets_gt =', mb_offsets_gt], summarize=100)
    # mb_orientations_gt = tf.Print(mb_orientations_gt, ['line 290(pplp) : mb_orientations_gt =', mb_orientations_gt], summarize=100)
    # Decode ground truth orientations
    with tf.variable_scope('avod_gt_angle_vectors'):
        mb_angle_vectors_gt = \
            orientation_encoder.tf_orientation_to_angle_vector(
                mb_orientations_gt)

    # Losses
    with tf.variable_scope('pplp_losses'):
        with tf.variable_scope('classification'):
            # mb_cls_logits = tf.Print(mb_cls_logits, ['line 305(pplp loss) : mb_cls_logits =', mb_cls_logits], summarize=1000)
            cls_loss = _get_cls_loss(model, mb_cls_logits, mb_cls_gt)

        with tf.variable_scope('regression'):
            # mb_angle_vectors_gt = tf.Print(mb_angle_vectors_gt, ['^^^^^^^^^^^^^^^^^^^^line 308(pplp loss) : mb_angle_vectors_gt =', mb_angle_vectors_gt], summarize=1000)
            # mb_angle_vectors_gt = tf.Print(mb_angle_vectors_gt, ['^^^^^^^^^^^^^^^^^^^^line 309(pplp loss) : tf.shape(mb_angle_vectors_gt) =', tf.shape(mb_angle_vectors_gt)], summarize=1000)
            # mb_angle_vectors = tf.Print(mb_angle_vectors, ['line 310(pplp loss) : mb_angle_vectors =', mb_angle_vectors], summarize=1000)
            # mb_angle_vectors = tf.Print(mb_angle_vectors, ['line 311(pplp loss) : tf.shape(mb_angle_vectors) =', tf.shape(mb_angle_vectors)], summarize=1000)
            final_reg_loss, offset_loss_norm, ang_loss_norm = _get_off_ang_loss(
                model, mb_offsets, mb_offsets_gt,
                mb_angle_vectors, mb_angle_vectors_gt,
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
    losses_output[KEY_ANG_LOSS_NORM] = ang_loss_norm

    return losses_output


def _build_cls_off_ang_logits_loss(model, prediction_dict):
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
    # Here, we divided the 360 degree into 16 classes, so the angle_vectors is a 16 colunms matrix.
    mb_angle_logits = prediction_dict[model.PRED_MB_ANGLE_VECTORS]

    # Ground Truth
    mb_cls_gt = prediction_dict[model.PRED_MB_CLASSIFICATIONS_GT]
    mb_offsets_gt = prediction_dict[model.PRED_MB_OFFSETS_GT]
    mb_orientations_gt = prediction_dict[model.PRED_MB_ORIENTATIONS_GT]

    # mb_cls_logits = tf.Print(mb_cls_logits, ['line 284(pplp) : mb_cls_logits =', mb_cls_logits], summarize=100)
    # mb_cls_softmax = tf.Print(mb_cls_softmax, ['line 285(pplp) : mb_cls_softmax =', mb_cls_softmax], summarize=100)
    # mb_offsets = tf.Print(mb_offsets, ['line 286(pplp) : mb_offsets =', mb_offsets], summarize=100)
    # mb_angle_logits = tf.Print(mb_angle_logits, ['line 287(pplp) : mb_angle_logits =', mb_angle_logits], summarize=100)
    # mb_cls_gt = tf.Print(mb_cls_gt, ['line 288(pplp) : mb_cls_gt =', mb_cls_gt], summarize=100)
    # mb_offsets_gt = tf.Print(mb_offsets_gt, ['line 289(pplp) : mb_offsets_gt =', mb_offsets_gt], summarize=100)
    # mb_orientations_gt = tf.Print(mb_orientations_gt, ['line 463(pplp loss) : mb_orientations_gt =', mb_orientations_gt], summarize=100)
    # Decode ground truth orientations
    with tf.variable_scope('avod_gt_angle_vectors'):
        mb_angle_logits_gt = \
            orientation_encoder.tf_orientation_to_angle_logits(
                mb_orientations_gt)

    # Losses
    with tf.variable_scope('pplp_losses'):
        with tf.variable_scope('classification'):
            # mb_cls_logits = tf.Print(mb_cls_logits, ['line 305(pplp loss) : mb_cls_logits =', mb_cls_logits], summarize=1000)
            cls_loss = _get_cls_loss(model, mb_cls_logits, mb_cls_gt)

        with tf.variable_scope('regression'):
            final_reg_loss, offset_loss_norm, ang_loss_norm = _get_off_ang_logits_loss(
                model, mb_offsets, mb_offsets_gt,
                mb_angle_logits, mb_angle_logits_gt,
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
    losses_output[KEY_ANG_LOSS_NORM] = ang_loss_norm

    return losses_output


def _build_ang_quat_loss(model, prediction_dict):
    """Builds classification, offset, and angle vector losses.

    Args:
        model: network model
        prediction_dict: prediction dictionary. PRED_MB_ANGLE_VECTORS are in quaternion form.

    Returns:
        losses_output: losses dictionary
    """

    # Minibatch Predictions
    mb_angle_quats = prediction_dict[model.PRED_ANGLE_VECTORS]

    # Ground Truth
    mb_angle_quats_gt = prediction_dict[model.PRED_ORIENTATIONS_QUATS_GT]

    mb_valid_mask = prediction_dict[model.PRED_VALID_MASK]

    mb_angle_quats = tf.boolean_mask(mb_angle_quats, mb_valid_mask)
    mb_angle_quats_gt = tf.boolean_mask(mb_angle_quats_gt, mb_valid_mask)
    # Losses
    with tf.variable_scope('orient_losses'):
        ang_loss_norm = _get_off_ang_quat_loss(
            model, mb_angle_quats, mb_angle_quats_gt)
        tf.summary.scalar('ang_loss_norm', ang_loss_norm)

    # Loss dictionary
    losses_output = dict()

    losses_output[KEY_ANG_LOSS_NORM] = ang_loss_norm

    return losses_output
