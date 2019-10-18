import tensorflow as tf
import math
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat
from transforms3d.euler import euler2mat, mat2euler, quat2euler, euler2quat


def tf_orientation_to_angle_vector(orientations_tensor):
    """ Converts orientation angles into angle unit vector representation.
        e.g. 45 -> [0.717, 0.717], 90 -> [0, 1]

    Args:
        orientations_tensor: A tensor of shape (N,) of orientation angles

    Returns:
        A tensor of shape (N, 2) of angle unit vectors in the format [x, y]
    """
    x = tf.cos(orientations_tensor)
    y = tf.sin(orientations_tensor)

    return tf.stack([x, y], axis=1)


def tf_orientation_to_angle_logits(orientations_tensor):  # shape=(?,)
    """ Converts orientation angles into angle logits representation.
        e.g. -0.198 -> [6.25e-05 6.25e-05 6.25e-05 6.25e-05 6.25e-05 6.25e-05 6.25e-05 0.999 6.25e-05 6.25e-05 6.25e-05 6.25e-05 6.25e-05 6.25e-05 6.25e-05 6.25e-05]

    Args:
        orientations_tensor: A tensor of shape (N,) of orientation angles

    Returns:
        A tensor of shape (N, 16) of angle logits in the format [6.25e-05, 6.25e-05, ..., 0.999, 6.25e-05, ..., 6.25e-05]
    """
    # orientations_tensor = tf.Print(orientations_tensor, ['to_angle_logits : orientations_tensor =', orientations_tensor], summarize=1000)
    pi = tf.constant([math.pi])
    orientation_length = tf.shape(orientations_tensor)[0]
    pies = tf.tile(pi, [orientation_length])

    # floor((orientations_tensor-(-pi))/(2pi/16))
    numerator = tf.math.add(orientations_tensor, pies)
    # numerator = tf.Print(numerator, ['to_angle_logits : numerator =', numerator], summarize=1000)
    denominator = tf.divide(pies, 8)
    # denominator = tf.Print(denominator, ['to_angle_logits : denominator =', denominator], summarize=1000)
    label_indices = tf.cast(tf.divide(numerator, denominator), tf.int32)
    # label_indices = tf.Print(label_indices, ['to_angle_logits : label_indices =', label_indices], summarize=1000)
    angle_logits = tf.one_hot(
        label_indices,
        depth=16,
        on_value=1.0 - 0.001,
        off_value=0.001 / 16)
    # angle_logits = tf.Print(angle_logits, ['to_angle_logits : angle_logits =', angle_logits], summarize=1000)
    return angle_logits  # shape=(?, 16)


def tf_angle_vector_to_orientation(angle_vectors_tensor):
    """ Converts angle unit vectors into orientation angle representation.
        e.g. [0.717, 0.717] -> 45, [0, 1] -> 90

    Args:
        angle_vectors_tensor: a tensor of shape (N, 2) of angle unit vectors
            in the format [x, y]

    Returns:
        A tensor of shape (N,) of orientation angles
    """
    x = angle_vectors_tensor[:, 0]
    y = angle_vectors_tensor[:, 1]

    return tf.atan2(y, x)


def tf_angle_quats_to_orientation(angle_vectors_tensor):
    """ Converts angle quaternion vectors into orientation angle representation.(only yaw angles)

    Args:
        angle_vectors_tensor: a tensor of shape (N, 4) of angle unit vectors
            in the format [w, x, y, z]

    Returns:
        A tensor of shape (N,) of orientation angles
    """

    norm = tf.sqrt(tf.reduce_sum(tf.square(angle_vectors_tensor), axis=1, keepdims=True))
    angle_vectors_tensor = angle_vectors_tensor / norm

    def quat_to_euler(quats):
        # Now the inputs are numpy arrays with the contents of the placeholder below
        yaws = []
        for j in range(quats.shape[0]):
            # Yes, very weird! When we tranlate from euler2quat, we use axes='rxyz', now we should use axes='ryzx'.
            # For annimation: https://quaternions.online/
            eulers = quat2euler(quats[j], axes='ryzx')
            yaw = eulers[0]  # pick the first index since we use y-z-x
            if yaws == []:
                yaws = np.array([yaw])
            else:
                yaws = np.append(yaws, yaw)
        # print('yaws = ', yaws)
        return yaws.astype(np.float32)
    yaw_angles = tf.py_func(quat_to_euler, [angle_vectors_tensor], tf.float32)
    yaw_angles.set_shape([None])  # shape=(?)
    return yaw_angles


def tf_angle_logits_to_orientation(angle_logits_tensor):
    """ Converts angle unit vectors into orientation angle representation.
        e.g. [0.1, 0.73, 0.03, ..., 0.01] -> -2.5525, because  (2*PI/16/2 + 1*2*PI/16 - PI) = -2.5525

    Args:
        angle_logits_tensor: a tensor of shape (N, 16) of angle logits
            in the format [theta1, theta2, ..., theta16]

    Returns:
        A tensor of shape (N,) of orientation angles
    """
    # angle_logits_tensor = tf.Print(angle_logits_tensor, ['logits_to_orientation : angle_logits_tensor =', angle_logits_tensor], summarize=1000)
    num_rows = tf.shape(angle_logits_tensor)[0]
    max_col = tf.argmax(angle_logits_tensor, axis=1)
    # max_col = tf.Print(max_col, ['logits_to_orientation : max_col =', max_col], summarize=1000)
    part1 = tf.constant([math.pi/16 - math.pi])
    parts1 = tf.tile(part1, [num_rows])
    part2 = tf.constant([math.pi/8])
    parts2 = tf.tile(part2, [num_rows])
    angles = tf.add(parts1, tf.multiply(tf.cast(max_col, tf.float32), parts2))
    # angles = tf.Print(angles, ['logits_to_orientation : angles =', angles], summarize=1000)

    return angles
