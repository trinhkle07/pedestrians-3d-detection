import numpy as np

from wavedata.tools.core.integral_image import IntegralImage
from wavedata.tools.core.integral_image_2d import IntegralImage2D

from pplp.core import format_checker


def get_empty_anchor_filter(anchors, voxel_grid_3d, density_threshold=1):
    """ Returns a filter for empty boxes from the given 3D anchor list

    Args:
        anchors: list of 3d anchors in the format
            N x [x, y, z, dim_x, dim_y, dim_z]
        voxel_grid_3d: a VoxelGrid object containing a 3D voxel grid of
            pointcloud used to filter the anchors
        density_threshold: minimum number of points in voxel to keep the anchor

    Returns:
        anchor filter: N Boolean mask
    """
    format_checker.check_anchor_format(anchors)

    # Get Integral image of the voxel, add 1 since filled = 0, empty is -1
    integral_image = IntegralImage(voxel_grid_3d.leaf_layout + 1)

    # Make cuboid container
    cuboid_container = np.zeros([len(anchors), 6]).astype(np.uint32)

    top_left_up = np.zeros([len(anchors), 3]).astype(np.float32)
    bot_right_down = np.zeros([len(anchors), 3]).astype(np.float32)

    # Calculate minimum corner
    top_left_up[:, 0] = anchors[:, 0] - (anchors[:, 3] / 2.)
    top_left_up[:, 1] = anchors[:, 1] - (anchors[:, 4])
    top_left_up[:, 2] = anchors[:, 2] - (anchors[:, 5] / 2.)

    # Calculate maximum corner
    bot_right_down[:, 0] = anchors[:, 0] + (anchors[:, 3] / 2.)
    bot_right_down[:, 1] = anchors[:, 1]
    bot_right_down[:, 2] = anchors[:, 2] + (anchors[:, 5] / 2.)

    # map_to_index() expects N x 3 points
    cuboid_container[:, :3] = voxel_grid_3d.map_to_index(
        top_left_up)
    cuboid_container[:, 3:] = voxel_grid_3d.map_to_index(
        bot_right_down)

    # Transpose to pass into query()
    cuboid_container = cuboid_container.T

    # Get point density score for each cuboid
    point_density_score = integral_image.query(cuboid_container)

    # Create the filter
    anchor_filter = point_density_score >= density_threshold

    # Flatten into shape (N,)
    anchor_filter = anchor_filter.flatten()

    return anchor_filter


def get_empty_anchor_filter_2d(anchors, voxel_grid_2d, density_threshold=1):
    """ Returns a filter for empty anchors from the given 2D anchor list

    Args:
        anchors: list of 3d anchors in the format
            N x [x, y, z, dim_x, dim_y, dim_z]
        voxel_grid_2d: a VoxelGrid object containing a 2D voxel grid of
            point cloud used to filter the anchors
        density_threshold: minimum number of points in voxel to keep the anchor

    Returns:
        anchor filter: N Boolean mask
    """
    format_checker.check_anchor_format(anchors)

    # Remove y dimensions from anchors to project into BEV
    anchors_2d = anchors[:, [0, 2, 3, 5]]

    # Get Integral image of the voxel, add 1 since filled = 0, empty is -1
    leaf_layout = voxel_grid_2d.leaf_layout_2d + 1
    leaf_layout = np.squeeze(leaf_layout)
    integral_image = IntegralImage2D(leaf_layout)

    # Make anchor container
    anchor_container = np.zeros([len(anchors_2d), 4]).astype(np.uint32)

    num_anchors = len(anchors_2d)

    # Set up objects containing corners of anchors
    top_left_up = np.zeros([num_anchors, 2]).astype(np.float32)
    bot_right_down = np.zeros([num_anchors, 2]).astype(np.float32)

    # Calculate minimum corner
    top_left_up[:, 0] = anchors_2d[:, 0] - (anchors_2d[:, 2] / 2.)
    top_left_up[:, 1] = anchors_2d[:, 1] - (anchors_2d[:, 3] / 2.)

    # Calculate maximum corner
    bot_right_down[:, 0] = anchors_2d[:, 0] + (anchors_2d[:, 2] / 2.)
    bot_right_down[:, 1] = anchors_2d[:, 1] + (anchors_2d[:, 3] / 2.)

    # map_to_index() expects N x 2 points
    anchor_container[:, :2] = voxel_grid_2d.map_to_index(
        top_left_up)
    anchor_container[:, 2:] = voxel_grid_2d.map_to_index(
        bot_right_down)

    # Transpose to pass into query()
    anchor_container = anchor_container.T

    # Get point density score for each anchor
    point_density_score = integral_image.query(anchor_container)

    # Create the filter
    anchor_filter = point_density_score >= density_threshold

    return anchor_filter


def get_occupancy_map_index(points, area_extents, map_shape):
    """ For points in camera x and z axis, returns the indices of them on the
        occupancy grid map.
        For the points that beyongs the grid map width or height, move then into
        the width and height range.

    Args:
        points: Nx2. [[x, z], [x, z], ... [x, z], [x, z]]
        area_extents: A 3x2 array that describes the 3D area of interest.
            [[xmin, xmax], [ymin, ymax], [zmin, zmax]] in camera coordinate system.
        map_shape: numpy.shape = (1, height, width)

    Returns:
        map_indices: Nx2, numpy array interger. [[x, z], [x, z], ... [x, z], [x, z]]
            But here, the origin is the left top corner.
    """
    origin_coordinates = np.ones(points.shape)  # default dtype=float64
    origin_coordinates[:, 0] = origin_coordinates[:, 0] * area_extents[0][0]
    origin_coordinates[:, 1] = origin_coordinates[:, 1] * area_extents[2][1]
    height = map_shape[1]
    width = map_shape[2]
    height_resolution = (area_extents[2][1]-area_extents[2][0]) / height
    width_resolution = (area_extents[0][1]-area_extents[0][0]) / width
    points_relative_x = points[:, 0] - origin_coordinates[:, 0]
    points_relative_z = origin_coordinates[:, 1] - points[:, 1]
    points_grid_x = np.array(points_relative_x/width_resolution, dtype=int)
    points_grid_z = np.array(points_relative_z/height_resolution, dtype=int)
    points_grid_x[np.where(points_grid_x < 0)] = 0
    points_grid_x[np.where(points_grid_x > (width-1))] = (width-1)
    points_grid_z[np.where(points_grid_z < 0)] = 0
    points_grid_z[np.where(points_grid_z > (height-1))] = (height-1)
    map_indices = np.stack((points_grid_x, points_grid_z), axis=1)

    return map_indices


def get_empty_anchor_filter_occupancy(anchors, occupancy_maps, area_extents):
    """ Returns a filter for empty anchors from the given 2D anchor list

    Args:
        anchors: list of 3d anchors in the format
            N x [x, y, z, dim_x, dim_y, dim_z]
        occupancy_maps: a Bird's Eye View image containing occupancy map used
            to filter the anchors. Unknow area is -1, free area is 0, occupied
            area is 1.
        area_extents:a 3x2 array that describes the 3D area of interest.
            [[xmin, xmax], [ymin, ymax], [zmin, zmax]] in camera coordinate system.

    Returns:
        anchor filter: N Boolean mask
    """
    format_checker.check_anchor_format(anchors)

    # Remove y dimensions from anchors to project into BEV
    anchors_2d = anchors[:, [0, 2, 3, 5]]  # anchors_2d = N x [x, z, dim_x, dim_z]
    # print('anchors_2d = ', anchors_2d)
    # print('occupancy_maps = ', occupancy_maps)
    # print('area_extents = ', area_extents)

    # Get Integral image of the voxel, add 1 since filled = 0, empty is -1
    # leaf_layout = occupancy_maps.leaf_layout_2d + 1
    # leaf_layout = np.squeeze(leaf_layout)
    # integral_image = IntegralImage2D(leaf_layout)

    # Make anchor container
    num_anchors = len(anchors_2d)
    anchor_container = np.zeros([num_anchors, 4]).astype(np.int)

    # Set up objects containing corners of anchors
    top_left_up = np.zeros([num_anchors, 2]).astype(np.float32)
    bot_right_down = np.zeros([num_anchors, 2]).astype(np.float32)

    # Calculate minimum corner
    top_left_up[:, 0] = anchors_2d[:, 0] - (anchors_2d[:, 2] / 2.)
    top_left_up[:, 1] = anchors_2d[:, 1] + (anchors_2d[:, 3] / 2.)

    # Calculate maximum corner
    bot_right_down[:, 0] = anchors_2d[:, 0] + (anchors_2d[:, 2] / 2.)
    bot_right_down[:, 1] = anchors_2d[:, 1] - (anchors_2d[:, 3] / 2.)

    # map_to_index() expects N x 2 points
    # For the points that beyongs the grid map width or height, move then into
    # the width and height range.
    occupancy_maps = np.array(occupancy_maps)
    anchor_container[:, :2] = get_occupancy_map_index(top_left_up, area_extents, occupancy_maps.shape)
    anchor_container[:, 2:] = get_occupancy_map_index(bot_right_down, area_extents, occupancy_maps.shape)

    anchor_filter = [True] * num_anchors
    # For each anchor, check if there's any non-free occupancy grid inside that anchor.
    # If all the grid inside that crop is free area, then set the filter to False.
    # Otherwise set it to True.
    for i in range(num_anchors):
        crop = occupancy_maps[0, anchor_container[i][1]:anchor_container[i][3], anchor_container[i][0]:anchor_container[i][2]]
        nonzero_num = np.count_nonzero(crop)
        if nonzero_num == 0:
            anchor_filter[i] = False

    # print('anchor_filter = ', anchor_filter)
    # print('len(anchor_filter) = ', len(anchor_filter))
    return anchor_filter


def get_iou_filter(iou_list, iou_range):
    """Returns a boolean filter array that is the output of a given IoU range

    Args:
        iou_list: A numpy array with a list of IoU values
        iou_range: A list of [lower_bound, higher_bound] for IoU range

    Returns:
        iou_filter: A numpy array of booleans that filters for valid range
    """
    # Get bounds
    lower_bound = iou_range[0]
    higher_bound = iou_range[1]

    min_valid_list = lower_bound < iou_list
    max_valid_list = iou_list < higher_bound

    # Get filter for values in between
    iou_filter = np.logical_and(min_valid_list, max_valid_list)

    return iou_filter
