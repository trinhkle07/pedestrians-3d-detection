import math

import numpy as np
from wavedata.tools.core.voxel_grid_2d import VoxelGrid2D

from pplp.core.bev_generators import bev_panoptic_generator
from pplp.core.bev_generators import bresenham_line

# Set True for Zed Cam, False for Panoptic Kinect.
is_zed_cam = False

if is_zed_cam:
    # Zed Cam
    K_color = np.array([
        [700.126, 0, 623.651],
        [0, 700.126, 539.6931883],
        [0, 0, 1]
    ], dtype=np.double)

    M_depth_2_color = np.array([[-0.0245, 0.9996, -0.0131, -0.042],
                                [0.04, -0.0122, -0.9991, 0.06],
                                [-0.9989, -0.025, -0.0397, -0.04],
                                [0., 0., 0., 1.]], dtype=np.double)
    color_width = 1280

else:
    # Panoptic Kinect
    K_color = np.array([
        [1057.538006, 0, 948.5231183],
        [0, 1057.538006, 539.6931883],
        [0, 0, 1]
    ], dtype=np.double)

    M_depth_2_color = np.array([[-0.9999544223, 0.009513874938, -0.0007996463422, -0.05283642437],
                                [-0.009514055211, -0.9999547157, 0.00022194079, 1.098798979e-05],
                                [-0.0007974986139, 0.0002295385539, 0.9999996557, -2.539374782e-05],
                                [0, 0, 0, 1]], dtype=np.double)
    color_width = 1920

R_depth_2_color = np.matrix(M_depth_2_color[0:3, 0:3])
T_depth_2_color = M_depth_2_color[0:3, 3].reshape((3, 1))
depth_cam_coord = np.array(R_depth_2_color * np.array([[0], [0], [0]]) + T_depth_2_color)

fx = K_color[0, 0]
fy = K_color[1, 1]
fovx = 2 * math.atan(color_width / (2 * fx)) # Compute field of view

# viz_idx = 0

class BevSlices(bev_panoptic_generator.BevGenerator):

    NORM_VALUES = {
        'lidar': np.log(16),
    }

    def __init__(self, config, panoptic_utils):
        """BEV maps created using slices of the point cloud.

        Args:
            config: bev_panoptic_generator protobuf config
            panoptic_utils: PanopticUtils object
        """

        # Parse config
        self.height_lo = config.height_lo
        self.height_hi = config.height_hi
        self.num_slices = config.num_slices

        self.panoptic_utils = panoptic_utils

        # Pre-calculated values
        self.height_per_division = \
            (self.height_hi - self.height_lo) / self.num_slices

    def filter_bev_pointclouds(self,
                               source,
                               point_cloud,
                               ground_plane,
                               area_extents,
                               voxel_size):
        """Filter the whole pointclouds and only keep those can be seen within
         the BEV slices.

        Args:
            source: point cloud source
            point_cloud: point cloud (3, N), should be in camera coordinate
                         frame already.
            ground_plane: ground plane coefficients
            area_extents: 3D area extents
                [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
            voxel_size: voxel size in m

        Returns:
            BEV maps dictionary
                height_maps: list of height maps
                density_map: density map
        """

        all_points = np.transpose(point_cloud)

        height_lo = self.height_lo
        height_hi = self.height_hi
        # print('height_lo = ', height_lo)
        # print('height_hi = ', height_hi)
        slice_filter = self.panoptic_utils.create_slice_filter(
            point_cloud,
            area_extents,
            ground_plane,
            height_lo,
            height_hi)

        # Apply slice filter
        slice_points = all_points[slice_filter]
        return slice_points

    def generate_bev(self,
                     source,
                     point_cloud,
                     ground_plane,
                     area_extents,
                     voxel_size):
        """Generates the BEV maps dictionary. One height map is created for
        each slice of the point cloud. One density map is created for
        the whole point cloud.

        Args:
            source: point cloud source
            point_cloud: point cloud (3, N)
            ground_plane: ground plane coefficients
            area_extents: 3D area extents
                [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
            voxel_size: voxel size in m

        Returns:
            BEV maps dictionary
                height_maps: list of height maps
                density_map: density map
        """
        all_points = np.transpose(point_cloud)

        occupancy_maps = []
        height_maps = []

        depth_cam_voxel_coord = np.floor(depth_cam_coord / voxel_size).astype(np.int32).reshape(-1)

        # Generate height map for each slice
        for slice_idx in range(self.num_slices): # pplp_pedestrian_panoptic.config: num_slices = 1
            height_lo = self.height_lo + slice_idx * self.height_per_division
            height_hi = height_lo + self.height_per_division
            # Get all point cloud within 3D areas and heights (like 3D box)
            slice_filter = self.panoptic_utils.create_slice_filter(
                point_cloud,
                area_extents, # area_extents: [-3.99, 3.99, -5.0, 3.0, 0.0, 6.995]
                ground_plane,
                height_lo, # -5.00 (in y direction)
                height_hi) # 2.00

            # Apply slice filter
            slice_points = all_points[slice_filter]

            # print('bev_panoptic_slices.py :: slice_points = ', slice_points)
            # Create Voxel Grid 2D
            voxel_grid_2d = VoxelGrid2D()
            voxel_grid_2d.voxelize_2d(
                slice_points, voxel_size, # voxel_size: 0.01
                extents=area_extents,
                ground_plane=ground_plane,
                create_leaf_layout=False)

            # Remove y values (all 0)
            # voxel_indices all positive
            voxel_indices = voxel_grid_2d.voxel_indices[:, [0, 2]] # voxel_indices: unique indexes - only 1 point is chosen to be occupied in one voxel

            # Bring cam to new origin
            depth_cam_index = (depth_cam_voxel_coord - voxel_grid_2d.min_voxel_coord).astype(int)
            color_cam_index = (np.array([0, 0, 0]) / voxel_size - voxel_grid_2d.min_voxel_coord).astype(int)

            # Create empty BEV images
            height_map = np.zeros((voxel_grid_2d.num_divisions[0], voxel_grid_2d.num_divisions[2]))

            # Only update pixels where voxels have max height values,
            # and normalize by height of slices
            voxel_grid_2d.heights = voxel_grid_2d.heights - height_lo
            height_map[voxel_indices[:, 0], voxel_indices[:, 1]] = \
                np.asarray(voxel_grid_2d.heights) / self.height_per_division

            height_maps.append(height_map)

            # Create empty BEV occupancy map
            occupancy_map = np.zeros((voxel_grid_2d.num_divisions[0], voxel_grid_2d.num_divisions[2]))

            fov_pt_right_x = area_extents[0][1]
            fov_pt_right_z = fov_pt_right_x / math.tan(fovx/2)

            fov_pt_left_x = - fov_pt_right_x
            fov_pt_left_z = fov_pt_right_z

            fov_pt_right = np.array([fov_pt_right_x, 0, fov_pt_right_z])
            fov_pt_left = np.array([fov_pt_left_x, 0, fov_pt_left_z])

            # print(depth_cam_index)

            fov_pt_right_index = (fov_pt_right / voxel_size - voxel_grid_2d.min_voxel_coord).astype(int)
            pts_on_fov_right_line = bresenham_line.supercover_line((color_cam_index[0], color_cam_index[1]),
                                                              (fov_pt_right_index[0], fov_pt_right_index[2]),
                                                         occupancy_map.shape[0], occupancy_map.shape[1])
            pts_on_fov_right_line_x, pts_on_fov_right_line_y = zip(*pts_on_fov_right_line)
            occupancy_map[list(pts_on_fov_right_line_x), list(pts_on_fov_right_line_y)] = -1

            fov_pt_left_index = (fov_pt_left / voxel_size - voxel_grid_2d.min_voxel_coord).astype(int)
            pts_on_fov_left_line = bresenham_line.supercover_line((color_cam_index[0], color_cam_index[1]),
                                                              (fov_pt_left_index[0], fov_pt_left_index[2]),
                                                              occupancy_map.shape[0], occupancy_map.shape[1])
            pts_on_fov_left_line_x, pts_on_fov_left_line_y = zip(*pts_on_fov_left_line)
            occupancy_map[list(pts_on_fov_left_line_x), list(pts_on_fov_left_line_y)] = -1

            # Set the occluded points behind pedestrians

            for pt in range(len(voxel_indices)):
                # TODO: fix. It should be depth_cam_index
                pts_on_line = bresenham_line.supercover_line((color_cam_index[0], color_cam_index[2]), (voxel_indices[pt, 0], voxel_indices[pt, 1]), occupancy_map.shape[0], occupancy_map.shape[1])
                ptcloud_index = pts_on_line.index((voxel_indices[pt, 0], voxel_indices[pt, 1]))
                occluded_pts = pts_on_line[ptcloud_index + 1:]
                if len(occluded_pts) == 0: continue
                occluded_xs, occluded_ys = zip(*occluded_pts)
                occupancy_map[list(occluded_xs), list(occluded_ys)] = -1

            # np.savetxt("/home/trinhle/fcav-projects/Panoptic/PPLP/pplp/core/bev_generators/height_maps_ogm/height_map_original.txt", occupancy_map)

            for (xl, yl) in pts_on_fov_left_line:
                for x in range(xl):
                    occupancy_map[x, yl] = -1

            for (xr, yr) in pts_on_fov_right_line:
                for y in range(yr):
                    occupancy_map[xr, y] = -1
            occupancy_map[voxel_indices[:, 0], voxel_indices[:, 1]] = 1
            occupancy_maps.append(occupancy_map)
            # np.savetxt("/home/trinhle/fcav-projects/Panoptic/PPLP/pplp/core/bev_generators/height_maps_ogm/height_map_completed_{}.txt".format(num), occupancy_map)

        # Rotate occupancy and height maps 90 degrees
        # (transpose and flip) is faster than np.rot90
        height_maps_out = [np.flip(height_maps[map_idx].transpose(), axis=0)
                           for map_idx in range(len(height_maps))]
        occupancy_maps_out = [np.flip(occupancy_maps[map_idx].transpose(), axis=0)
                              for map_idx in range(len(occupancy_maps))]
        # global viz_idx
        # np.savetxt("/home/trinhle/fcav-projects/Panoptic/PPLP/pplp/core/bev_generators/height_maps_ogm/height_map_rotated_{0:04d}.txt".format(viz_idx), occupancy_maps_out[0])
        # viz_idx += 1

        density_slice_filter = self.panoptic_utils.create_slice_filter(
            point_cloud,
            area_extents,
            ground_plane,
            self.height_lo,
            self.height_hi)

        density_points = all_points[density_slice_filter]

        # Create Voxel Grid 2D
        density_voxel_grid_2d = VoxelGrid2D()
        density_voxel_grid_2d.voxelize_2d(
            density_points,
            voxel_size,
            extents=area_extents,
            ground_plane=ground_plane,
            create_leaf_layout=False)

        # Generate density map
        density_voxel_indices_2d = \
            density_voxel_grid_2d.voxel_indices[:, [0, 2]]

        density_map = self._create_density_map(
            num_divisions=density_voxel_grid_2d.num_divisions,
            voxel_indices_2d=density_voxel_indices_2d,
            num_pts_per_voxel=density_voxel_grid_2d.num_pts_in_voxel,
            norm_value=self.NORM_VALUES[source])

        bev_maps = dict()
        bev_maps['height_maps'] = height_maps_out
        bev_maps['occupancy_maps'] = occupancy_maps_out
        bev_maps['density_map'] = density_map

        return bev_maps
