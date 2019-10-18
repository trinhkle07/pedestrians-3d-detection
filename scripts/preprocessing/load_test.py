# Add this block for ROS python conflict
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    sys.path.remove('$HOME/segway_kinetic_ws/devel/lib/python2.7/dist-packages')
except ValueError:
    pass

import numpy as np
import cv2
from PIL import Image

# np.set_printoptions(threshold=np.nan)
file_name = '/home/trinhle/Panoptic/projects/bodypose2dsim/160422_ultimatum1/training/rgb_images/500100018739.jpg'
color_img = Image.open(file_name)

color_img.load()
color_img_data = np.asarray(color_img, dtype="int8")

file_name = '/home/trinhle/LidarPoseBEV/src/PPLP/pplp/data/mini_batches/iou_2d/panoptic/train/lidar/Pedestrian[mrcnn]/500100018739.npy'
mrcnn_results = np.load(file_name)

# print('mrcnn_results[0] = ', mrcnn_results[0])
# print('mrcnn_results.shape = ', mrcnn_results.shape)
# print('mrcnn_results.item().get(rois) = ', mrcnn_results.item().get('rois'))
image_mrcnn_feature_input = mrcnn_results.item().get('features')
# features: [N, 28, 28, 17]
image_mrcnn_bbox_input = mrcnn_results.item().get('rois')
# rois: [batch, N, (y1, x1, y2, x2)] detection bounding boxes

image_mask_input = mrcnn_results.item().get('masks')
# print('image_mrcnn_feature_input = ', image_mrcnn_feature_input)
print('image_mrcnn_feature_input.shape = ', image_mrcnn_feature_input.shape)
# print('image_mrcnn_bbox_input = ', image_mrcnn_bbox_input)
# print('image_mask_input = ', image_mask_input)
# for i in range(image_mrcnn_feature_input.shape[0]):
for i in range(1):
    color_img_data_crop = color_img_data[image_mrcnn_bbox_input[i, 0]:image_mrcnn_bbox_input[i, 2], image_mrcnn_bbox_input[i, 1]:image_mrcnn_bbox_input[i, 3], :]
    print('color_img_data_crop = ', color_img_data_crop)

    color_img_data_crop_resize = cv2.resize(color_img_data_crop.astype(np.float32), (280, 280))
    color_img_data_crop_resize_int = color_img_data_crop_resize.astype(np.int8)
    color_img_croped = Image.fromarray(color_img_data_crop_resize_int, 'RGB')
    color_img_croped.show(title="Person")

    for j in range(17):
        # plot features
        grey_feature = np.expand_dims(image_mrcnn_feature_input[i, :, :, j], axis=2)
        feature_max = np.max(grey_feature)
        feature_min = np.min(grey_feature)
        grey_feature = (grey_feature-feature_min)*255/(feature_max-feature_min)
        grey_feature = np.tile(grey_feature, (1, 1, 3))

        # print('color_img_data_crop_resize = ', color_img_data_crop_resize)
        # print('grey_feature = ', grey_feature)

        # superimpose_img_resize = (color_img_data_crop_resize + grey_feature)/2
        grey_feature_resize = cv2.resize(grey_feature, (280, 280))
        grey_feature_int = grey_feature_resize.astype(np.int8)
        img = Image.fromarray(grey_feature_int, 'RGB')
        img.show(title="feature")


