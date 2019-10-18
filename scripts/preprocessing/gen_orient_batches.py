# Add this block for ROS python conflict
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    sys.path.remove('$HOME/segway_kinetic_ws/devel/lib/python2.7/dist-packages')
except ValueError:
    pass

import numpy as np
import os
from wavedata.tools.obj_detection import obj_utils

mini_batch_dir = 'pplp/data/mini_batches/iou_2d/panoptic/train/lidar'


def check_for_npy_existing(classes_name, sub_str, sample_name, subsub_str=None):
    """
    Checks if a mini batch file exists already

    Args:
        classes_name (str): classes name, e.g. 'Car', 'Pedestrian',
            'Cyclist', 'People'
        sub_str: a name for folder subname
        sample_name (str): sample name from dataset, e.g. '000123'

    Returns:
        True if the anchors info file already exists
    """

    file_name = make_file_path(classes_name,
                               sub_str,
                               sample_name,
                               subsub_str= subsub_str)
    if os.path.exists(file_name):
        return True

    return False


def make_file_path(classes_name, sub_str, sample_name, subsub_str=None):
    """Make a full file path to the mini batches

    Args:
        classes_name: name of classes ('Car', 'Pedestrian', 'Cyclist',
            'People')
        sub_str: a name for folder subname
        sample_name: sample name, e.g. '000123'

    Returns:
        The anchors info file path. Returns the folder if
            sample_name is None
    """

    if sample_name:
        if subsub_str:
            return mini_batch_dir + '/' + classes_name + \
                '[' + sub_str + ']/' + \
                subsub_str + '/' + \
                sample_name + ".npy"
        else:
            return mini_batch_dir + '/' + classes_name + \
                '[' + sub_str + ']/' + \
                sample_name + ".npy"
    else:
        if subsub_str:
            return mini_batch_dir + '/' + classes_name + \
                '[' + sub_str + ']/' + subsub_str
        else:
            return mini_batch_dir + '/' + classes_name + \
                '[' + sub_str + ']'


def read_mrcnn_from_file(classes_name, sample_name):
    """
    Reads the MRCNN info matrix from a file

    Args:
        classes_name (str): classes name, e.g. 'Car', 'Pedestrian',
            'Cyclist', 'People'
        sample_name (str): name of sample, e.g. '000123'

    Returns:
        mrcnn_results: {'scores': array(dtype=float32),
                        'features': array(dtype=float32),
                        'keypoints': array(),
                        'class_ids': array(dtype=int32),
                        'masks': array(dtype=float32),
                        'rois': array(dtype=int32),
                        'full_masks': array(dtype=uint8)
    """

    sub_str = 'mrcnn'
    results = {}

    file_name = make_file_path(classes_name,
                               sub_str,
                               sample_name)
    print('read_mrcnn_from_file :: file_name = ', file_name)
    # Load from npy file
    results = np.load(file_name)
    return results


def save_numpy_to_file(classes_name, sub_str, sample_name,
                       numpy_results=None):
    """
    Saves the MRCNN info matrix to a file

    Args:
        classes_name (str): classes name, e.g. 'Car', 'Pedestrian',
            'Cyclist', 'People'
        anchor_strides: anchor strides
        sample_name (str): name of sample, e.g. '000123'
        mrcnn_results: To Do
    """
    if numpy_results:
        # print('mrcnn_results = ', mrcnn_results)
        # Save msakrcnn_result
        file_name = make_file_path(classes_name,
                                   sub_str,
                                   sample_name)
        print('save_numpy_to_file :: file_name = ', file_name)
        np.save(file_name, numpy_results)

# np.hstack(array1,array2) vstack concatenate
# result['rois'] = array
    else:
        results = {}
        file_name = make_file_path(classes_name,
                                   sub_str,
                                   sample_name)
        print('save_numpy_to_file : results empty : file_name = ', file_name)
        # Save to npy file
        np.save(file_name, results)


def main():
    # Get image name for given cluster
    sample_name = '500100008677'
    img_idx = int(sample_name)
    print('img_idx = ', img_idx)
    classes_name = 'Pedestrian'
    sub_str = 'orient'
    # Check for existing files and skip to the next
    if check_for_npy_existing(classes_name, sub_str, sample_name):
        print("Sample {} already existed".format(sample_name))
        pass
    else:

        # Here is the MaskRCNN result for image 500100008677:
        # mrcnn_result = read_mrcnn_from_file(classes_name, sample_name)
        # rois = mrcnn_result.item().get('rois')  #[y1,x1,y2,x2]
        # print('rois = ', rois)
        # rois =  [[ 313  339  931  605]
        #  [ 206  631 1080 1025]
        #  [ 345 1044  836 1216]
        #  [   0 1224 1080 1815]]

        # Here is the new(rearranged) ground truth for image 500100008677:
        # Pedestrian 0.00 0 0 296.6047253544661 264.09461079442656 626.4882209075746 976.3531382411459 1.7005607 0.7177887493822134 0.43045915102889426 -1.321877924802265 1.0532322877898808 2.936737651253294 0.5213697381808344
        # Pedestrian 0.00 0 0 619.6859203625773 129.42926341034263 1047.8673467660992 1080.0 1.6354 0.677518392987404 0.4245842109951549 -0.21308230750209334 1.0179379082054192 1.9665064743805873 -1.0007620901680785
        # Pedestrian 0.00 0 0 1042.7051415153514 331.2323586388291 1216.172668075389 850.2639457948231 1.6848714 0.6145644067408184 0.4942507228282457 0.620123562293023 1.022690231477789 3.738146974738853 3.1132133955650314
        # Pedestrian 0.00 0 0 1135.0325332954062 0.0 1920.0 1080.0 1.730273 0.6505859274834352 0.7061923815414736 0.6156645920067837 1.040354438453105 1.2074371310152239 -1.6158589561126675
        # 1 type Describes the type of object: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare'
        # 1 truncated Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries
        # 1 occluded Integer (0,1,2,3) indicating occlusion state: 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown
        # 1 alpha Observation angle of object, ranging [-pi..pi]
        # 4 bbox 2D bounding box of object in the image (0-based index): contains left, top, right, bottom pixel coordinates
        # 3 dimensions 3D object dimensions: height, width, length (in meters)
        # 3 location 3D object location x,y,z in camera coordinates (in meters)
        # 1 rotation_y Rotation ry around Y-axis in camera coordinates [-pi..pi]
        # 1 score Only for results: Float, indicating confidence in detection, needed for p/r curves, higher is better.
        results = {}
        orient_gt = np.array([[296.6047253544661, 264.09461079442656, 626.4882209075746, 976.3531382411459, 1.7005607, 0.7177887493822134, 0.43045915102889426, -1.321877924802265, 1.0532322877898808, 2.936737651253294, 0.5213697381808344],
                              [619.6859203625773, 129.42926341034263, 1047.8673467660992, 1080.0, 1.6354, 0.677518392987404, 0.4245842109951549, -0.21308230750209334, 1.0179379082054192, 1.9665064743805873, -1.0007620901680785],
                              [1042.7051415153514, 331.2323586388291, 1216.172668075389, 850.2639457948231, 1.6848714, 0.6145644067408184, 0.4942507228282457, 0.620123562293023, 1.022690231477789, 3.738146974738853, 3.1132133955650314],
                              [1135.0325332954062, 0.0, 1920.0, 1080.0, 1.730273, 0.6505859274834352, 0.7061923815414736, 0.6156645920067837, 1.040354438453105, 1.2074371310152239, -1.6158589561126675]])
        results['boxes_3d'] = orient_gt
        save_numpy_to_file(classes_name, sub_str, sample_name, results)

    sample_name = '500100008799'
    if check_for_npy_existing(classes_name, sub_str, sample_name):
        print("Sample {} already existed".format(sample_name))
        pass
    else:
        # Here is the MaskRCNN result for image 500100008799:
        # mrcnn_result = read_mrcnn_from_file(classes_name, sample_name)
        # rois = mrcnn_result.item().get('rois')  #[y1,x1,y2,x2]
        # print('rois = ', rois)
        # rois =  [[   0  986 1072 1501]
        #          [ 313  496  856  663]
        #          [  88   67 1066  406]
        #          [ 290  300  969  523]
        #          [ 367  969  800 1078]
        #          [ 341 1385  918 1509]]

        results = {}
        orient_gt = np.array([[910.1966326255737, 0.0, 1636.8000768996615, 1080.0, 1.84594, 0.7760509694619536, 0.4388653901815411, 0.4006145675694894, 1.013628052340647, 1.2815075958378248, -2.114496277200524],
                              [482.822162343592, 280.29601436114103, 697.291580413098, 869.1384411860638, 1.785169, 0.636169833820086, 0.34933946975518126, -1.184168993251622, 1.0123265963492125, 3.5589641294214274, 0.4074797756203337],
                              [31.39064032625749, 0.0, 412.1286333470219, 1080.0, 1.723057, 0.617044220813856, 0.4144319853991087, -1.096452030226585, 1.0279430428550695, 1.655185718149433, -0.539345884008696],
                              [282.283127810287, 239.78779757419775, 539.9617368387146, 1009.1864328654314, 1.7146559, 0.6789736924018451, 0.3301401069755387, -1.3292702875258207, 1.0562656381721365, 2.6952723569830543, -0.03234437147012374],
                              [956.3979609183284, 365.30809450463016, 1128.0136271820704, 811.2314773537853, 1.63847, 0.6021112304521172, 0.3424221231277215, 0.3761851571002706, 1.0141574599886747, 4.141488788343012, 1.9291042228368813],
                              [1279.8713315737218, 270.38739624561913, 1523.9240688633583, 920.5326650491917, 1.740053, 0.5502812553432291, 0.42018035033287615, 1.3283609744078233, 1.0312533074195263, 3.1407227269123634, 2.8977103571510234]])
        results['boxes_3d'] = orient_gt
        save_numpy_to_file(classes_name, sub_str, sample_name, results)


if __name__ == '__main__':
    main()
