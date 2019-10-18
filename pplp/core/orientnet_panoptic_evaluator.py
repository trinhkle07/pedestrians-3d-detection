"""Common functions for evaluating checkpoints.
"""

import time
import os
import numpy as np
from multiprocessing import Process

import tensorflow as tf

from pplp.core import box_3d_panoptic_encoder
from pplp.core import evaluator_panoptic_utils
from pplp.core import summary_utils
from pplp.core import trainer_utils

from pplp.core.models.orientnet_panoptic_model import OrientModel

from transforms3d.quaternions import quat2mat, mat2quat
from transforms3d.euler import euler2mat, mat2euler

tf.logging.set_verbosity(tf.logging.INFO)
KEY_SUM_ANG_LOSS = 'sum_ang_loss'
KEY_SUM_TOTAL_LOSS = 'sum_total_loss'

KEY_NUM_VALID_REG_SAMPLES = 'num_valid_reg_samples'


class Evaluator:

    def __init__(self,
                 model,
                 dataset_config,
                 eval_config,
                 skip_evaluated_checkpoints=True,
                 eval_wait_interval=30,
                 do_panoptic_native_eval=True):
        """Evaluator class for evaluating model's detection output.

        Args:
            model: An instance of DetectionModel
            dataset_config: Dataset protobuf configuration
            eval_config: Evaluation protobuf configuration
            skip_evaluated_checkpoints: (optional) Enables checking evaluation
                results directory and if the folder names with the checkpoint
                index exists, it 'assumes' that checkpoint has already been
                evaluated and skips that checkpoint.
            eval_wait_interval: (optional) The number of seconds between
                looking for a new checkpoint.
            do_panoptic_native_eval: (optional) flag to enable running panoptic native
                eval code.
        """

        # Get model configurations
        self.model = model
        self.dataset_config = dataset_config
        self.eval_config = eval_config

        self.model_config = model.model_config
        self.model_name = self.model_config.model_name
        self.full_model = isinstance(self.model, OrientModel)

        self.paths_config = self.model_config.paths_config
        self.checkpoint_dir = self.paths_config.checkpoint_dir

        self.skip_evaluated_checkpoints = skip_evaluated_checkpoints
        self.eval_wait_interval = eval_wait_interval

        self.do_panoptic_native_eval = do_panoptic_native_eval

        # Create a variable tensor to hold the global step
        self.global_step_tensor = tf.Variable(
            0, trainable=False, name='global_step')

        eval_mode = eval_config.eval_mode
        if eval_mode not in ['val', 'test']:
            raise ValueError('Evaluation mode can only be set to `val`'
                             'or `test`')

        if not os.path.exists(self.checkpoint_dir):
            raise ValueError('{} must have at least one checkpoint entry.'
                             .format(self.checkpoint_dir))

        if self.do_panoptic_native_eval:
            if self.eval_config.eval_mode == 'val':
                # Copy panoptic native eval code into the predictions folder
                evaluator_panoptic_utils.copy_panoptic_native_code(
                    self.model_config.checkpoint_name)

        allow_gpu_mem_growth = self.eval_config.allow_gpu_mem_growth
        if allow_gpu_mem_growth:
            # GPU memory config
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = allow_gpu_mem_growth
            self._sess = tf.Session(config=config)
        else:
            self._sess = tf.Session()

        if eval_mode == 'val':
            # The model should return a dictionary of predictions
            self._prediction_dict = self.model.build()
            # Setup loss and summary writer in val mode only
            self._loss_dict, self._total_loss = \
                self.model.loss(self._prediction_dict)

            self.summary_writer, self.summary_merged = \
                evaluator_panoptic_utils.set_up_summary_writer(self.model_config,
                                                               self._sess)

        else:
            self._prediction_dict = self.model.build(testing=True)
            self._loss_dict = None
            self._total_loss = None
            self.summary_writer = None
            self.summary_merged = None

        self._saver = tf.train.Saver()

        # Add maximum memory usage summary op
        # This op can only be run on device with gpu
        # so it's skipped on travis
        is_travis = 'TRAVIS' in os.environ
        if not is_travis:
            # tf 1.4
            # tf.summary.scalar('bytes_in_use',
            #                   tf.contrib.memory_stats.BytesInUse())
            tf.summary.scalar('max_bytes',
                              tf.contrib.memory_stats.MaxBytesInUse())

    def run_checkpoint_once(self, checkpoint_to_restore):
        """Evaluates network metrics once over all the validation samples.

        Args:
            checkpoint_to_restore: The directory of the checkpoint to restore.
        """

        self._saver.restore(self._sess, checkpoint_to_restore)

        data_split = self.dataset_config.data_split
        predictions_base_dir = self.paths_config.pred_dir

        eval_interval = self.eval_config.eval_interval

        num_samples = self.model.dataset.num_samples
        train_val_test = self.model._train_val_test
        print('train_val_test = ', train_val_test)

        # If train_val_test is not 'val', validation would be False
        validation = train_val_test == 'val'

        global_step = trainer_utils.get_global_step(
            self._sess, self.global_step_tensor)

        # Add folders to save predictions
        prop_score_predictions_dir = predictions_base_dir + \
            "/proposals_and_scores/{}/{}".format(
                data_split, global_step)
        trainer_utils.create_dir(prop_score_predictions_dir)

        if self.full_model:
            print('self.full_model == True')
            # Make sure the box representation is valid
            box_rep = self.model_config.avod_config.avod_box_representation
            if box_rep not in ['box_3d', 'box_8c', 'box_8co',
                               'box_4c', 'box_4ca']:
                raise ValueError('Invalid box representation {}.'.
                                 format(box_rep))

            orientnet_predictions_dir = predictions_base_dir + \
                "/final_predictions_and_scores/{}/{}".format(
                    data_split, global_step)
            trainer_utils.create_dir(orientnet_predictions_dir)

            if box_rep in ['box_8c', 'box_8co', 'box_4c', 'box_4ca']:
                box_corners_dir = predictions_base_dir + \
                    "/final_boxes_{}_and_scores/{}/{}".format(
                        box_rep, data_split, global_step)
                trainer_utils.create_dir(box_corners_dir)

            # OrientNet average losses dictionary
            eval_orientnet_losses = self._create_orientnet_losses_dict()
        else:
            raise NotImplementedError('ERROR: Only support for full_model!!')

        num_valid_samples = 0
        now_num_among_sample = 0

        # Keep track of feed_dict and inference time
        total_feed_dict_time = []
        total_inference_time = []

        # Run through a single epoch
        current_epoch = self.model.dataset.epochs_completed
        while current_epoch == self.model.dataset.epochs_completed and num_valid_samples < eval_interval:
            now_num_among_sample += 1

            # Keep track of feed_dict speed
            start_time = time.time()
            feed_dict, has_meaningful_loss = self.model.create_feed_dict()
            feed_dict_time = time.time() - start_time

            # Get sample name from model
            sample_name = self.model.sample_info['sample_name']

            # File paths for saving proposals and predictions
            rpn_file_path = prop_score_predictions_dir + "/{}.txt".format(
                sample_name)

            if self.full_model:
                orientation_file_path = orientnet_predictions_dir + \
                    "/{}.txt".format(sample_name)

                if box_rep in ['box_8c', 'box_8co', 'box_4c', 'box_4ca']:
                    box_corners_file_path = box_corners_dir + \
                        '/{}.txt'.format(sample_name)
                    print('box_corners_file_path = ', box_corners_file_path)

            # print('feed_dict = ', feed_dict)
            if not feed_dict:
                print('No useful minibatch for global_step ', global_step, '\n\n')
                continue

            # Do predictions, loss calculations, and summaries
            if validation:
                if self.summary_merged is not None:
                    predictions, eval_losses, eval_total_loss, summary_out = \
                        self._sess.run([self._prediction_dict,
                                        self._loss_dict,
                                        self._total_loss,
                                        self.summary_merged],
                                       feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_out, global_step)

                else:
                    predictions, eval_losses, eval_total_loss = \
                        self._sess.run([self._prediction_dict,
                                        self._loss_dict,
                                        self._total_loss],
                                       feed_dict=feed_dict)
                # print('eval_losses = ', eval_losses)
                # print('eval_total_loss = ', eval_total_loss)
                if self.full_model:
                    if has_meaningful_loss:
                        orientation_loss = eval_losses[OrientModel.LOSS_FINAL_ORIENTATION]

                        # if np.isnan(orientation_loss):
                        #     print('Move on, orientation_loss is NAN.')
                        #     continue

                        print('eval_losses = ', eval_losses)
                        self._update_orientnet_losses(eval_orientnet_losses,
                                                      orientation_loss,
                                                      eval_total_loss,
                                                      global_step)
                    # Save orientation predictions to npy files for future use in PPLP Net.
                    orient_results = {}
                    pred_orient_img_center = predictions[
                        OrientModel.PRED_ANGLES]
                    classes_name = 'Pedestrian'
                    sub_str = 'orient_pred'
                    orient_results['orient_pred'] = pred_orient_img_center
                    # print('line 254 orient_results = ', orient_results)
                    self._save_orient_prediction_to_file(classes_name, sub_str, sample_name, orient_results)

                    # Save orientation predictions to txt files
                    orientation_predictions = \
                        self.get_orient_predictions(predictions,
                                                    box_rep)
                    np.savetxt(orientation_file_path, orientation_predictions, fmt='%.5f')
                    print('Saving predictions_and_scores, orientation_file_path = ', orientation_file_path)
                else:
                    raise NotImplementedError('ERROR: Please set to Full Mode')

            else:
                # Test mode --> train_val_test == 'test'
                inference_start_time = time.time()
                # Don't calculate loss or run summaries for test
                predictions = self._sess.run(self._prediction_dict,
                                             feed_dict=feed_dict)
                inference_time = time.time() - inference_start_time

                # Add times to list
                total_feed_dict_time.append(feed_dict_time)
                total_inference_time.append(inference_time)

                # Save orientation predictions to npy files for future use in PPLP Net.
                orient_results = {}
                pred_orient_img_center = predictions[
                    OrientModel.PRED_ANGLES]
                classes_name = 'Pedestrian'
                sub_str = 'orient_pred'
                orient_results['orient_pred'] = pred_orient_img_center
                self._save_orient_prediction_to_file(classes_name, sub_str, sample_name, orient_results)

            num_valid_samples += 1
            print("Step {}: {}({}valid) / {}, Inference on sample {}".format(
                global_step, now_num_among_sample, num_valid_samples, num_samples,
                sample_name))

        # end while current_epoch == model.dataset.epochs_completed:

        if validation:
            if self.full_model:
                print('@@@@@@@@@@ save_prediction_losses_results @@@@@@@@@@')
                self.save_prediction_losses_results(
                    eval_orientnet_losses,
                    num_valid_samples,
                    global_step,
                    predictions_base_dir,
                    box_rep=box_rep)

                # Panoptic native evaluation, do this during validation
                # and when running Pplp model.
                # Store predictions in panoptic format
                if self.do_panoptic_native_eval:
                    print('^^^^^^^^^^^^^ save_prediction_losses_results ^^^^^^^^^^^^^')
                    self.run_panoptic_native_eval(global_step)
            else:
                raise NotImplementedError('ERROR: Please set to Full Mode for validation')

            print("Step {}: Finished evaluation, results saved to {}".format(
                global_step, prop_score_predictions_dir))

        else:
            # Test mode --> train_val_test == 'test'
            evaluator_panoptic_utils.print_inference_time_statistics(
                total_feed_dict_time, total_inference_time)

    def run_latest_checkpoints(self):
        """Evaluation function for evaluating all the existing checkpoints.
        This function just runs through all the existing checkpoints.

        Raises:
            ValueError: if model.checkpoint_dir doesn't have at least one
                element.
        """

        if not os.path.exists(self.checkpoint_dir):
            raise ValueError('{} must have at least one checkpoint entry.'
                             .format(self.checkpoint_dir))

        # Load the latest checkpoints available
        trainer_utils.load_checkpoints(self.checkpoint_dir,
                                       self._saver)

        num_checkpoints = len(self._saver.last_checkpoints)

        if self.skip_evaluated_checkpoints:
            already_evaluated_ckpts = self.get_evaluated_ckpts(
                self.model_config, self.model_name)

        ckpt_indices = np.asarray(self.eval_config.ckpt_indices)
        if ckpt_indices is not None:
            print('ckpt_indices = ', ckpt_indices)
            if ckpt_indices[0] == -1:
                # Restore the most recent checkpoint
                ckpt_idx = num_checkpoints - 1
                ckpt_indices = [ckpt_idx]
                print('***************Restore the most recent checkpoint********')
                print('ckpt_idx = ', ckpt_idx)
                print('ckpt_indices = ', ckpt_indices)
            for ckpt_idx in ckpt_indices:
                # checkpoint_to_restore = self._saver.last_checkpoints[ckpt_idx]
                print('*** Warning!! This is a hack!')
                checkpoint_to_restore = '/home/trinhle/PPLP/src/PPLP/pplp/data/outputs/orientation_pedestrian_panoptic/checkpoints/orientation_pedestrian_panoptic-00306952'
                # checkpoint_to_restore = '/home/trinhle/LidarPoseBEV/src/PPLP/pplp/data/outputs/orientation_pedestrian_panoptic/checkpoints/orientation_pedestrian_panoptic-00306952'
                print('checkpoint_to_restore = ', checkpoint_to_restore)
                self.run_checkpoint_once(checkpoint_to_restore)

        else:
            last_checkpoint_id = -1
            number_of_evaluations = 0
            # go through all existing checkpoints
            for ckpt_idx in range(num_checkpoints):
                checkpoint_to_restore = self._saver.last_checkpoints[ckpt_idx]
                ckpt_id = evaluator_panoptic_utils.strip_checkpoint_id(
                    checkpoint_to_restore)

                # Check if checkpoint has been evaluated already
                already_evaluated = ckpt_id in already_evaluated_ckpts
                if already_evaluated or ckpt_id <= last_checkpoint_id:
                    number_of_evaluations = max((ckpt_idx + 1,
                                                 number_of_evaluations))
                    continue

                self.run_checkpoint_once(checkpoint_to_restore)
                number_of_evaluations += 1

                # Save the id of the latest evaluated checkpoint
                last_checkpoint_id = ckpt_id

    def repeated_checkpoint_run(self):
        """Periodically evaluates the checkpoints inside the `checkpoint_dir`.

        This function evaluates all the existing checkpoints as they are being
        generated. If there are none, it sleeps until new checkpoints become
        available. Since there is no synchronization guarantee for the trainer
        and evaluator, at each iteration it reloads all the checkpoints and
        searches for the last checkpoint to continue from. This is meant to be
        called in parallel to the trainer to evaluate the models regularly.

        Raises:
            ValueError: if model.checkpoint_dir doesn't have at least one
                element.
        """

        if not os.path.exists(self.checkpoint_dir):
            raise ValueError('{} must have at least one checkpoint entry.'
                             .format(self.checkpoint_dir))

        # Copy panoptic native eval code into the predictions folder
        if self.do_panoptic_native_eval:
            evaluator_panoptic_utils.copy_panoptic_native_code(
                self.model_config.checkpoint_name)

        if self.skip_evaluated_checkpoints:
            already_evaluated_ckpts = self.get_evaluated_ckpts(
                self.model_config, self.full_model)
        tf.logging.info(
            'Starting evaluation at ' +
            time.strftime(
                '%Y-%m-%d-%H:%M:%S',
                time.gmtime()))

        last_checkpoint_id = -1
        number_of_evaluations = 0
        while True:
            # Load current checkpoints available
            trainer_utils.load_checkpoints(self.checkpoint_dir,
                                           self._saver)
            num_checkpoints = len(self._saver.last_checkpoints)

            start = time.time()

            if number_of_evaluations >= num_checkpoints:
                tf.logging.info('No new checkpoints found in %s.'
                                'Will try again in %d seconds',
                                self.checkpoint_dir,
                                self.eval_wait_interval)
            else:
                for ckpt_idx in range(num_checkpoints):
                    checkpoint_to_restore = \
                        self._saver.last_checkpoints[ckpt_idx]
                    ckpt_id = evaluator_panoptic_utils.strip_checkpoint_id(
                        checkpoint_to_restore)

                    # Check if checkpoint has been evaluated already
                    already_evaluated = ckpt_id in already_evaluated_ckpts
                    if already_evaluated or ckpt_id <= last_checkpoint_id:
                        number_of_evaluations = max((ckpt_idx + 1,
                                                     number_of_evaluations))
                        continue

                    self.run_checkpoint_once(checkpoint_to_restore)
                    number_of_evaluations += 1

                    # Save the id of the latest evaluated checkpoint
                    last_checkpoint_id = ckpt_id

            time_to_next_eval = start + self.eval_wait_interval - time.time()
            if time_to_next_eval > 0:
                time.sleep(time_to_next_eval)

    def _update_orientnet_losses(self,
                                 eval_orientnet_losses,
                                 orientation_loss,
                                 eval_total_loss,
                                 global_step):
        """Helper function to calculate the evaluation average losses.

        Args:
            eval_orientnet_losses: A dictionary containing all the average
                losses.
            orientation_loss: A scalar loss of orientations.
            global_step: Global step at which the metrics are computed.
        """

        print("Step {}: Eval Loss: orientation {:.3f}, "
              "total {:.3f}".format(
                    global_step,
                    orientation_loss,
                    eval_total_loss))

        # Get the loss sums from the losses dict
        sum_ang_loss = eval_orientnet_losses[KEY_SUM_ANG_LOSS]

        sum_ang_loss += orientation_loss

        # update the losses sums
        eval_orientnet_losses.update({KEY_SUM_ANG_LOSS:
                                      sum_ang_loss})

    def save_prediction_losses_results(self,
                                       eval_losses,
                                       num_valid_samples,
                                       global_step,
                                       predictions_base_dir,
                                       box_rep):
        """Helper function to save the PPLP loss evaluation results.

        Args:
            eval_avod_losses: A dictionary containing the loss sums
            num_valid_samples: An int, number of valid evaluated samples
                i.e. samples with valid ground-truth.
            global_step: Global step at which the metrics are computed.
            predictions_base_dir: Base directory for storing the results.
            box_rep: A string, the format of the 3D bounding box
                one of 'box_3d', 'box_8c' etc.
        """
        if box_rep in ['box_3d', 'box_4ca']:
            sum_ang_loss = eval_losses[KEY_SUM_ANG_LOSS]
            print('sum_ang_loss = ', sum_ang_loss)
        else:
            sum_ang_loss = 0

        # Write summaries
        if box_rep in ['box_3d', 'box_4ca']:
            summary_utils.add_scalar_summary(
                'orient_losses/orientation',
                sum_ang_loss,
                self.summary_writer, global_step)

        print("Step {}: Average Losses: "
              "orientation {:.5f}".format(
                global_step,
                sum_ang_loss,
                  ))

        # Append to end of file
        avg_loss_file_path = predictions_base_dir + '/orient_losses.csv'
        if box_rep in ['box_3d', 'box_4ca']:
            with open(avg_loss_file_path, 'ba') as fp:
                np.savetxt(fp,
                           [np.hstack(
                            [global_step,
                             sum_ang_loss]
                            )],
                           fmt='%d, %.5f')
        else:
            raise NotImplementedError('Saving losses not implemented')

    def _create_orientnet_losses_dict(self):
        """Returns a dictionary of the losses sum for averaging.
        """
        eval_orientnet_losses = dict()

        # Initialize orientnet losses
        eval_orientnet_losses[KEY_SUM_ANG_LOSS] = 0
        eval_orientnet_losses[KEY_SUM_TOTAL_LOSS] = 0

        return eval_orientnet_losses

    def get_evaluated_ckpts(self,
                            model_config,
                            model_name):
        """Finds the evaluated checkpoints.

        Examines the evaluation average losses file to find the already
        evaluated checkpoints.

        Args:
            model_config: Model protobuf configuration
            model_name: A string representing the model name.

        Returns:
            already_evaluated_ckpts: A list of checkpoint indices, or an
                empty list if no evaluated indices are found.
        """

        already_evaluated_ckpts = []

        # check for previously evaluated checkpoints
        # regardless of model, we are always evaluating rpn, but we do
        # this check based on model in case the evaluator got interrupted
        # and only saved results for one model
        paths_config = model_config.paths_config

        predictions_base_dir = paths_config.pred_dir
        if model_name == 'avod_model':
            avg_loss_file_path = predictions_base_dir + '/avod_avg_losses.csv'
        else:
            avg_loss_file_path = predictions_base_dir + '/rpn_avg_losses.csv'

        if os.path.exists(avg_loss_file_path):
            avg_losses = np.loadtxt(avg_loss_file_path, delimiter=',')
            if avg_losses.ndim == 1:
                # one entry
                already_evaluated_ckpts = np.asarray(
                    [avg_losses[0]], np.int32)
            else:
                already_evaluated_ckpts = np.asarray(avg_losses[:, 0],
                                                     np.int32)

        return already_evaluated_ckpts

    def get_orient_predictions(self, predictions,
                               box_rep,
                               testing=False):
        """Returns the predictions and scores stacked for saving to file.

        Args:
            predictions: A dictionary containing the model outputs.
            box_rep: A string indicating the format of the 3D bounding
                boxes i.e. 'box_3d', 'box_8c' etc.
            testing: True, if this is not validation mode.

        Returns:
            predictions_and_scores: A numpy array of shape
                (number_of_predicted_orientation, 1), containing the final
                prediction orientations.
        """

        # Predicted orientation from layers
        # all angle_vectors and orientations are correspond to pedestrians at the image center.
        pred_orient_img_center = predictions[
            OrientModel.PRED_ANGLES]
        pred_quat_img_center = predictions[
            OrientModel.PRED_ANGLE_VECTORS]  # Not normed.
        # print('pred_orient_img_center = ', pred_orient_img_center)
        # print('pred_quat_img_center = ', pred_quat_img_center)

        if not testing:
            pred_boxes_3d = predictions[
                OrientModel.PRED_BOXES_3D]
            final_pred_boxes_3d = pred_boxes_3d
            # [x, y, z, l, w, h, ry]
            # print('final_pred_boxes_3d.shape = ', pred_boxes_3d.shape)
            # print('final_pred_boxes_3d = ', pred_boxes_3d)
            final_pred_scores = np.ones([pred_boxes_3d.shape[0], 1])
            final_pred_types = np.zeros([pred_boxes_3d.shape[0], 1])  # We only have one type, which is Pedestrian. So type index should be zero.

        else:
            # print('pred_orient_img_center.shape = ', pred_orient_img_center.shape)
            final_pred_boxes_3d = np.ones([pred_orient_img_center.shape[0], 7])
            final_pred_boxes_3d[:, 6] = pred_orient_img_center
            final_pred_scores = np.ones([pred_orient_img_center.shape[0], 1])
            final_pred_types = np.zeros([pred_orient_img_center.shape[0], 1])  # We only have one type, which is Pedestrian. So type index should be zero.

        if not testing:
            # Rotate the orientations back to the groundtruth of each pedestrian.
            for j in range(len(pred_orient_img_center)):
                if np.isnan(pred_boxes_3d[j][0]):
                    # If this result is from an invalid Mask RCNN crop, then we
                    # plot the 3D location out of the valid area.
                    final_pred_boxes_3d[j, :] = [0.0, 0.0, -10.0, 1.0, 1.0, 1.0, 0.0]
                else:
                    # use translation to get apparant orientation of object in the
                    # camera view.
                    pitch_angle = np.arctan2(pred_boxes_3d[j][0], pred_boxes_3d[j][2]+pred_boxes_3d[j][5]/2);
                    roll_angle = -np.arctan2(pred_boxes_3d[j][1], pred_boxes_3d[j][2]+pred_boxes_3d[j][5]/2);
                    # When the camera coordinate is defined as:
                    # x points to the ground; y points down to the floor;
                    # z shoot out from the camera.
                    # Then angle definition should be:
                    # (make sure the definition is the same in all orientation-related files!!)
                    # pitch_angle = np.arctan2(label_boxes_3d[j][0], label_boxes_3d[j][2]);
                    # roll_angle = np.arctan2(label_boxes_3d[j][1]-label_boxes_3d[j][5]/2, label_boxes_3d[j][2]);
                    # print('For people #', j, ': pitch_angle=', pitch_angle, '; roll_angle=', roll_angle, '; yaw_angle=', label_boxes_3d[j][6])
                    rot = euler2mat(roll_angle, pitch_angle, 0, axes='rxyz')
                    R_old = quat2mat(pred_quat_img_center[j])
                    R_new = np.dot(rot, R_old)
                    new_orient = mat2euler(R_new, axes='ryzx')
                    # Yes, very weird! When we tranlate from euler2quat, we use axes='rxyz', now we should use axes='ryzx'.
                    # For annimation: https://quaternions.online/
                    # print('new_orient = ', new_orient)
                    # When pred_boxes_3d[i ,:] are all np.nan, new_orient =  (-0.0, nan, nan)
                    final_pred_boxes_3d[j, 6] = new_orient[0]  # pick the first index since we use y-z-x

        # Stack into prediction format
        predictions = np.column_stack(
            [final_pred_boxes_3d,
             final_pred_scores,
             final_pred_types])

        return predictions

    def get_avod_predicted_box_corners_and_scores(self,
                                                  predictions,
                                                  box_rep):

        if box_rep in ['box_8c', 'box_8co']:
            final_pred_box_corners = predictions[OrientModel.PRED_TOP_BOXES_8C]
        elif box_rep in ['box_4c', 'box_4ca']:
            final_pred_box_corners = predictions[OrientModel.PRED_TOP_BOXES_4C]

        # Append score and class index (object type)
        final_pred_softmax = predictions[
            OrientModel.PRED_TOP_CLASSIFICATION_SOFTMAX]

        # Find max class score index
        not_bkg_scores = final_pred_softmax[:, 1:]
        final_pred_types = np.argmax(not_bkg_scores, axis=1)

        # Take max class score (ignoring background)
        final_pred_scores = np.array([])
        for pred_idx in range(len(final_pred_box_corners)):
            all_class_scores = not_bkg_scores[pred_idx]
            max_class_score = all_class_scores[final_pred_types[pred_idx]]
            final_pred_scores = np.append(final_pred_scores, max_class_score)

        if box_rep in ['box_8c', 'box_8co']:
            final_pred_box_corners = np.reshape(final_pred_box_corners,
                                                [-1, 24])
        # Stack into prediction format
        # print('final_pred_box_corners = ', final_pred_box_corners)
        predictions_and_scores = np.column_stack(
            [final_pred_box_corners,
             final_pred_scores,
             final_pred_types])

        # print('predictions_and_scores = ', predictions_and_scores)
        return predictions_and_scores

    def run_panoptic_native_eval(self, global_step):
        """Calls the panoptic native C++ evaluation code.

        It first saves the predictions in panoptic format. It then creates two
        child processes to run the evaluation code. The native evaluation
        hard-codes the IoU threshold inside the code, so hence its called
        twice for each IoU separately.

        Args:
            global_step: Global step of the current checkpoint to be evaluated.
        """

        # Panoptic native evaluation, do this during validation
        # and when running Pplp model.
        # Store predictions in panoptic format
        evaluator_panoptic_utils.save_predictions_in_panoptic_format(
            self.model,
            self.model_config.checkpoint_name,
            self.dataset_config.data_split,
            self.eval_config.kitti_score_threshold,
            global_step)

        checkpoint_name = self.model_config.checkpoint_name
        kitti_score_threshold = self.eval_config.kitti_score_threshold
        print('kitti_score_threshold = ', kitti_score_threshold)

        # Create a separate processes to run the native evaluation
        native_eval_proc = Process(
            target=evaluator_panoptic_utils.run_panoptic_native_script, args=(
                checkpoint_name, kitti_score_threshold, global_step))
        native_eval_proc_05_iou = Process(
            target=evaluator_panoptic_utils.run_panoptic_native_script_with_05_iou,
            args=(checkpoint_name, kitti_score_threshold, global_step))
        # Don't call join on this cuz we do not want to block
        # this will cause one zombie process - should be fixed later.
        native_eval_proc.start()
        native_eval_proc_05_iou.start()

    def make_file_path(self, classes_name, sub_str, sample_name, subsub_str=None):
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
        mini_batch_dir = 'pplp/data/mini_batches/iou_2d/panoptic/train/lidar'
        if sample_name:
            if subsub_str:
                return mini_batch_dir + '/' + classes_name + \
                    '[' + sub_str + ']/' + \
                    subsub_str + '/' + \
                    sample_name + '.npy'
            else:
                return mini_batch_dir + '/' + classes_name + \
                    '[' + sub_str + ']/' + \
                    sample_name + '.npy'
        else:
            if subsub_str:
                return mini_batch_dir + '/' + classes_name + \
                    '[' + sub_str + ']/' + subsub_str
            else:
                return mini_batch_dir + '/' + classes_name + \
                    '[' + sub_str + ']'

    def _save_orient_prediction_to_file(self, classes_name, sub_str, sample_name,
                                        orient_results=None):
        """
        Saves the MRCNN info matrix to a file

        Args:
            classes_name (str): classes name, e.g. 'Car', 'Pedestrian',
                'Cyclist', 'People'
            anchor_strides: anchor strides
            sample_name (str): name of sample, e.g. '000123'
            orient_results: To Do
        """
        if orient_results:
            # print('orient_results = ', orient_results)
            # Save msakrcnn_result
            file_name = self.make_file_path(classes_name,
                                            sub_str,
                                            sample_name)
            print('_save_orient_prediction_to_file :: file_name = ', file_name)
            np.save(file_name, orient_results)

# np.hstack(array1,array2) vstack concatenate
# result['rois'] = array
        else:
            results = {}
            file_name = self.make_file_path(classes_name,
                                            sub_str,
                                            sample_name)
            print('_save_orient_prediction_to_file : results empty : file_name = ', file_name)
            # Save to npy file
            np.save(file_name, results)
