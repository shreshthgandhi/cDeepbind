import argparse
import os.path
from datetime import datetime
from time import time

import numpy as np
import tensorflow as tf
import yaml

import deepbind_model.calibrate_model as calib
import deepbind_model.utils as utils


def main(target_protein, model_size_flag, model_testing_list, num_calibrations, recalibrate, num_final_runs,
         train_epochs):
    traindir = {}
    for model_type in model_testing_list:
        model_dir = os.path.join('../models/', target_protein, model_type, model_size_flag,
                                 datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(model_dir)
        traindir[model_type] = model_dir
    best_config = {}


    for model_type in model_testing_list:
        calib_temp = utils.load_calibration(target_protein, model_type, 'small', '../calibrations')
        if recalibrate:
            best_config[model_type], _, _ = calib.calibrate_model(target_protein,
                                                                  num_calibrations=num_calibrations,
                                                                  model_type=model_type,
                                                                  flag='small')
        elif calib_temp:
            best_config[model_type] = calib_temp
        else:
            best_config[model_type], _, _ = calib.calibrate_model(target_protein,
                                                                  num_calibrations=num_calibrations,
                                                                  model_type=model_type,
                                                                  flag='small')
        best_config[model_type]['epochs'] = train_epochs

    inf = utils.load_data(target_protein)
    models = []
    inputs = []
    input_data = {}
    input_config = utils.input_config(model_size_flag)

    with tf.Graph().as_default():
        for i, model_type in enumerate(model_testing_list):
            input_data[model_type] = utils.model_input(input_config, inf, model_type, validation=False)
            for runs in range(num_final_runs):
                with tf.variable_scope('model' + str(runs + i * num_final_runs)):
                    models.append(utils.model(best_config[model_type],
                                              input_data[model_type],
                                              model_type))
                    inputs.append(input_data[model_type])
        with tf.Session() as session:
            (test_cost, test_pearson) = \
                utils.train_model_parallel_rnacs(session, best_config[model_type],
                                           models, inputs,
                                           early_stop=False)
            for i, model_type in enumerate(model_testing_list):
                test_cost_filtered = np.zeros(test_cost[:, -1].shape)
                #TODO save entire ensemble except for models with NAN pearson correlation
                for count, num in enumerate(test_pearson[:, -1]):
                    if np.isnan(num):
                        test_cost_filtered[count] = np.inf
                    else:
                        test_cost_filtered[count] = test_cost[count, -1]
                best_model_idx = np.argmin(test_cost_filtered[i * num_final_runs:(i + 1) * num_final_runs])

                abs_best_model_idx = i * num_final_runs + best_model_idx
                best_model_vars = tf.contrib.framework.get_variables(scope='model' + str(abs_best_model_idx))
                saver = tf.train.Saver(best_model_vars)
                saver.save(session, os.path.join(traindir[model_type], target_protein + '_best_model.ckpt'))

                pearson = test_pearson[abs_best_model_idx, -1]
                cost = test_cost[abs_best_model_idx, -1]
                print("Pearson correlation for %s using %s is %.4f" % (
                target_protein, model_type, test_pearson[abs_best_model_idx, -1]))
                utils.save_result(target_protein, model_type,
                                  model_size_flag, cost, pearson,
                                  save_dir='../results',
                                  model_index=abs_best_model_idx,
                                  model_dir=traindir[model_type])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=None, type=int, nargs='+')
    parser.add_argument('--protein', default=None, nargs='+')
    parser.add_argument('--configuration', default=None)
    args = parser.parse_args()
    config = yaml.load(open(args.configuration, 'r'))
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))
    start_time = time()
    if not (config.get('summary_only', False)):
        for i, protein_id in enumerate(args.protein):
            main(target_protein=protein_id, model_size_flag=config.get('model_scale', 'large'),
                 model_testing_list=config.get('model_testing_list', ['RNN_struct']),
                 num_calibrations=config.get('num_calibrations', 5),
                 recalibrate=config.get('recalibrate', False),
                 num_final_runs=config.get('num_final_runs', 3),
                 train_epochs=config.get('train_epochs', 15))
            elapsed_time = (time() - start_time)
            print("Time left is" + str(elapsed_time * ((len(args.protein) / (i + 1))-1)))
    average_time = (time() - start_time) / len(args.protein)
    print("Finished process in %.4f seconds per protein" % (average_time))
