import argparse
import os.path
from datetime import datetime
from time import time

import numpy as np
import tensorflow as tf
import yaml
import deepbind_model.calibrate_model as calib
import deepbind_model.utils as utils


def main(train_config):
    target_protein = train_config['protein']
    model_type = train_config['model_type']
    traindir = os.path.join(train_config['model_dir'], target_protein, model_type, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(traindir)
    if 'full' in target_protein:
        train_config['protein'] = target_protein.split('_')[0]
    best_config = calib.calibrate_model(train_config)
    train_config['protein'] = target_protein
    inf = utils.load_data(target_protein)
    models = []
    inputs = []
    input_config = utils.input_config('large')
    num_final_runs = train_config['num_final_runs']
    with tf.Graph().as_default():
        # for i, model_type in enumerate(model_testing_list):
        input_data = utils.model_input(input_config, inf, model_type, validation=False)
        for runs in range(num_final_runs):
            with tf.variable_scope('model' + str(runs)):
                models.append(utils.model(best_config,input_data, model_type))
                inputs.append(input_data)
        sv = tf.train.Supervisor(logdir=traindir)
        with sv.managed_session() as session:
            (test_cost, test_pearson, cost_ensemble, pearson_ensemble) = \
                utils.train_model_parallel(session, train_config,
                                           models, inputs,
                                           epochs=train_config['train_epochs'],
                                           early_stop=True,
                                           savedir=os.path.join(traindir, target_protein + '_best_model.ckpt'),
                                           saver=sv.saver)
            # (test_cost, test_pearson) = \
            #     utils.train_model_parallel_rnacs(session, best_config,
            #                                      models, inputs,
            #                                      early_stop=False)
            # for i, model_type in enumerate(model_testing_list):
            # test_cost_filtered = np.zeros(test_cost[:, -1].shape)
            # #TODO save entire ensemble except for models with NAN pearson correlation
            # for count, num in enumerate(test_pearson[:, -1]):
            #     if np.isnan(num):
            #         test_cost_filtered[count] = np.inf
            #     else:
            #         test_cost_filtered[count] = test_cost[count, -1]
            # best_model_idx = np.argmin(test_cost_filtered[i * num_final_runs:(i + 1) * num_final_runs])

            # abs_best_model_idx = i * num_final_runs + best_model_idx
            # best_model_vars = tf.contrib.framework.get_variables(scope='model' + str(abs_best_model_idx))
            # saver = tf.train.Saver(best_model_vars)
            # sv.saver.save(session, os.path.join(traindir, target_protein + '_best_model.ckpt'))

            pearson = pearson_ensemble
            cost = cost_ensemble
            print("Pearson correlation for %s using %s is %.4f" % (
            target_protein, model_type, pearson))
            utils.save_result(train_config,
                              ensemble_size=num_final_runs,
                              new_pearson=pearson,new_cost=cost,
                              model_dir=traindir)


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
            config.update(**{'protein': protein_id})
            main(train_config=config)
            elapsed_time = (time() - start_time)
            print("Time left is" + str(elapsed_time * ((len(args.protein) / (i + 1))-1)))
    average_time = (time() - start_time) / len(args.protein)
    print("Finished process in %.4f seconds per protein" % (average_time))
    utils.summarize(config)
