import argparse
import os.path

import numpy as np
import tensorflow as tf
import yaml

import deepbind_model.utils as utils


def main(target_protein, model_type):
    config = utils.load_calibration(target_protein, model_type, 'small', '../calibrations')
    result_file = yaml.load(open('../results_final/' + target_protein + '_' + model_type + '_large.yml'))
    model_dir = result_file['model_dir']
    model_idx = result_file['model_index']
    inf = utils.load_data(target_protein)
    input_data = utils.Deepbind_input(utils.input_config('large'), inf, model_type, validation=False)
    with tf.Graph().as_default():
        with tf.variable_scope('model' + str(model_idx)):
            model = utils.Deepbind_model(config, input_data, model_type)
        best_model_vars = tf.contrib.framework.get_variables(scope='model' + str(model_idx))
        saver = tf.train.Saver(best_model_vars)
        with tf.Session() as sess:
            saver.restore(sess,
                          os.path.join(model_dir, target_protein + '_best_model.ckpt'))
            # scores = utils.evaluate_model_parallel(sess, config, [model], input_data)
            # utils.compute_gradient(sess, config, model, input_data)
            # print(scores)
            (cost_train, cost_test,training_pearson, test_pearson,training_scores,test_scores)  = utils.run_epoch_parallel(sess, [model], input_data,config,epoch=1,train=False,verbose=False,testing=True,scores=True)
            new_pearson_test = test_pearson[0,0]
            new_pearson_train = training_pearson[0,0]
            new_cost_train = cost_train[0]
            new_cost_test = cost_test[0]
            # inf = np.load('../results_final/'+target_protein+'_'+model_type+'_large.npz')
            # keywords = {}
            # for key in inf.keys():
            #     keywords[key] = inf[key]
            # keywords['cost'] = new_cost_test
            # keywords['pearson'] = new_pearson_test
            print("True pearson for %s is %f"%(target_protein,test_pearson[0,0]))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=None, type=int, nargs='+')
    parser.add_argument('--protein', default=None, nargs='+')
    parser.add_argument('--model_type', default=None)

    args = parser.parse_args()
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))
    for protein_id in args.protein:
        main(target_protein=protein_id,
         model_type=args.model_type)
