import argparse
import os.path

import numpy as np
import tensorflow as tf
import yaml

import deepbind_model.utils as utils


def main(target_protein, model_type):
    config = utils.load_calibration(target_protein, model_type, 'small', '../calibrations')
    result_file = yaml.load(open('../results_final/' + target_protein + '_' + model_type + '_large.yml'))
    target_file = '../data/rnac/npz_archives/' + str(target_protein) + '.npz'
    model_dir = result_file['model_dir']
    model_idx = result_file['model_index']
    if not (os.path.isfile(target_file)):
        utils.load_data(target_id_list=[target_protein])
    inf = np.load(target_file)
    input_data = utils.Deepbind_input(utils.input_config('large'), inf, model_type, validation=False)
    with tf.Graph().as_default():
        with tf.variable_scope('model' + str(model_idx)):
            model = utils.Deepbind_model(config, input_data, model_type)
        best_model_vars = tf.contrib.framework.get_variables(scope='model' + str(model_idx))
        saver = tf.train.Saver(best_model_vars)
        with tf.Session() as sess:
            saver.restore(sess,
                          os.path.join(model_dir, target_protein + '_best_model.ckpt'))
            scores = utils.evaluate_model_parallel(sess, config, [model], input_data)
        print(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=None, type=int, nargs='+')
    parser.add_argument('--protein', default=None)
    parser.add_argument('--model_type', default=None)

    args = parser.parse_args()
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))
    main(target_protein=args.protein,
         model_type=args.model_type)
