import argparse
import os.path

import numpy as np
import scipy.stats as stats
import tensorflow as tf

import deepbind_model.utils as utils


def main(model_dir, target_protein, model_size_flag, model_type):
    config = utils.load_calibration(target_protein, model_type, 'small', '../calibrations')
    target_file = '../data/rnac/npz_archives/' + str(target_protein) + '.npz'
    if not (os.path.isfile(target_file)):
        utils.load_data(target_id_list=[target_protein])
    inf = np.load(target_file)
    input_data = utils.Deepbind_input(utils.input_config(model_size_flag), inf, model_type, validation=False)
    with tf.Graph().as_default():
        with tf.variable_scope('model1'):
            model = utils.Deepbind_model(config, input_data, model_type)
        best_model_vars = tf.contrib.framework.get_variables(scope='model1')
        saver = tf.train.Saver(best_model_vars)
        data = np.concatenate([inf['data_one_hot_training'], np.transpose(inf['structures_train'], [0, 2, 1])], axis=-1)
        with tf.Session() as sess:
            saver.restore(sess,
                          os.path.join(model_dir, target_protein + '_best_model.ckpt'))
            scores = sess.run(model._predict_op, feed_dict={model._x: data})
        scores_test = inf['labels_training']
        print('Pearson correlation on test-set is', stats.pearsonr(scores, scores_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=None, type=int, nargs='+')
    parser.add_argument('--protein', default=None)
    parser.add_argument('--model_type', default=None)
    parser.add_argument('--model_scale',default=None)
    parser.add_argument('--model_dir', default=None)

    args = parser.parse_args()
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))
    main(model_dir=args.model_dir, target_protein=args.protein, model_size_flag=args.model_scale,
         model_type=args.model_type)
