import argparse
import os.path

import numpy as np
import tensorflow as tf
import yaml

import deepbind_model.utils as utils


def main(target_protein, model_type, evaluation_type, CLIPSEQ_experiment=None):
    config = utils.load_calibration(target_protein, model_type, 'small', '../calibrations')
    if not (config):
        print("[!] No trained model to evaluate")
        print("[!] Exiting")
        return -1
    result_file = yaml.load(open('../results_final/' + target_protein + '_' + model_type + '_large.yml'))
    model_dir = result_file['model_dir']
    model_idx = result_file['model_index']
    if evaluation_type == 'CLIPSEQ':
        inf = np.load('../data/GraphProt_CLIP_sequences/npz_archives/' + CLIPSEQ_experiment + '.npz')
        input_data = utils.Deepbind_clip_input_struct(inf)
    else:
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
            if evaluation_type == 'CLIPSEQ':
                # auc = utils.run_clip_epoch_parallel(sess, [model], input_data, config)
                auc = utils.run_clip_epoch_shorter(sess, [model], input_data, config)
                print(target_protein, CLIPSEQ_experiment, auc)
                result_dict = {'auc': float(auc)}
                save_dir = '../results_final/'
                print(auc)

                yaml.dump(result_dict,
                          open(os.path.join(save_dir,
                                            target_protein + '_' + CLIPSEQ_experiment + '_' + model_type + '_updated' + '.yml'),
                               'w'))
            else:

                # grad_test, predictions_test = utils.compute_gradient(sess, model, input_data, config)
                # np.savez(model_dir + protein_id + 'gradient_scores.npz', gradients=grad_test,
                #          predictions=predictions_test, labels=input_data.test_labels, test_seqs=input_data.test_data)

                (cost_train, cost_test, training_pearson, test_pearson, training_scores,
                 test_scores) = utils.run_epoch_parallel(sess, [model], input_data, config, epoch=1, train=False,
                                                         verbose=False, testing=True, scores=True)

                print("True pearson for %s is %f" % (target_protein, test_pearson[0, 0]))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=None, type=int, nargs='+')
    parser.add_argument('--protein', default=None, nargs='+')
    parser.add_argument('--model_type', default=None)
    parser.add_argument('--evaluation_type', default='RNAC_2013')
    parser.add_argument('--CLIPSEQ_experiment', default=None, nargs='+')
    args = parser.parse_args()
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))
    if args.CLIPSEQ_experiment:
        for protein_id, clip_exp in zip(args.protein, args.CLIPSEQ_experiment):
            main(target_protein=protein_id,
                 model_type=args.model_type,
                 evaluation_type=args.evaluation_type,
                 CLIPSEQ_experiment=clip_exp)
    else:
        for protein_id in args.protein:
            main(target_protein=protein_id,
                 model_type=args.model_type,
                 evaluation_type=args.evaluation_type,
                 CLIPSEQ_experiment=args.CLIPSEQ_experiment)
