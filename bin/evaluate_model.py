import argparse
import os.path

import numpy as np
import tensorflow as tf
import yaml

import deepbind_model.utils as utils


def main(target_protein, model_type, evaluation_type, CLIPSEQ_experiment=None):
    if 'full' in target_protein:
        protein_name = target_protein.split('_')[0]
    else:
        protein_name = target_protein
    config = utils.load_calibration({'hp_dir':'hyperparameters','model_type':model_type,'protein':protein_name})
    if not (config):
        print("[!] No trained model to evaluate")
        print("[!] Exiting")
        return -1
    result_file = yaml.load(open('results/' + target_protein + '_' + model_type + '.yml'))
    model_dir = result_file['model_dir']
    model_ensemble_size = result_file['ensemble_size']
    # model_idx = result_file['model_index']
    if evaluation_type == 'CLIPSEQ':
        assert CLIPSEQ_experiment
        # utils.load_data_clipseq_shorter(CLIPSEQ_experiment)
        inf = np.load('data/GraphProt_CLIP_sequences/npz_archives/' + CLIPSEQ_experiment + '.npz')
        input_data = utils.ClipInputStruct(inf)
    else:
        inf = utils.load_data(target_protein)
        input_data = utils.model_input(utils.input_config('large'), inf, model_type, validation=False)
    with tf.Graph().as_default():
        models = []
        for i in range(model_ensemble_size):
            with tf.variable_scope('model' + str(i)):
                models.append(utils.model(config, input_data, model_type))
        # best_model_vars = tf.contrib.framework.get_variables(scope='model' + str(model_idx))
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,
                          os.path.join(model_dir, target_protein + '_best_model.ckpt'))
            if evaluation_type == 'CLIPSEQ':
                auc = utils.run_clip_epoch_shorter(sess, models, input_data, config)
                print(target_protein, CLIPSEQ_experiment, auc)
                result_dict = {'auc': float(auc)}
                save_dir = 'results/'
                print(auc)

                yaml.dump(result_dict,
                          open(os.path.join(save_dir,
                                            target_protein + '_' + CLIPSEQ_experiment + '_' + model_type + '_updated' + '.yml'),
                               'w'))
            else:
                (cost_train, cost_test, pearson_test, pearson_ensemble, cost_ensemble) = utils.run_epoch_parallel(sess, models, input_data, {'minib':2000}, epoch=1, train=False,
                                                         verbose=False, testing=True, scores=False)
                # (cost_train, cost_test, training_pearson, test_pearson, training_scores,
                #  test_scores) = utils.run_epoch_parallel(sess, models, input_data, config, epoch=1, train=False,
                #                                          verbose=False, testing=True, scores=True)

                print("True pearson for %s is %f" % (target_protein, pearson_ensemble))




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
