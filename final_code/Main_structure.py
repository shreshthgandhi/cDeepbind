import tensorflow as tf
import numpy as np
import calibrate_model as calib
import models as utils
import os.path
from datetime import  datetime
import argparse

def main(target_protein=None, model_size_flag =None, model_testing_list=None, num_calibrations=5,recalibrate=False):
    traindir = {}
    for model_type in model_testing_list:
        model_dir = os.path.join('../models/'+target_protein+'/'+model_type, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(model_dir)
        traindir[model_type] = model_dir
    best_config = {}

    for model_type in model_testing_list:
        calib_temp = utils.load_calibration(target_protein, model_type, model_size_flag, '../calibrations')
        if recalibrate:
            best_config[model_type], _, _ = calib.calibrate_model(target_protein,
                                                                  num_calibrations=5,
                                                                  model_type=model_type,
                                                                  flag=model_size_flag)

        elif calib_temp:
            best_config[model_type]=calib_temp
        else:
            best_config[model_type], _, _ = calib.calibrate_model(target_protein,
                                                                  num_calibrations=5,
                                                                  model_type=model_type,
                                                                  flag=model_size_flag)

    target_file = '../data/rnac/npz_archives/' + str(target_protein) + '.npz'
    if not(os.path.isfile(target_file)):
        utils.load_data(target_id_list=[target_protein])
    inf = np.load(target_file)
    models = []
    inputs =[]
    input_data = {}
    num_final_runs = 3
    test_epochs = best_config[model_testing_list[0]].epochs // best_config[model_testing_list[0]].test_interval
    total_num_models = num_final_runs * len(model_testing_list)
    test_pearson = np.zeros([total_num_models,test_epochs])
    test_cost = np.zeros([total_num_models,test_epochs])

    input_config = utils.input_config(model_size_flag)

    with tf.Graph().as_default():
        for i,model_type in enumerate(model_testing_list):
            input_data[model_type] = utils.Deepbind_input(input_config, inf, model_type, validation=False)
            for runs in range(num_final_runs):
                with tf.variable_scope('model'+str(runs+i*num_final_runs)):
                    models.append(utils.Deepbind_model(best_config[model_type],
                                                              input_data[model_type],
                                                              model_type))
                    inputs.append(input_data[model_type])
        with tf.Session()   as session:
            # print("learning_rate=%.6f"% best_config[model_type].eta_model)
            (test_cost,test_pearson) = \
                utils.train_model_parallel(session, best_config[model_type],
                                           models, inputs,
                                           early_stop=False)
            for i,model_type in enumerate(model_testing_list):
                best_model_idx = np.argmin(test_cost[i*num_final_runs:(i+1)*num_final_runs,-1])

                abs_best_model_idx = i*num_final_runs + best_model_idx
                best_model_vars = tf.contrib.framework.get_variables(scope='model' + str(abs_best_model_idx))
                saver = tf.train.Saver(best_model_vars)
                # model_best_id = traindir[model_type] + '/'+target_protein +  'best_model.ckpt'

                saver.save(session,os.path.join(traindir[model_type],target_protein+'_best_model.ckpt'))
            # for runs in range(num_final_runs):
            #     # input_data[model_type] = utils.Deepbind_input(input_config, inf, model_type, validation=False)
            #     # models[model_type].input = input_data[model_type]
            #     # with tf.variable_scope("Model", reuse=True):
            #     #     models[model_type] = utils.Deepbind_model(best_config[model_type],
            #     #                                               input_data[model_type],
            #     #                                               model_type)
            #     with tf.Session() as session:
            #
            #         (best_pearson[runs],
            #         last_pearson[runs],
            #         best_epoch[runs]) = utils.train_model(session,
            #                                           best_config[model_type],
            #                                           models[model_type],
            #                                           early_stop=True)
            #         model_id = '../tmp/'+target_protein+str(runs)+'.ckpt'
            #         saver.save(session, model_id)
                pearson = test_pearson[abs_best_model_idx,-1]
                cost = test_cost[abs_best_model_idx,-1]
                print("Pearson correlation for %s using %s is %.4f"%(target_protein,model_type, test_pearson[abs_best_model_idx,-1]))
                result_id = traindir[model_type]+'/results_final/'+target_protein+str(model_type)
                utils.save_result(target_protein, model_type,model_size_flag,cost,pearson,'../results_final')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=None, type=int, nargs='+')
    parser.add_argument('--protein', default=None)
    parser.add_argument('--model_type', default=None,nargs='+')
    parser.add_argument('--num_calibrations', default=5, type=int)
    parser.add_argument('--model_scale',default=None)
    parser.add_argument('--recalibrate', default=False)
    args = parser.parse_args()
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))
    main( target_protein=args.protein, model_size_flag=args.model_scale,
                  model_testing_list=args.model_type, num_calibrations=args.num_calibrations,
                    recalibrate=args.recalibrate)
    # if testing:
    #     result_file = open('../results_final/summary.tsv', 'w')
    #     heading  = 'Protein\t' + '\t'.join(models) +'\n'
    #     result_file.write(heading)
    #     for protein in targets:
    #         # print(protein)
    #         inf = {}
    #         for model_type in models:
    #             inf[model_type] = np.load('../results_final/'+protein+model_type+'.npz')
    #         # result_file = open('../results/final/summary.log','w')
    #         # result_file.write(heading)
    #         values = protein + '\t'+ '\t'.join([str(inf[model_type]['pearson']) for model_type in models])+'\n'
    #         result_file.write(values)
    #         # for model_type in models:
    #         #     print(model_type)
    #         #     print(inf[model_type]['pearson'])