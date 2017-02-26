import tensorflow as tf
import numpy as np
import scipy as Sci
import sys
from sklearn import cross_validation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import calibrate_model as calib
import models as utils
import os.path
from datetime import  datetime
# %matplotlib inline

def main(target_protein='RNCMPT00168', model_size_flag ='small'):
    calibration_flag = True

    # model_size_flag = 'small'
    model_testing_list = ['CNN_struct', 'CNN']
    traindir = {}
    for model_type in model_testing_list:
        model_dir = os.path.join('../models/'+model_type, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(model_dir)
        traindir[model_type] = model_dir

    if calibration_flag:
        best_config, best_metric = calib.calibrate_model(target_protein,
                                                         num_calibrations=10,
                                                         model_testing_list=model_testing_list,
                                                         flag=model_size_flag)
    # if not(calibration_flag):
    #     config = best_config(target_protein, model_testing_list)
    # load_data_rnac(target_id_list=target_protein)
    # config = best_config
    target_file = '../data/rnac/npz_archives/' + str(target_protein) + '.npz'
    if not(os.path.isfile(target_file)):
        utils.load_data(target_id_list=[target_protein])
    inf = np.load(target_file)
    models = {}
    input_data = {}
    num_final_runs = 3

    input_config = utils.input_config(model_size_flag)

    for model_type in model_testing_list:
        best_pearson = np.zeros([num_final_runs])
        last_pearson = np.zeros([num_final_runs])
        best_epoch = np.zeros([num_final_runs])
        with tf.Graph().as_default():
            input_data[model_type] = utils.Deepbind_input(input_config, inf, model_type, validation=False)
            models = []
            for runs in range(num_final_runs):
                with tf.name_scope('model'+str(runs)):
                    models.append(utils.Deepbind_model(best_config[model_type],
                                                              input_data[model_type],
                                                              model_type))
            # with tf.variable_scope("Model", reuse=True):
            #     models[model_type] = utils.Deepbind_model(best_config[model_type],
            #                                               input_data[model_type],
            #                                               model_type)

            with tf.Session()    as session:
                (best_pearson, last_pearson, best_epoch) = \
                    utils.train_model_parallel(session, best_config[model_type],
                                               models, input_data[model_type],
                                               early_stop=True)
                best_model = np.argmax(last_pearson)
                best_model_vars = tf.contrib.framework.get_variables(scope='model' + str(best_model))
                saver = tf.train.Saver(best_model_vars)
                model_best_id = traindir[model_type] + '/'+target_protein +  'best_model.ckpt'
                saver.save(session,model_best_id)
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

            print("Pearson correlation for %s using %s is %.4f"%(target_protein,model_type, np.max(last_pearson)))
            result_id = traindir[model_type]+'/results_final/'+target_protein+str(model_type)
            os.makedirs(traindir[model_type]+'/results_final/')
            np.savez(result_id, pearson = np.max(last_pearson))




            # with tf.Session() as sess:
            #     # saver = tf.train.Saver()
            #     saver.restore(sess, model_best_id)
            #     model_final_location = '../models/' + target_protein+'.ckpt'
            #     saver.save(sess, model_final_location)


        # saver = tf.train.Saver()
        # for model in model_testing_list:
        #     with tf.Session() as session:
        #         session.run(tf.initialize_all_variables())
        #         for i in range(config.epochs):
        #             _ = run_epoch(session,models[model],i, eval_op=models[model].train_op)
        #             if i%config.test_interval == 0:
        #                 (cost_train, cost_test, pearson_test) = run_epoch(session,
        #                                                                 models[model],
        #                                                                 i,
        #                                                                 verbose=True,
        #                                                                 testing=True)
        #         print("Saving model")
        #         saver.save(session, 'model/'+target_protein[0], global_step = i)
        #         result_id = 'results/'+target_protein[0]+'_'+str(i)
        #         
if __name__ == "__main__":
    # main()
    # targets = ['RNCMPT00168', 'RNCMPT00076', 'RNCMPT00268', 'RNCMPT00038', 'RNCMPT00111']
    # targets = ['RNCMPT00100',
    #                  'RNCMPT00101',
    #                  'RNCMPT00102',
    #                  'RNCMPT00103',
    #                  'RNCMPT00104',
    #                  'RNCMPT00105',
    #                  'RNCMPT00106',
    #                  'RNCMPT00107',
    #                  'RNCMPT00108',
    #                  'RNCMPT00109',
    #                  'RNCMPT00010',
    #                  'RNCMPT00110',
    #                  'RNCMPT00111',
    #                  'RNCMPT00112',
    #                  'RNCMPT00113',
    #                  'RNCMPT00114',
    #                  'RNCMPT00116',
    #                  'RNCMPT00117',
    #                  'RNCMPT00118',
    #                  'RNCMPT00119',
    #                  'RNCMPT00011',
    #                  'RNCMPT00120',
    #                  'RNCMPT00121',
    #                  'RNCMPT00122',
    #                  'RNCMPT00123',
    #                  'RNCMPT00124',
    #                  'RNCMPT00126',
    #                  'RNCMPT00127',
    #                  'RNCMPT00129',
    #                  'RNCMPT00012',
    #                  'RNCMPT00131',
    #                  'RNCMPT00132',
    #                  'RNCMPT00133',
    #                  'RNCMPT00134',
    #                  'RNCMPT00136',
    #                  'RNCMPT00137',
    #                  'RNCMPT00138',
    #                  'RNCMPT00139',
    #                  'RNCMPT00013',
    #                  'RNCMPT00140',
    #                  'RNCMPT00141',
    #                  'RNCMPT00142',
    #                  'RNCMPT00143',
    #                  'RNCMPT00144',
    #                  'RNCMPT00145',
    #                  'RNCMPT00146',
    #                  'RNCMPT00147',
    #                  'RNCMPT00148',
    #                  'RNCMPT00149',
    #                  'RNCMPT00014',
    #                  'RNCMPT00150',
    #                  'RNCMPT00151',
    #                  'RNCMPT00152',]
    targets = ['RNCMPT00158']
    models = ['CNN_struct', 'CNN']
    testing = True
    training = True
    if training:
        for protein in targets:
            main( target_protein=protein, model_size_flag='small')
    if testing:
        result_file = open('../results_final/summary.tsv', 'w')
        heading  = 'Protein\t' + '\t'.join(models) +'\n'
        result_file.write(heading)
        for protein in targets:
            # print(protein)
            inf = {}
            for model_type in models:
                inf[model_type] = np.load('../results_final/'+protein+model_type+'.npz')
            # result_file = open('../results/final/summary.log','w')
            # result_file.write(heading)
            values = protein + '\t'+ '\t'.join([str(inf[model_type]['pearson']) for model_type in models])+'\n'
            result_file.write(values)
            # for model_type in models:
            #     print(model_type)
            #     print(inf[model_type]['pearson'])