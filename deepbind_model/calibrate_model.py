import tensorflow as tf
import numpy as np
import deepbind_model.utils as utils
import os.path


def calibrate_model(target_protein, num_calibrations,
                    model_type, flag):
    print("Performing %d calibration trials for %s %s model"%(num_calibrations, target_protein,model_type))

    target_file = '../data/rnac/npz_archives/'+str(target_protein)+'.npz'
    if not(os.path.isfile(target_file)):
        utils.load_data(target_id_list=[target_protein])
    inf = np.load(target_file)
    input_configuration = utils.input_config(flag)
    configs = utils.generate_configs(num_calibrations, model_type, flag)

    inputs = []
    inputs.append(utils.Deepbind_input(input_configuration, inf, model_type, validation=True, fold_id=1))
    inputs.append(utils.Deepbind_input(input_configuration, inf, model_type, validation=True, fold_id=2))
    inputs.append(utils.Deepbind_input(input_configuration, inf, model_type, validation=True, fold_id=3))

    folds = configs[0]['folds']
    test_epochs = configs[0]['epochs'] // configs[0]['test_interval']
    best_epoch = np.zeros([num_calibrations])
    test_cost = np.zeros([folds,num_calibrations,test_epochs])
    test_pearson = np.zeros([folds,num_calibrations,test_epochs])

    for fold in range(folds):
        print("Evaluating fold %d"%fold)
        with tf.Graph().as_default():
            models = []
            for i in range(num_calibrations):
                with tf.variable_scope('model' + str(i)):
                    models.append(utils.Deepbind_model(configs[i], inputs[fold], model_type))
            with tf.Session() as session:
                (test_cost[fold,:,:],
                 test_pearson[fold,:,:]) = \
                    utils.train_model_parallel(session, configs[0], models, inputs[fold], early_stop=False)

    test_cost = np.mean(np.transpose(test_cost,[1,2,0]),axis=2)
    test_pearson = np.mean(np.transpose(test_pearson,[1,2,0]),axis=2)
    best_epoch = np.argmin(test_cost,axis=1)

    best_calibration_idx = int(np.argmin(np.min(test_cost,axis=1)))


    best_calibration= configs[best_calibration_idx]
    best_cost = np.min(test_cost[best_calibration_idx])
    best_pearson = np.max(test_pearson[best_calibration_idx])

    best_calibration['early_stop_epochs'] = int(best_epoch[best_calibration_idx])
    utils.save_calibration(target_protein,model_type,flag, best_calibration,
                           best_cost,best_pearson,'../calibrations')
    return (best_calibration, best_cost,best_pearson)
