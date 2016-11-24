import tensorflow as tf
import numpy as np
import models as utils
import os.path


#
# class Config_class(object):
#     """Generates configuration"""
#     def __init__(self, eta=0.01, momentum=0.9, lam=0.00001,
#                  minib=100, test_interval=10,
#                  motif_len=16, num_motifs=16, init_scale=0.01, flag='small'):
#         self.eta_model = eta
#         self.momentum_model = momentum
#         self.lam_model = lam
#         # self.epochs = epochs
#         self.minib = minib
#         self.test_interval = test_interval
#         self.motif_len = motif_len
#         self.num_motifs = num_motifs
#         self.init_scale = init_scale
#         self.folds = 3
#         if flag == 'large':
#             self.training_frac = 1
#
#             self.test_frac = 1
#             self.epochs = 300
#             self.early_stop_epochs = 300
#         elif flag == 'medium':
#             self.training_frac = 0.5
#             self.test_frac = 0.5
#             self.epochs = 200
#             self.early_stop_epochs = 200
#
#         else:
#             self.training_frac = 0.1
#             self.test_frac = 0.1
#             self.epochs = 100
#             self.early_stop_epochs = 100
#
# class input_config(object):
#     """Generates configuration for processing input to model"""
#     def ___init___(self, flag):
#         self.folds = 3
#         if flag == 'large':
#             self.training_frac = 1
#             self.test_frac  = 1
#         elif flag == 'medium':
#             self.training_frac = 0.5
#             self.test_frac = 0.5
#         else:
#             self.training_frac = 0.1
#             self.test_frac = 0.1
#
#
#
#
# def generate_configs_CNN(num_calibrations, flag='small'):
#     configs = []
#     for i in range(num_calibrations):
#         eta = np.float32(10**(np.random.uniform(-1,-4)))
#         momentum = np.float32(np.random.uniform(0.95,0.99))
#         lam = np.float32(10**(np.random.uniform(-3,-10)))
#         init_scale = np.float32(10**(np.random.uniform(-7,-3)))
#         minib = 100
#         test_interval = 10
#         motif_len = 16
#         num_motifs = 16
#         configs.append = Config_class(eta,momentum,lam,minib,test_interval,motif_len,num_motifs,init_scale,flag)
#     return configs
#
# def generate_configs_CNN_struct(num_calibrations, flag='small'):
#     configs = []
#     for i in range(num_calibrations):
#         eta = np.float32(10**(np.random.uniform(-1,-4)))
#         momentum = np.float32(np.random.uniform(0.95,0.99))
#         lam = np.float32(10**(np.random.uniform(-3,-10)))
#         init_scale = np.float32(10**(np.random.uniform(-7,-3)))
#         minib = 100
#         test_interval = 10
#         motif_len = 16
#         num_motifs = 16
#         configs.append = Config_class(eta,momentum,lam,minib,test_interval,motif_len,num_motifs,init_scale,flag)
#     return configs
#
# def generate_configs(num_calibrations, model_type, flag='small'):
#     if model_type=='CNN':
#         return generate_configs_CNN(num_calibrations, flag)
#     if model_type=='CNN_struct':
#         return generate_configs_CNN_struct(num_calibrations, flag)
#
# def train_model(config, model):
#     # with tf.Graph().as_default():
#     test_epochs = config.epochs // config.test_interval
#     cost_train = np.zeros([test_epochs])
#     cost_test = np.zeros([test_epochs])
#     pearson_test = np.zeros([test_epochs])
#     with tf.Session() as session:
#         session.run(tf.initialize_all_variables())
#         for i in range(config.epochs):
#             _ = utils.run_epoch(session, model, i, eval_op=model.train_op)
#             if i % config.test_interval == 0:
#                 step = i // config.test_interval
#                 (cost_train[step], cost_test[step], pearson_test[step]) = \
#                     utils.run_epoch(session, model, i, verbose=True, testing=True)
#     best_epoch = np.argmax(pearson_test) * config.test_interval
#     best_pearson = np.max(pearson_test)
#     last_pearson = pearson_test[-1]
#     return (best_pearson, last_pearson, best_epoch)

def calibrate_model(target_protein='RNCMPT00168', num_calibrations=5,
                    model_testing_list=['CNN_struct'], flag='small'):
    print("Performing %d calibration trials for %s"%(num_calibrations, target_protein))

    # utils.load_data(target_id_list=[target_protein])
    target_file = '../data/rnac/npz_archives/'+str(target_protein)+'.npz'
    if not(os.path.isfile(target_file)):
        utils.load_data(target_id_list=[target_protein])
    inf = np.load(target_file)
    config_lists = {}
    input_lists = {}
    input_configuration = utils.input_config(flag)
    best_pearson = {}
    last_pearson = {}
    best_epoch = {}
    best_calib_idx = {}
    best_epoch_final = {}
    for model in model_testing_list:
        config_lists[model] = utils.generate_configs(num_calibrations, model, flag)
    for model in model_testing_list:
        inputs = []
        inputs.append(utils.Deepbind_input(input_configuration, inf, model, validation=True, fold_id=1))
        inputs.append(utils.Deepbind_input(input_configuration, inf, model, validation=True, fold_id=2))
        inputs.append(utils.Deepbind_input(input_configuration, inf, model, validation=True, fold_id=3))
        input_lists[model] = inputs
    for model_type in model_testing_list:
        best_pearson[model_type] = np.zeros([num_calibrations, config_lists[model_type][0].folds])
        last_pearson[model_type] = np.zeros([num_calibrations, config_lists[model_type][0].folds])
        best_epoch[model_type] = np.zeros([num_calibrations, config_lists[model_type][0].folds])

        configs = config_lists[model_type]  #For each model type a list of configs
        inputs = input_lists[model_type] #Each is a list containing the 3 folds
        for cal_idx in range(num_calibrations):   #Run 3 folds for each calibration
            config_calib = configs[cal_idx] #Model config for this calibration
            print("%.2f%% calibrations complete for %s"%(((cal_idx+1)*100)/num_calibrations,model_type))
            with tf.Graph().as_default():
                models = {} #Generate 3 models, one for each fold
                for fold_idx in range(config_calib.folds):
                    print("fold %d" %(fold_idx+1) )
                    with tf.variable_scope(model_type, reuse=True): #Have a different variable scope for each model type
                        m = utils.Deepbind_model(config_calib, inputs[fold_idx], model_type)
                    with tf.Session() as session:
                        (best_pearson[model_type][cal_idx][fold_idx],
                        last_pearson[model_type][cal_idx][fold_idx],
                        best_epoch[model_type][cal_idx][fold_idx]) = utils.train_model(session,
                                                                                       config_calib, m)
        best_calib_idx[model_type] = int(np.argmax(np.mean(best_pearson[model_type], axis=1)))
        best_fold = int(np.argmax(best_pearson[model_type][best_calib_idx[model_type]]))
        best_epoch_final[model_type] = best_epoch[model_type][best_calib_idx[model_type]][best_fold]
        # best_pearson[model_type] = np.mean(best_pearson[model_type], axis=1)
        # last_pearson[model_type] = np.mean(last_pearson[model_type], axis=1)
    best_calibrations = {}
    for model_type in model_testing_list:
        best_calibrations[model_type] = config_lists[model_type][best_calib_idx[model_type]]
        best_calibrations[model_type].early_stop_epochs = int(best_epoch_final[model_type])
        best_pearson[model_type] = np.mean(best_pearson[model_type][best_calib_idx[model_type]])
    return (best_calibrations, best_pearson)
