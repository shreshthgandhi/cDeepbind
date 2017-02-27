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
                    model_type=None, flag=None):
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

    folds = configs[0].folds
    test_epochs = configs[0].epochs // configs[0].test_interval
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

    best_calibration.early_stop_epochs = int(best_epoch[best_calibration_idx])
    utils.save_calibration(target_protein,model_type,flag, best_calibration,
                           best_cost,best_pearson,'../calibrations')
    return (best_calibration, best_cost,best_pearson)
