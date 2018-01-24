import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from math import ceil as ceil
import tensorflow as tf

class StructInputRNAcompeteS(object):
    """The deepbind_CNN model input with structure"""

    def __init__(self, config, inf, validation=False, fold_id=1):
        self.folds = folds = config.folds
        (data_one_hot_training, labels_training,
         data_one_hot_test, labels_test,
         structures_training, structures_test,
         training_cases, test_cases,
         seq_length) = (inf["data_one_hot_training"], inf["labels_training"],
                        inf["data_one_hot_test"], inf["labels_test"],
                        inf["structures_train"], inf["structures_test"],
                        inf["training_cases"], inf["test_cases"],
                        inf["seq_length"])
        labels_training = (labels_training - np.mean(labels_training)) / np.sqrt(np.var(labels_training))
        labels_test = (labels_test - np.mean(labels_test)) / np.sqrt(np.var(labels_test))
        self.training_cases = int(training_cases * config.training_frac)
        self.test_cases = int(test_cases * config.test_frac)
        train_index = range(self.training_cases)
        validation_index = range(self.test_cases)
        if validation:
            kf = KFold(n_splits=folds)
            indices = kf.split(np.random.choice(training_cases, replace=False, size=self.training_cases))
            check = 1
            for train_idx, val_idx in indices:
                if(check == fold_id):
                    train_index = train_idx
                    validation_index = val_idx
                    break
                check += 1
            self.training_data = data_one_hot_training[train_index]
            self.test_data = data_one_hot_training[validation_index]
            self.training_labels = labels_training[train_index]
            self.test_labels = labels_training[validation_index]
            self.training_struct = np.transpose(structures_training[train_index],[0,2,1])
            self.test_struct = np.transpose(structures_training[validation_index],[0,2,1])
            self.training_lens = np.array(inf['seq_len_train'],np.int32)[train_index]
            self.test_lens = np.array(inf['seq_len_test'],np.int32)[validation_index]
        else:
            self.training_data = data_one_hot_training[0:self.training_cases]
            self.test_data = data_one_hot_test[0:self.test_cases]
            self.training_labels = labels_training[0:self.training_cases]
            self.test_labels = labels_test[0:self.test_cases]
            self.training_struct = np.transpose(structures_training[0:self.training_cases],[0,2,1])
            self.test_struct = np.transpose(structures_test[0:self.test_cases],[0,2,1])
            self.training_lens = np.array(inf['seq_len_train'], np.int32)[0:self.training_cases]
            self.test_lens = np.array(inf['seq_len_test'], np.int32)[0:self.test_cases]

        self.training_data=np.append(self.training_data,self.training_struct,axis=2)
        self.test_data=np.append(self.test_data,self.test_struct,axis=2)
        self.seq_length = int(seq_length)
        self.training_cases = self.training_data.shape[0]
        self.test_cases = self.test_data.shape[0]

def run_epoch_parallel_rnacs(session, models, input_data, config, epoch, train=False, verbose=False, testing=False,
                       scores=False):
    Nbatch_train = int(ceil(input_data.training_cases * 1.0 / config['minib']))
    Nbatch_test = int(ceil(input_data.test_cases * 1.0 / config['minib']))

    training_scores = np.zeros([len(models), input_data.training_cases])
    test_scores = np.zeros([len(models), input_data.test_cases])
    minib = config['minib']
    num_models = len(models)
    cost_temp = np.zeros([num_models])
    auc_train = np.zeros([num_models])
    for step in range(Nbatch_train):
        fetches = {}
        feed_dict ={}
        for i, model in enumerate(models):
            feed_dict[model.x] = input_data.training_data[(minib * step): (minib * (step + 1)), :, :]
            feed_dict[model.y_true] = input_data.training_labels[(minib * step): (minib * (step + 1))]
            feed_dict[model.seq_lens] = input_data.training_lens[(minib * step): minib * (step + 1)]
            fetches["cost" + str(i)] = model.cost
            if train:
                fetches["eval_op" + str(i)] = model.train_op
            fetches["predictions" + str(i)] = model.predict_op
        vals = session.run(fetches, feed_dict)
        for j in range(num_models):
            cost_temp[j] += vals["cost"+str(j)]
            training_scores[j, (minib * step): (minib * (step + 1))] = vals['predictions' + str(j)]
    cost_train = cost_temp / Nbatch_train
    for j in range(num_models):
        auc_train[j] = roc_auc_score(input_data.training_labels,training_scores[j,:])
    cost_temp = np.zeros([num_models])
    if testing or scores:
        auc_test = np.zeros([num_models,2])
        for step in range(Nbatch_test):
            feed_dict = {}
            fetches = {}
            for i, model in enumerate(models):
                feed_dict[model.x] = input_data.test_data[(minib * step): (minib * (step + 1)), :, :]
                feed_dict[model.y_true] = input_data.test_labels[(minib * step): (minib * (step + 1))]
                feed_dict[model.seq_lens] = input_data.test_lens[(minib * step): minib * (step + 1)]
                fetches["cost" + str(i)] = model.cost
                fetches["predictions" + str(i)] = model.predict_op
            vals = session.run(fetches, feed_dict)
            for j in range(num_models):
                cost_temp[j] += vals["cost" + str(j)]
                test_scores[j, (minib * step): (minib * (step + 1))] = vals['predictions' + str(j)]
        cost_test = cost_temp / Nbatch_test
        auc_ensemble = roc_auc_score(input_data.test_labels, np.mean(test_scores, axis=0))
        for j in range(num_models):
            auc_test[j] = roc_auc_score(input_data.test_labels, test_scores[j,:])
        if verbose:
            best_model = np.argmin(cost_train)
            print(
                "Epoch:%04d, Train cost(min)=%0.4f, Train pearson=%0.4f, Test cost(min)=%0.4f, Test Pearson(max)=%0.4f Ensemble Pearson=%0.4f" %
                (epoch + 1, cost_train[best_model], auc_train[best_model], cost_test[best_model],
                 auc_test[best_model], auc_ensemble))
            print(["%.4f" % p for p in auc_test])

def train_model_parallel(session, config, models, input_data):
    """Trains a list of models in parallel. Expects a list of inputs of equal length as models. Config file is u """
    epochs = config['epochs']
    test_epochs = epochs // config['test_interval']
    num_models = len(models)
    cost_train = np.zeros([test_epochs, num_models])
    cost_test = np.zeros([test_epochs, num_models])
    pearson_test = np.zeros([test_epochs, num_models])
    session.run(tf.global_variables_initializer())
    for i in range(epochs):
        _ = run_epoch_parallel_rnacs(session, models, input_data, config, i, train=True)
        if i % config['test_interval'] == 0:
            step = i //config['test_interval']
            (cost_train[step], cost_test[step], pearson_test[step]) = \
            run_epoch_parallel_rnacs(session, models, input_data, config, i, train=False,
                               verbose=True, testing = True)

    cost_test = np.transpose(cost_test,[1,0])
    pearson_test = np.transpose(pearson_test,[1,0])
    return (cost_test,pearson_test)

def generate_configs_CNN(num_calibrations, flag='small'):
    configs = []
    for i in range(num_calibrations):
        eta = np.float32(10 ** (np.random.uniform(-2, -6)))
        lam = np.float32(10 ** (np.random.uniform(-3, -6)))
        init_scale = np.float32(10 ** (np.random.uniform(-7, -3)))
        minib = 100
        test_interval = 10
        num_conv_layers = np.random.choice([2, 3])
        filter_lengths = [16 // (2 ** i) for i in range(num_conv_layers)]
        num_filters = [16 * (i + 1) for i in range(num_conv_layers)]
        strides = np.random.choice([1], size=num_conv_layers)
        pool_windows = np.random.choice([1], size=num_conv_layers)
        final_pool = np.random.choice(['max', 'avg', 'max_avg'])
        batchnorm = np.random.choice([True, False])
        temp_config = {'eta_model': eta, 'lam_model': lam, 'minib': minib,
                       'test_interval': test_interval, 'filter_lengths': filter_lengths, 'num_filters': num_filters,
                       'num_conv_layers': num_conv_layers, 'strides': strides,
                       'pool_windows': pool_windows,
                       'batchnorm': batchnorm,
                       'final_pool': final_pool,
                       'init_scale': init_scale, 'flag': flag}

        configs.append(create_config_dict(**temp_config))
    return configs

def generate_configs_RNN(num_calibrations, flag='small'):
    configs = []
    for i in range(num_calibrations):
        eta = np.float32(10 ** (np.random.uniform(-2, -6)))
        momentum = np.float32(np.random.uniform(0.95, 0.99))
        lam = np.float32(10 ** (np.random.uniform(-3, -6)))
        init_scale = np.float32(10 ** (np.random.uniform(-7, -3)))
        minib = 1000
        test_interval = 10
        motif_len = 16
        num_motifs = np.random.choice([8, 16, 24])
        lstm_size = np.random.choice([10, 20, 30])

        temp_config = {'eta_model': eta, 'momentum_model': momentum, 'lam_model': lam, 'minib': minib,
                       'test_interval': test_interval, 'motif_len': motif_len,
                       'lstm_size': lstm_size,
                       'num_motifs': num_motifs, 'init_scale': init_scale, 'flag': flag}

        configs.append(create_config_dict(**temp_config))
    return configs

def create_config_dict(**kwargs):
    config = {}
    config.update(kwargs)
    config['folds'] = 3
    if config['flag']=='large':
        config['epochs'] = 15
        config['early_stop_epochs'] = 15
        config['test_interval'] = 1
    elif config['flag'] == 'medium':
        config['epochs'] = 10
        config['early_stop_epochs'] = 10
        config['test_interval'] = 1
    else:
        config['epochs'] = 10
        config['early_stop_epochs'] = 10
        config['test_interval'] = 1
    return config

