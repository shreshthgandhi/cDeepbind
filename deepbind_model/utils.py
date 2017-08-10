import os.path
from math import ceil as ceil

import numpy as np
import scipy.stats as stats
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


class Deepbind_clip_input_struct(object):
    def __init__(self, inf):
        (data_one_hot, labels,
         structures,
         total_cases, seq_length) = (inf["data_one_hot"], inf["labels"],
                                     inf["structures"], inf["total_cases"],
                                     inf["seq_length"])
        self.data = np.append(data_one_hot, np.transpose(structures, [0, 2, 1]), axis=2)
        self.seq_len = seq_length
        self.total_cases = self.data.shape[0]
        self.labels = labels

class Deepbind_no_struct_input(object):
    """The deepbind_CNN model input without structure"""

    def __init__(self, config, inf, validation=False, fold_id=1):
        self.folds = folds = config.folds
        (data_one_hot_training, labels_training,
         data_one_hot_test, labels_test,
         training_cases, test_cases,
         seq_length) = (inf["data_one_hot_training"], inf["labels_training"],
                        inf["data_one_hot_test"], inf["labels_test"],
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
            indices = kf.split(range(self.training_cases))
            check = 1
            for train_idx, val_idx in indices:
                if(check == fold_id):
                    train_index = train_idx
                    validation_index = val_idx
                    break
                check = check + 1
            self.training_data = data_one_hot_training[train_index]
            self.test_data = data_one_hot_training[validation_index]
            self.training_labels = labels_training[train_index]
            self.test_labels = labels_training[validation_index]
        else:
            self.training_data = data_one_hot_training[0:self.training_cases]
            self.test_data = data_one_hot_test[0:self.test_cases]
            self.training_labels = labels_training[0:self.training_cases]
            self.test_labels = labels_test[0:self.test_cases]
#
        self.seq_length = int(seq_length)
        self.training_cases = self.training_data.shape[0]
        self.test_cases = self.test_data.shape[0]


class Deepbind_struct_input(object):
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
        else:
            self.training_data = data_one_hot_training[0:self.training_cases]
            self.test_data = data_one_hot_test[0:self.test_cases]
            self.training_labels = labels_training[0:self.training_cases]
            self.test_labels = labels_test[0:self.test_cases]
            self.training_struct = np.transpose(structures_training[0:self.training_cases],[0,2,1])
            self.test_struct = np.transpose(structures_test[0:self.test_cases],[0,2,1])
        self.training_data=np.append(self.training_data,self.training_struct,axis=2)
        self.test_data=np.append(self.test_data,self.test_struct,axis=2)
        self.seq_length = int(seq_length)
        self.training_cases = self.training_data.shape[0]
        self.test_cases = self.test_data.shape[0]

def Deepbind_input(input_config,inf,model,validation=False,fold_id=1):
    if 'struct' in model or 'STRUCT' in model:
        return Deepbind_struct_input(input_config, inf, validation, fold_id)
    else:
        return Deepbind_no_struct_input(input_config, inf, validation, fold_id)

class Deepbind_CNN_model(object):
    """The deepbind_CNN model without structure"""

    def __init__(self, config, input_):
        self._config = config
        self._input = input_
        self.weight_initializer = tf.truncated_normal_initializer(stddev=config['init_scale'])
        self.rna_sequence = tf.placeholder(tf.float32, shape=[None, None, 4], name='input_sequence')
        self.target_scores = tf.placeholder(tf.float32, shape=[None], name='target_scores')
        self.target_scores_exp = tf.expand_dims(self.target_scores, 1)
        conv_input = self.rna_sequence
        for layer in range(config['num_conv_layers']):
            if layer == (config['num_conv_layers'] - 1):
                self.conv_output = tf.layers.conv1d(inputs=conv_input, filters=1,
                                                    kernel_size=config['filter_lengths'][layer],
                                                    strides=config['strides'][layer],
                                                    padding='SAME', activation=None,
                                                    kernel_initializer=self.weight_initializer,
                                                    name='conv_layer_' + str(layer))
            else:
                self.conv_output = tf.layers.conv1d(inputs=conv_input, filters=config['num_filters'][layer],
                                                    kernel_size=config['filter_lengths'][layer],
                                                    strides=config['strides'][layer],
                                                    padding='SAME', activation=tf.nn.relu,
                                                    kernel_initializer=self.weight_initializer,
                                                    name='conv_layer_' + str(layer))

            conv_input = self.conv_output
        self.conv_output = tf.squeeze(self.conv_output, axis=2)
        final_pool = config.get('final_pool', 'max')
        if final_pool == 'max':
            self.target_predictions = tf.reduce_max(self.conv_output, axis=[1], name='max_pool', keep_dims=True)
        if final_pool == 'avg':
            self.target_predictions = tf.reduce_mean(self.conv_output, axis=[1], name='avg_pool', keep_dims=True)
        if final_pool == 'max_avg':
            max_pool = tf.reduce_max(self.conv_output, axis=[1], name='max_pool', keep_dims=True)
            avg_pool = tf.reduce_mean(self.conv_output, axis=[1], name='avg_pool', keep_dims=True)
            self.target_predictions = tf.layers.dense(tf.concat([max_pool, avg_pool], axis=1), units=1,
                                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                          scale=config['lam_model']),
                                                      name='target_prediction')
        self.loss = tf.losses.mean_squared_error(self.target_scores_exp, self.target_predictions,
                                                 scope='mean_squared_error')
        self._train_op = tf.contrib.layers.optimize_loss(self.loss, tf.contrib.framework.get_global_step(),
                                                         learning_rate=tf.constant(config['eta_model'], tf.float32),
                                                         optimizer='Adam',
                                                         clip_gradients=config.get('gradient_clip_value', 20.0),
                                                         name='train_op')

    @property
    def input(self):
        return self._input

    @property
    def config(self):
        return self._config

    @property
    def cost(self):
        return self.loss

    @property
    def train_op(self):
        return self._train_op

    @property
    def predict_op(self):
        return tf.squeeze(self.target_predictions)

    @property
    def x(self):
        return self.rna_sequence

    @property
    def y_true(self):
        return self.target_scores


class Deepbind_CNN_struct_model(object):
    """The deepbind_CNN model with structure"""

    def __init__(self, config, input_):
        self._config = config
        self._input = input_
        self.weight_initializer = tf.truncated_normal_initializer(stddev=config['init_scale'])
        self.rna_sequence = tf.placeholder(tf.float32, shape=[None, None, 9], name='input_sequence')
        self.target_scores = tf.placeholder(tf.float32, shape=[None], name='target_scores')
        self.target_scores_exp = tf.expand_dims(self.target_scores, 1)
        conv_input = self.rna_sequence
        for layer in range(config['num_conv_layers']):
            if layer == (config['num_conv_layers'] - 1):
                self.conv_output = tf.layers.conv1d(inputs=conv_input, filters=1,
                                                    kernel_size=config['filter_lengths'][layer],
                                                    strides=config['strides'][layer],
                                                    padding='SAME', activation=None,
                                                    kernel_initializer=self.weight_initializer,
                                                    name='conv_layer_' + str(layer))
            else:
                self.conv_output = tf.layers.conv1d(inputs=conv_input, filters=config['num_filters'][layer],
                                                    kernel_size=config['filter_lengths'][layer],
                                                    strides=config['strides'][layer],
                                                    padding='SAME', activation=tf.nn.relu,
                                                    kernel_initializer=self.weight_initializer,
                                                    name='conv_layer_' + str(layer))

            conv_input = self.conv_output
        self.conv_output = tf.squeeze(self.conv_output, axis=2)
        final_pool = config.get('final_pool', 'max')
        if final_pool == 'max':
            self.target_predictions = tf.reduce_max(self.conv_output, axis=[1], name='max_pool', keep_dims=True)
        if final_pool == 'avg':
            self.target_predictions = tf.reduce_mean(self.conv_output, axis=[1], name='avg_pool', keep_dims=True)
        if final_pool == 'max_avg':
            max_pool = tf.reduce_max(self.conv_output, axis=[1], name='max_pool', keep_dims=True)
            avg_pool = tf.reduce_mean(self.conv_output, axis=[1], name='avg_pool', keep_dims=True)
            self.target_predictions = tf.layers.dense(tf.concat([max_pool, avg_pool], axis=1), units=1,
                                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                          scale=config['lam_model']),
                                                      name='target_prediction')
        self.loss = tf.losses.mean_squared_error(self.target_scores_exp, self.target_predictions,
                                                 scope='mean_squared_error')
        self._train_op = tf.contrib.layers.optimize_loss(self.loss, tf.contrib.framework.get_global_step(),
                                                         learning_rate=tf.constant(config['eta_model'], tf.float32),
                                                         optimizer='Adam',
                                                         clip_gradients=config.get('gradient_clip_value', 20.0),
                                                         name='train_op')

    @property
    def input(self):
        return self._input

    @property
    def config(self):
        return self._config

    @property
    def cost(self):
        return self.loss

    @property
    def train_op(self):
        return self._train_op

    @property
    def predict_op(self):
        return tf.squeeze(self.target_predictions)

    @property
    def x(self):
        return self.rna_sequence

    @property
    def y_true(self):
        return self.target_scores


# class Deepbind_RNN_struct_model(object):
#     """The deepbind_RNN model with structure"""
#
#     def __init__(self, config, input_):
#         self._config = config
#         self._input = input_
#         self.weight_initializer = tf.truncated_normal_initializer(stddev=config['init_scale'])
#         self.rna_sequence = tf.placeholder(tf.float32, shape=[None, None, 9], name='input_sequence')
#         self.target_scores = tf.placeholder(tf.float32, shape=[None], name='target_scores')
#         self.target_scores_exp = tf.expand_dims(self.target_scores, 1)
#         conv_input = self.rna_sequence
#         for layer in range(config['num_conv_layers']):
#             if layer == (config['num_conv_layers'] - 1):
#                 self.conv_output = tf.layers.conv1d(inputs=conv_input, filters=config['num_filters'][layer],
#                                                     kernel_size=config['filter_lengths'][layer],
#                                                     strides=config['strides'][layer],
#                                                     padding='SAME', activation=None,
#                                                     kernel_initializer=self.weight_initializer,
#                                                     name='conv_layer_' + str(layer))
#             else:
#                 self.conv_output = tf.layers.conv1d(inputs=conv_input, filters=config['num_filters'][layer],
#                                                     kernel_size=config['filter_lengths'][layer],
#                                                     strides=config['strides'][layer],
#                                                     padding='SAME', activation=tf.nn.relu,
#                                                     kernel_initializer=self.weight_initializer,
#                                                     name='conv_layer_' + str(layer))
#
#             conv_input = self.conv_output
#         if config.get('bidirectional_LSTM', False):
#             lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=config['lstm_size'],
#                                                    initializer=self.weight_initializer,
#                                                    )
#             lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=config['lstm_size'],
#                                                    initializer=self.weight_initializer,
#                                                    )
#             ((output_fw, output_bw), state) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
#                                                                               self.conv_output, dtype=tf.float32,
#                                                                               scope='bidirectional_lstm')
#             self.lstm_output = tf.concat([output_fw[:, -1, :], output_bw[:, -1, :]], 1, name='concatenated_lstm_output')
#         else:
#             lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=config['lstm_size'],
#                                                    initializer=self.weight_initializer,
#                                                    )
#             output, state = tf.nn.dynamic_rnn(lstm_fw_cell, self.conv_output, dtype=tf.float32,
#                                               scope='unidirectional_lstm')
#             self.lstm_output = output[:, -1, :]
#         self.target_predictions = tf.layers.dense(self.lstm_output, units=1,
#                                                   kernel_regularizer=
#                                                   tf.contrib.layers.l2_regularizer(scale=config['lam_model']),
#                                                   name='target_prediction')
#         self.loss = tf.losses.mean_squared_error(self.target_scores_exp, self.target_predictions,
#                                                  scope='mean_squared_error')
#         self._train_op = tf.contrib.layers.optimize_loss(self.loss, tf.contrib.framework.get_global_step(),
#                                                          learning_rate=tf.constant(config['eta_model'], tf.float32),
#                                                          optimizer='Adam',
#                                                          clip_gradients=config.get('gradient_clip_value', 20.0),
#                                                          name='train_op')
#
#     @property
#     def input(self):
#         return self._input
#
#     @property
#     def config(self):
#         return self._config
#
#     @property
#     def cost(self):
#         return self.loss
#
#     @property
#     def train_op(self):
#         return self._train_op
#
#     @property
#     def predict_op(self):
#         return tf.squeeze(self.target_predictions)
#
#     @property
#     def x(self):
#         return self.rna_sequence
#
#     @property
#     def y_true(self):
#         return self.target_scores

# class Deepbind_RNN_struct_model(object):
#     """The deepbind_RNN model with structure"""
#
#     def __init__(self, config, input_):
#         self._config = config
#         eta_model = config['eta_model']
#         momentum_model = config['momentum_model']
#         lam_model = config['lam_model']
#         seq_length = input_.seq_length
#
#         self.motif_len = config['motif_len']  # Tunable Motif length
#         self.num_motifs = config['num_motifs']  # Number of tunable motifs
#         self.motif_len2 = config['motif_len']
#         self.num_motifs2 = config['num_motifs']
#         self._init_op = tf.global_variables_initializer()
#
#         self._x = x = tf.placeholder(tf.float32, shape=[None, seq_length, 9], name='One_hot_data')
#         self._y_true = y_true = tf.placeholder(tf.float32, shape=[None], name='Labels')
#
#         x_image = tf.reshape(x, [-1, seq_length, 1, 9])
#
#         W_conv1 = tf.Variable(tf.random_normal([self.motif_len, 1, 9, self.num_motifs], stddev=0.01), name='W_Conv1')
#         b_conv1 = tf.Variable(tf.constant(0.001, shape=[self.num_motifs]), name='b_conv1')
#
#         h_conv1 = tf.nn.conv2d(x_image, W_conv1,
#                                strides=[1, 1, 1, 1], padding='SAME')
#         h_relu_conv1 = tf.nn.relu(h_conv1 + b_conv1, name='First_layer_output')
#         W_conv2 = tf.Variable(tf.random_normal([self.motif_len2, 1, self.num_motifs2, 1]), name='W_conv2')
#         b_conv2 = tf.Variable(tf.constant(0.001, shape=[1]), name='b_conv2')
#         h_conv2 = tf.nn.conv2d(h_relu_conv1, W_conv2,
#                                strides=[1, 1, 1, 1], padding='SAME')
#         n_hidden = 10
#         W_hidden = tf.Variable(tf.random_normal([1, n_hidden]), name='W_hidden')
#         b_hidden = tf.Variable(tf.constant(0.001, shape=[n_hidden]), name='b_hidden')
#         W_out = tf.Variable(tf.random_normal([n_hidden, 1]), name='W_hidden')
#         b_out = tf.Variable(tf.constant(0.001, shape=[1]), name='b_hidden')
#
#         h_input = tf.reshape(tf.squeeze(h_conv2, axis=[3]), [-1, 1])
#         h_input = tf.matmul(h_input, W_hidden)
#         h_input = tf.reshape(h_input, [-1, seq_length, n_hidden])
#         # h_input = tf.unstack(value=h_input,axis=1)
#         lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
#         outputs, state = tf.nn.dynamic_rnn(lstm_cell, h_input, dtype=tf.float32)
#         h_final = tf.squeeze(
#             tf.matmul(tf.squeeze(tf.slice(outputs, [0, tf.shape(outputs)[1] - 1, 0], [-1, 1, -1])), W_out) + b_out)
#
#         cost_batch = tf.square(h_final - y_true)
#         self._cost = cost = tf.reduce_mean(cost_batch)
#         # tf.scalar_summary("Training Loss", cost)
#         norm_w = (tf.reduce_sum(tf.abs(W_conv1)) + tf.reduce_sum(tf.abs(W_conv2)))
#         # optimizer = tf.train.MomentumOptimizer(learning_rate=eta_model,
#         #                                        momentum=momentum_model)
#         optimizer = tf.train.AdamOptimizer(learning_rate=eta_model)
#
#         self._train_op = optimizer.minimize(cost + norm_w * lam_model)
#         self._predict_op = h_final
#
#         # summaries = []
#         #
#         # summaries.append(tf.summary.scalar('cost', self.cost))
#         # summaries.append(tf.summary.histogram('first_layer', h_relu_conv1))
#         # summaries.append(tf.summary.histogram('final_layer', h_final))
#         #
#         # self.summary_op = tf.summary.merge(summaries)
#
#     def initialize(self, session):
#         session.run(self._init_op)
#
#     @property
#     def input(self):
#         return self._input
#
#     @property
#     def config(self):
#         return self._config
#
#     @property
#     def cost(self):
#         return self._cost
#
#     @property
#     def train_op(self):
#         return self._train_op
#
#     @property
#     def predict_op(self):
#         return self._predict_op
#
#     @property
#     def x(self):
#         return self._x
#
#     @property
#     def y_true(self):
#         return self._y_true


class Deepbind_RNN_struct_model(object):
    """The deepbind_RNN model with structure"""

    def __init__(self, config, input_):
        self._config = config
        eta_model = config['eta_model']
        lam_model = config['lam_model']
        self.motif_len = config['motif_len']  # Tunable Motif length
        self.num_motifs = config['num_motifs']  # Number of tunable motifs
        self.motif_len2 = config['motif_len']
        self._init_op = tf.global_variables_initializer()
        self._x = x = tf.placeholder(tf.float32, shape=[None, None, 9], name='One_hot_data')
        self._y_true = y_true = tf.placeholder(tf.float32, shape=[None], name='Labels')
        x_image = tf.expand_dims(x, 2)

        W_conv1 = tf.Variable(tf.random_normal([self.motif_len, 1, 9, self.num_motifs], stddev=0.01), name='W_Conv1')
        b_conv1 = tf.Variable(tf.constant(0.001, shape=[self.num_motifs]), name='b_conv1')

        h_conv1 = tf.nn.conv2d(x_image, W_conv1,
                               strides=[1, 1, 1, 1], padding='SAME')
        h_relu_conv1 = tf.nn.relu(h_conv1 + b_conv1, name='First_layer_output')
        W_conv2 = tf.Variable(tf.random_normal([self.motif_len2, 1, self.num_motifs, 1]), name='W_conv2')
        b_conv2 = tf.Variable(tf.constant(0.001, shape=[1]), name='b_conv2')
        h_conv2 = tf.nn.conv2d(h_relu_conv1, W_conv2,
                               strides=[1, 1, 1, 1], padding='SAME')
        n_hidden = config.get('lstm_size', 10)
        W_out = tf.Variable(tf.random_normal([n_hidden, 1]), name='W_hidden')
        b_out = tf.Variable(tf.constant(0.001, shape=[1]), name='b_hidden')
        h_input = tf.squeeze(tf.nn.relu(h_conv2 + b_conv2), axis=[3], name='lstm_input')
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs, state = tf.nn.dynamic_rnn(lstm_cell, h_input, dtype=tf.float32)
        # seq_current = tf.shape(outputs)[1]
        # lstm_output_layer = tf.reshape((tf.matmul(tf.reshape(outputs,[-1,n_hidden]),W_out)+b_out),[-1,seq_current,1])
        # h_final = tf.reduce_max(lstm_output_layer, axis=[1,2])
        h_final = tf.squeeze(
            tf.matmul(tf.squeeze(tf.slice(outputs, [0, tf.shape(outputs)[1] - 1, 0], [-1, 1, -1]), axis=[1]),
                      W_out) + b_out)

        cost_batch = tf.square(h_final - y_true)
        self._cost = cost = tf.reduce_mean(cost_batch, name='cost')
        norm_w = (tf.reduce_sum(tf.abs(W_conv1)) + tf.reduce_sum(tf.abs(W_conv2)) + tf.reduce_sum(tf.abs(W_out)))
        optimizer = tf.train.AdamOptimizer(learning_rate=eta_model)

        self._train_op = optimizer.minimize(cost + norm_w * lam_model)
        self._predict_op = h_final
        self.lstm_output_layer = outputs
        self.lstm_state_layer = state
        self.lstm_scalar_weight = W_out
        self.lstm_scalar_bias = b_out

    def initialize(self, session):
        session.run(self._init_op)

    @property
    def input(self):
        return self._input

    @property
    def config(self):
        return self._config

    @property
    def cost(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op

    @property
    def predict_op(self):
        return self._predict_op

    @property
    def x(self):
        return self._x

    @property
    def y_true(self):
        return self._y_true

class Deepbind_RNN_model(object):
    """The deepbind_RNN model with structure"""

    def __init__(self, config, input_):
        self._config = config
        eta_model = config['eta_model']
        lam_model = config['lam_model']
        self.motif_len = config['motif_len']  # Tunable Motif length
        self.num_motifs = config['num_motifs']  # Number of tunable motifs
        self.motif_len2 = config['motif_len']
        self._init_op = tf.global_variables_initializer()
        self._x = x = tf.placeholder(tf.float32, shape=[None, None, 4], name='One_hot_data')
        self._y_true = y_true = tf.placeholder(tf.float32, shape=[None], name='Labels')
        x_image = tf.expand_dims(x, 2)

        W_conv1 = tf.Variable(tf.random_normal([self.motif_len, 1, 4, self.num_motifs], stddev=0.01), name='W_Conv1')
        b_conv1 = tf.Variable(tf.constant(0.001, shape=[self.num_motifs]), name='b_conv1')

        h_conv1 = tf.nn.conv2d(x_image, W_conv1,
                               strides=[1, 1, 1, 1], padding='SAME')
        h_relu_conv1 = tf.nn.relu(h_conv1 + b_conv1, name='First_layer_output')
        W_conv2 = tf.Variable(tf.random_normal([self.motif_len2, 1, self.num_motifs, 1]), name='W_conv2')
        b_conv2 = tf.Variable(tf.constant(0.001, shape=[1]), name='b_conv2')
        h_conv2 = tf.nn.conv2d(h_relu_conv1, W_conv2,
                               strides=[1, 1, 1, 1], padding='SAME')
        n_hidden = config.get('lstm_size', 20)
        W_out = tf.Variable(tf.random_normal([n_hidden, 1]), name='W_hidden')
        b_out = tf.Variable(tf.constant(0.001, shape=[1]), name='b_hidden')
        h_input = tf.squeeze(tf.nn.relu(h_conv2 + b_conv2), axis=[3], name='lstm_input')
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs, state = tf.nn.dynamic_rnn(lstm_cell, h_input, dtype=tf.float32)
        h_final = tf.squeeze(
            tf.matmul(tf.squeeze(tf.slice(outputs, [0, tf.shape(outputs)[1] - 1, 0], [-1, 1, -1])), W_out) + b_out)

        cost_batch = tf.square(h_final - y_true)
        self._cost = cost = tf.reduce_mean(cost_batch, name='cost')
        norm_w = (tf.reduce_sum(tf.abs(W_conv1)) + tf.reduce_sum(tf.abs(W_conv2)) + tf.reduce_sum(tf.abs(W_out)))
        optimizer = tf.train.AdamOptimizer(learning_rate=eta_model)

        self._train_op = optimizer.minimize(cost + norm_w * lam_model)
        self._predict_op = h_final

    def initialize(self, session):
        session.run(self._init_op)

    @property
    def input(self):
        return self._input

    @property
    def config(self):
        return self._config

    @property
    def cost(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op

    @property
    def predict_op(self):
        return self._predict_op

    @property
    def x(self):
        return self._x

    @property
    def y_true(self):
        return self._y_true


#
# class Deepbind_RNN_model(object):
#     """The deepbind_RNN model without structure"""
#
#     def __init__(self, config, input_):
#         self._config = config
#         self._input = input_
#         self.weight_initializer = tf.truncated_normal_initializer(stddev=config['init_scale'])
#         self.rna_sequence = tf.placeholder(tf.float32, shape=[None, None, 4], name='input_sequence')
#         self.target_scores = tf.placeholder(tf.float32, shape=[None], name='target_scores')
#         self.target_scores_exp = tf.expand_dims(self.target_scores, 1)
#         conv_input = self.rna_sequence
#         for layer in range(config['num_conv_layers']):
#             if layer == (config['num_conv_layers'] - 1):
#                 self.conv_output = tf.layers.conv1d(inputs=conv_input, filters=config['num_filters'][layer],
#                                                     kernel_size=config['filter_lengths'][layer],
#                                                     strides=config['strides'][layer],
#                                                     padding='SAME', activation=None,
#                                                     kernel_initializer=self.weight_initializer,
#                                                     name='conv_layer_' + str(layer))
#             else:
#                 self.conv_output = tf.layers.conv1d(inputs=conv_input, filters=config['num_filters'][layer],
#                                                     kernel_size=config['filter_lengths'][layer],
#                                                     strides=config['strides'][layer],
#                                                     padding='SAME', activation=tf.nn.relu,
#                                                     kernel_initializer=self.weight_initializer,
#                                                     name='conv_layer_' + str(layer))
#
#             conv_input = self.conv_output
#         if config.get('bidirectional_LSTM', False):
#             lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=config['lstm_size'],
#                                                    initializer=self.weight_initializer,
#                                                    )
#             lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=config['lstm_size'],
#                                                    initializer=self.weight_initializer,
#                                                    )
#             ((output_fw, output_bw), state) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
#                                                                               self.conv_output, dtype=tf.float32,
#                                                                               scope='bidirectional_lstm')
#             self.lstm_output = tf.concat([output_fw[:, -1, :], output_bw[:, -1, :]], 1,
#                                          name='concatenated_lstm_output')
#         else:
#             lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=config['lstm_size'],
#                                                    initializer=self.weight_initializer,
#                                                    )
#             output, state = tf.nn.dynamic_rnn(lstm_fw_cell, self.conv_output, dtype=tf.float32,
#                                               scope='unidirectional_lstm')
#             self.lstm_output = output[:, -1, :]
#         self.target_predictions = tf.layers.dense(self.lstm_output, units=1,
#                                                   kernel_regularizer=
#                                                   tf.contrib.layers.l2_regularizer(scale=config['lam_model']),
#                                                   name='target_prediction')
#         self.loss = tf.losses.mean_squared_error(self.target_scores_exp, self.target_predictions,
#                                                  scope='mean_squared_error')
#         self._train_op = tf.contrib.layers.optimize_loss(self.loss, tf.contrib.framework.get_global_step(),
#                                                          learning_rate=tf.constant(config['eta_model'], tf.float32),
#                                                          optimizer='Adam',
#                                                          clip_gradients=config.get('gradient_clip_value', 20.0),
#                                                          name='train_op')
#
#     @property
#     def input(self):
#         return self._input
#
#     @property
#     def config(self):
#         return self._config
#
#     @property
#     def cost(self):
#         return self.loss
#
#     @property
#     def train_op(self):
#         return self._train_op
#
#     @property
#     def predict_op(self):
#         return tf.squeeze(self.target_predictions)
#
#     @property
#     def x(self):
#         return self.rna_sequence
#
#     @property
#     def y_true(self):
#         # return self.target_scores


def Deepbind_model(config, input, model_type):
    if model_type == 'CNN':
        return Deepbind_CNN_model(config, input)
    elif model_type == 'CNN_struct':
        return Deepbind_CNN_struct_model(config, input)
    elif model_type == 'RNN_struct':
        return Deepbind_RNN_struct_model(config, input)
    elif model_type == 'RNN':
        return Deepbind_RNN_model(config, input)


#
# def run_clip_epoch_parallel(session, models, input_data, config):
#     if isinstance(input_data,list):
#         Nbatch = int(ceil(input_data[0].total_cases * 1.0 / config['minib']))
#         scores = np.zeros([len(models), input_data[0].total_cases])
#     else:
#         Nbatch = int(ceil(input_data.total_cases * 1.0 / config['minib']))
#         scores = np.zeros([len(models), input_data.total_cases])
#     minib = config['minib']
#     num_models = len(models)
#     auc = np.zeros([num_models])
#     for step in range(Nbatch):
#         fetches = {}
#         feed_dict = {}
#         if isinstance(input_data,list):
#             for i,(model,input) in enumerate(zip(models,input_data)):
#                 feed_dict[model.x] = input.data[(minib * step): (minib * (step + 1)), :, :]
#                 feed_dict[model.y_true] = input.labels[(minib * step): (minib * (step + 1))]
#                 fetches["predictions" + str(i)] = model.predict_op
#         else:
#             for i, model in enumerate(models):
#                 feed_dict[model.x] = input_data.data[(minib * step): (minib * (step + 1)), :, :]
#                 feed_dict[model.y_true] = input_data.labels[(minib * step): (minib * (step + 1))]
#                 fetches["predictions" + str(i)] = model.predict_op
#         vals = session.run(fetches, feed_dict)
#         for j in range(num_models):
#             scores[j, (minib * step): (minib * (step + 1))] = vals['predictions' + str(j)]
#     for j in range(num_models):
#         if isinstance(input_data, list):
#             auc[j, :] = roc_auc_score(input_data[j].labels, scores[j, :])
#         else:
#             auc[j, :] = roc_auc_score(input_data.labels, scores[j, :])
#     return auc
#
#
# def convert_to_one_hot(input_seq):
#     one_hot = np.ones([len(input_seq,4])*0.25
#     for i, nuc in enumerate(input_seq):
#         if nuc == 'A' or 'a':
#             one_hot[i,:] = []

def run_clip_epoch_parallel(session, model, input_data, config):
    model = model[0]
    scores = np.zeros(input_data.total_cases)
    auc = 0
    window_size = 40
    Nbatch = input_data.total_cases
    for step in range(Nbatch):
        fetches = {}
        feed_dict = {}
        seq_batch = input_data.data[step:step + 1, :, :]
        last_element = np.where(seq_batch[0, :, 0] == 0)[0][0]
        n_batch_seq = (last_element - window_size)
        batch_scores = np.zeros([n_batch_seq])
        for seq_step in range(n_batch_seq):
            input_sequence = seq_batch[:, seq_step:seq_step + window_size, :]
            feed_dict[model.x] = input_sequence
            fetches['predictions'] = model.predict_op
            vals = session.run(fetches, feed_dict)
            batch_scores[seq_step] = vals['predictions']
        scores[step] = np.max(batch_scores)
    auc = roc_auc_score(input_data.labels, scores)
    return auc


# def run_clip_epoch_parallel(session, models, input_data, config):
#
#     if isinstance(input_data, list):
#         Nbatch = int(ceil(input_data[0].total_cases * 1.0 / config['minib']))
#         scores = np.zeros([len(models), input_data[0].total_cases])
#     else:
#         Nbatch = int(ceil(input_data.total_cases * 1.0 / config['minib']))
#         scores = np.zeros([len(models), input_data.total_cases])
#     minib = config['minib']
#     num_models = len(models)
#     auc = np.zeros([num_models])
#     window_size = 40
#     stride = 1
#     for step in range(Nbatch):
#         fetches = {}
#         feed_dict = {}
#         if isinstance(input_data, list):
#             for i, (model, input) in enumerate(zip(models, input_data)):
#                 minib_seq = input.data[(minib * step): (minib * (step + 1)), :, :]
#                 # n_batch_seq = int(ceil(minib_seq.shape[1] * 1.0 / window_size))
#                 n_batch_seq = (minib_seq.shape[1] - window_size) / stride
#                 batch_scores = np.zeros([len(models), minib_seq.shape[0], n_batch_seq])
#                 for seq_step in range(0, n_batch_seq, step=stride):
#                     # input_sequence = minib_seq[:, (window_size * seq_step): (window_size * (seq_step + 1)), :]
#                     input_sequence = minib_seq[:, seq_step:seq_step + window_size, :]
#                     feed_dict[model.x] = input_sequence
#                     fetches["predictions" + str(i)] = model.predict_op
#                     vals = session.run(fetches, feed_dict)
#                     for j in range(num_models):
#                         batch_scores[j, :, seq_step] = vals['predictions' + str(j)]
#                         # feed_dict[model.x] = input.data[(minib * step): (minib * (step + 1)), :, :]
#                         # feed_dict[model.y_true] = input.labels[(minib * step): (minib * (step + 1))]
#                         # fetches["predictions" + str(i)] = model.predict_op
#         else:
#             for i, model in enumerate(models):
#                 minib_seq = input_data.data[(minib * step): (minib * (step + 1)), :, :]
#                 # n_batch_seq = int(ceil(minib_seq.shape[1] * 1.0 / window_size))
#                 n_batch_seq = (minib_seq.shape[1] - window_size) / stride
#                 batch_scores = np.zeros([len(models), minib_seq.shape[0], n_batch_seq])
#                 for seq_step in range(n_batch_seq):
#                     # input_sequence = minib_seq[:, (window_size * seq_step): (window_size * (seq_step + 1)), :]
#                     input_sequence = minib_seq[:, seq_step:seq_step + window_size, :]
#                     feed_dict[model.x] = input_sequence
#                     # feed_dict[model.x] = np.concatenate([input_sequence,np.zeros([input_sequence.shape[0],3,input_sequence.shape[2]])],axis=1)
#                     fetches["predictions" + str(i)] = model.predict_op
#                     vals = session.run(fetches, feed_dict)
#                     for j in range(num_models):
#                         batch_scores[j, :, seq_step] = vals['predictions' + str(j)]
#                         # feed_dict[model.x] = input_data.data[(minib * step): (minib * (step + 1)), :, :]
#                         # feed_dict[model.y_true] = input_data.labels[(minib * step): (minib * (step + 1))]
#                         # fetches["predictions" + str(i)] = model.predict_op
#         # vals = session.run(fetches, feed_dict)
#         for j in range(num_models):
#             scores[j, (minib * step): (minib * (step + 1))] = np.max(batch_scores, axis=2)
#     for j in range(num_models):
#         if isinstance(input_data, list):
#             auc[j] = roc_auc_score(input_data[j].labels, scores[j, :])
#         else:
#             auc[j] = roc_auc_score(input_data.labels, scores[j, :])
#     return auc




def run_epoch_parallel(session, models, input_data, config, epoch, train=False, verbose=False, testing=False,
                       scores=False):
    if isinstance(input_data,list):
        Nbatch_train = int(ceil(input_data[0].training_cases * 1.0 / config['minib']))
        Nbatch_test = int(ceil(input_data[0].test_cases * 1.0 / config['minib']))

        training_scores = np.zeros([len(models), input_data[0].training_cases])
        test_scores = np.zeros([len(models), input_data[0].test_cases])
    else:
        Nbatch_train = int(ceil(input_data.training_cases * 1.0 / config['minib']))
        Nbatch_test = int(ceil(input_data.test_cases * 1.0 / config['minib']))

        training_scores = np.zeros([len(models), input_data.training_cases])
        test_scores = np.zeros([len(models), input_data.test_cases])

    minib = config['minib']
    num_models = len(models)
    cost_temp = np.zeros([num_models])
    pearson_train = np.zeros([num_models, 2])

    for step in range(Nbatch_train):
        fetches = {}
        feed_dict = {}
        if isinstance(input_data,list):
            for i,(model,input) in enumerate(zip(models,input_data)):
                feed_dict[model.x] = input.training_data[(minib * step): (minib * (step + 1)), :, :]
                feed_dict[model.y_true] = input.training_labels[(minib * step): (minib * (step + 1))]
                fetches["cost" + str(i)] = model.cost
                if train:
                    fetches["eval_op" + str(i)] = model.train_op

                fetches["predictions" + str(i)] = model.predict_op
        else:
            for i, model in enumerate(models):
                feed_dict[model.x] = input_data.training_data[(minib * step): (minib * (step + 1)), :, :]
                feed_dict[model.y_true] = input_data.training_labels[(minib * step): (minib * (step + 1))]
                fetches["cost"+str(i)] = model.cost
                if train:
                    fetches["eval_op" +str(i)] = model.train_op
                fetches["predictions" + str(i)] = model.predict_op
        vals = session.run(fetches, feed_dict)
        for j in range(num_models):
            cost_temp[j] += vals["cost"+str(j)]
            training_scores[j, (minib * step): (minib * (step + 1))] = vals['predictions' + str(j)]
    cost_train = cost_temp / Nbatch_train
    for j in range(num_models):
        if isinstance(input_data, list):
            pearson_train[j, :] = stats.pearsonr(input_data[j].training_labels, training_scores[j, :])
        else:
            pearson_train[j, :] = stats.pearsonr(input_data.training_labels, training_scores[j, :])
    cost_temp = np.zeros([num_models])
    if testing or scores:
        pearson_test = np.zeros([num_models, 2])
        for step in range(Nbatch_test):
            feed_dict = {}
            fetches = {}

            if isinstance(input_data, list):
                for i, (model, input) in enumerate(zip(models, input_data)):
                    feed_dict[model.x] = input.test_data[(minib * step): (minib * (step + 1)), :, :]
                    feed_dict[model.y_true] = input.test_labels[(minib * step): (minib * (step + 1))]
                    fetches["cost" + str(i)] = model.cost
                    fetches["predictions" + str(i)] = model.predict_op
            else:
                for i, model in enumerate(models):
                    feed_dict[model.x] = input_data.test_data[(minib * step): (minib * (step + 1)), :, :]
                    feed_dict[model.y_true] = input_data.test_labels[(minib * step): (minib * (step + 1))]
                    fetches["cost"+str(i)] = model.cost
                    fetches["predictions"+str(i)] = model.predict_op
            vals = session.run(fetches, feed_dict)

            for j in range(num_models):
                cost_temp[j] += vals["cost" + str(j)]
                test_scores[j, (minib * step): (minib * (step + 1))] = vals['predictions' + str(j)]
                # if isinstance(input_data,list):
                #     mbatchY_test = input_data[i].test_labels[(minib * step): (minib * (step + 1))]
                # else:
                #     mbatchY_test = input_data.test_labels[(minib * step): (minib * (step + 1))]
                # cost_test[j] += vals["cost"+str(j)]
                # pearson_test[j] += stats.pearsonr(mbatchY_test, vals["predictions"+str(j)])[0]
        cost_test = cost_temp / Nbatch_test
        for j in range(num_models):
            if isinstance(input_data, list):
                pearson_test[j, :] = stats.pearsonr(input_data[j].test_labels, test_scores[j, :])
            else:
                pearson_test[j, :] = stats.pearsonr(input_data.test_labels, test_scores[j, :])
        if verbose:
            best_model = np.argmin(cost_train)
            print (
            "Epoch:%04d, Train cost(min)=%0.4f, Train pearson=%0.4f, Test cost(min)=%0.4f, Test Pearson(max)=%0.4f" %
            (epoch + 1, cost_train[best_model], pearson_train[best_model][0], cost_test[best_model],
             pearson_test[best_model][0]))
            print(["%.4f" % p for p in pearson_test[:, 0]])
        if scores:
            return (cost_train, cost_test, pearson_train, pearson_test, training_scores, test_scores)
        return (cost_train, cost_test, pearson_test[:, 0])
    return cost_train

def train_model_parallel(session, config, models, input_data, early_stop = False):
    """Trains a list of models in parallel. Expects a list of inputs of equal length as models. Config file is u """
    if early_stop:
        epochs = config['early_stop_epochs']
    else:
        epochs = config['epochs']
    test_epochs = epochs // config['test_interval']
    num_models = len(models)
    cost_train = np.zeros([test_epochs, num_models])
    cost_test = np.zeros([test_epochs, num_models])
    pearson_test = np.zeros([test_epochs, num_models])
    session.run(tf.global_variables_initializer())
    for i in range(epochs):
        _ = run_epoch_parallel(session, models, input_data, config, i, train=True)
        if i % config['test_interval'] == 0:
            step = i //config['test_interval']
            (cost_train[step], cost_test[step], pearson_test[step]) = \
            run_epoch_parallel(session, models, input_data, config, i, train=False,
                               verbose=True, testing = True)

    cost_test = np.transpose(cost_test,[1,0])
    pearson_test = np.transpose(pearson_test,[1,0])
    return (cost_test,pearson_test)


def compute_gradient(session, model, input_data, config):
    Nbatch_test = int(ceil(input_data.test_cases * 1.0 / config['minib']))
    minib = config['minib']
    predictions_test = np.zeros(shape=input_data.test_data.shape[0])
    gradients_test = np.zeros(shape=input_data.test_data.shape)

    # for step in range(Nbatch_train):
    #     fetches = {}
    #     feed_dict = {}
    #     feed_dict[model.x] = input_data.training_data[(minib * step): (minib * (step + 1)), :, :]
    #     feed_dict[model.y_true] = input_data.training_labels[(minib * step): (minib * (step + 1))]
    #     fetches["predictions"] = model.predict_op
    #     fetches['gradient'] = tf.gradients(model.cost, model.x)
    #     vals = session.run(fetches, feed_dict)
    #     predictions_train.append(vals['predictions'])
    #     gradients_train.append(vals['gradient'])

    for step in range(Nbatch_test):
        fetches = {}
        feed_dict = {}
        feed_dict[model.x] = input_data.test_data[(minib * step): (minib * (step + 1)), :, :]
        feed_dict[model.y_true] = input_data.test_labels[(minib * step): (minib * (step + 1))]
        fetches["predictions"] = model.predict_op
        fetches['gradient'] = tf.gradients(model.cost, model.x)
        vals = session.run(fetches, feed_dict)
        gradients_test[(minib * step): (minib * (step + 1)), :, :] = (vals['gradient'][0])
        predictions_test[(minib * step): (minib * (step + 1))] = (vals['predictions'])
    return predictions_test, gradients_test


def evaluate_model_parallel(session, config, models, input_data):
    """Evaluates a list of models in parallel. Expects a list of inputs of equal length as models"""
    num_models = len(models)
    cost_train = np.zeros([num_models])
    cost_test = np.zeros([num_models])
    pearson_test = np.zeros([num_models])
    (cost_train, cost_test, pearson_test) = \
        run_epoch_parallel(session, models, input_data, config, 0, train=False, verbose=True, testing=True)
    return (cost_test, pearson_test)


def score_model_parallel(session, config, models, input_data):
    if isinstance(input_data, list):
        Nbatch_train = int(ceil(input_data[0].training_cases * 1.0 / config['minib']))
        Nbatch_test = int(ceil(input_data[0].test_cases * 1.0 / config['minib']))
        training_scores = np.zeros([len(models), input_data[0].training_cases])
        test_scores = np.zeros([len(models), input_data[0].test_cases])
    else:
        Nbatch_train = int(ceil(input_data.training_cases * 1.0 / config['minib']))
        Nbatch_test = int(ceil(input_data.test_cases * 1.0 / config['minib']))
        training_scores = np.zeros([len(models), input_data.training_cases])
        test_scores = np.zeros([len(models), input_data.test_cases])
    minib = config['minib']
    num_models = len(models)

    for step in range(Nbatch_train):
        fetches = {}
        feed_dict = {}
        if isinstance(input_data, list):
            for i, (model, input) in enumerate(zip(models, input_data)):
                feed_dict[model.x] = input.training_data[(minib * step): (minib * (step + 1)), :, :]
                feed_dict[model.y_true] = input.training_labels[(minib * step): (minib * (step + 1))]
                fetches["predictions" + str(i)] = model.predict_op

        else:
            for i, model in enumerate(models):
                feed_dict[model.x] = input_data.training_data[(minib * step): (minib * (step + 1)), :, :]
                feed_dict[model.y_true] = input_data.training_labels[(minib * step): (minib * (step + 1))]
                fetches["predictions" + str(i)] = model.predict_op

        vals = session.run(fetches, feed_dict)
        for j in range(num_models):
            training_scores[j, (minib * step): (minib * (step + 1))] = vals['predictions' + str(j)]

    for step in range(Nbatch_test):
        feed_dict = {}
        fetches = {}

        if isinstance(input_data, list):
            for i, (model, input) in enumerate(zip(models, input_data)):
                feed_dict[model.x] = input.test_data[(minib * step): (minib * (step + 1)), :, :]
                feed_dict[model.y_true] = input.test_labels[(minib * step): (minib * (step + 1))]
                fetches["predictions" + str(i)] = model.predict_op
        else:
            for i, model in enumerate(models):
                feed_dict[model.x] = input_data.test_data[(minib * step): (minib * (step + 1)), :, :]
                feed_dict[model.y_true] = input_data.test_labels[(minib * step): (minib * (step + 1))]
                fetches["predictions" + str(i)] = model.predict_op
        vals = session.run(fetches, feed_dict)

        for j in range(num_models):
            test_scores[j, (minib * step): (minib * (step + 1))] = vals['predictions' + str(j)]
    for j in range(num_models):
        pearson_test = stats.pearsonr(input_data.test_labels, test_scores[j, :])
        pearson_training = stats.pearsonr(input_data.training_labels, training_scores[j, :])
    return (training_scores, test_scores, pearson_training, pearson_test)



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


def save_calibration(protein, model_type,flag, config,new_cost,new_pearson, save_dir):
    file_name = os.path.join(save_dir,protein+'_'+model_type+'_'+flag+'.npz')
    save_new = True
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if (os.path.isfile(file_name)):
        file = np.load(file_name)
        if new_cost >= file['cost']:
            save_new = False

    if (save_new):
        print("[*] Updating best calibration for %s %s %s"%(protein,model_type,flag))
        kwargs = {'pearson': new_pearson, 'cost': new_cost}
        for key in config.keys():
            kwargs[key] = config[key]
        np.savez(file_name, **kwargs)
    else:
        print("[*] Retaining existing calibration for %s %s %s" % (protein, model_type, flag))


def save_result(protein, model_type, flag, new_cost, new_pearson, save_dir, model_index, model_dir):
    import yaml
    file_name = os.path.join(save_dir, protein + '_' + model_type + '_' + flag + '.npz')
    save_new = True
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if (os.path.isfile(file_name)):
        file = np.load(file_name)
        if new_cost >= file['cost']:
            save_new = False

    if (save_new):
        print("[*] Updating best result for %s %s %s" % (protein, model_type, flag))
        np.savez(file_name,
                 cost=new_cost,
                 pearson=new_pearson
                 )
        result_dict = {'cost': float(new_cost), 'pearson': float(new_pearson), 'model_index': int(model_index),
                       'model_dir': str(model_dir)}
        yaml.dump(result_dict, open(os.path.join(save_dir, protein + '_' + model_type + '_' + flag + '.yml'),'w'))

def load_calibration(protein, model_type, flag, save_dir):
    file_name = os.path.join(save_dir, protein+'_'+model_type+'_'+flag+'.npz')
    if not os.path.isfile(file_name):
        print("[!] Model is not pre-calibrated!")
        return False
    print("[*] Loading existing best calibration for %s %s %s" % (protein,model_type,flag))
    inf = np.load(file_name)
    loaded_config = {'flag':flag}
    loaded_config.update(inf)
    config_new = create_config_dict(**loaded_config)
    return config_new



class input_config(object):
    """Generates configuration for processing input to model"""
    def __init__(self, flag):
        self.folds = 3
        if flag == 'large':
            self.training_frac = 1
            self.test_frac  = 1
        elif flag == 'medium':
            self.training_frac = 0.5
            self.test_frac = 1
        else:
            self.training_frac = 0.1
            self.test_frac = 1


def load_data_rnac2013(target_id_list=None, fold_filter='A'):
    # type: (object, object) -> object
    infile_seq = open('../data/rnac/sequences.tsv')
    infile_target = open('../data/rnac/targets.tsv')
    seq_train = []
    seq_test = []
    target_train = []
    target_test = []
    exp_ids_train = []
    exp_ids_test = []

    infile_structA = open('../data/rnac/combined_profile_rnacA.txt')
    infile_structB = open('../data/rnac/combined_profile_rnacB.txt')
    structures_A = []
    structures_B = []
    seq_len_train = 41
    num_struct_classes = 5

    seq_len_train = 0
    seq_len_test = 0

    target_names = infile_target.readline().split()
    if target_id_list is None:
        target_id_list = target_names
    target_cols_idx = np.zeros(len(target_id_list), dtype=int)
    # target_cols_idx = target_names.index(target_id_list)

    for i in range(len(target_id_list)):
        target_cols_idx[i] = target_names.index(target_id_list[i])
    infile_seq.readline()
    for line_seq in infile_seq:
        seq = line_seq.split('\t')[2].strip()
        line_target = infile_target.readline()
        target = [line_target.split('\t')[i] for i in target_cols_idx]
        fold = line_seq.split('\t')[0].strip()
        target_np = np.array(target, dtype=float)
        if np.any(np.isnan(target_np)):
            continue
        if fold in fold_filter:
            seq_train.append(seq)
            target_train.append(target)
            exp_ids_train.append(line_seq.split('\t')[1].strip())
            seq_len_train = max(seq_len_train, len(seq))
        else:
            seq_test.append(seq)
            target_test.append(target)
            exp_ids_test.append(line_seq.split('\t')[1].strip())
            seq_len_test = max(seq_len_test, len(seq))

    iter_train = 0
    seq_length = max(seq_len_test, seq_len_train)
    iter_test = 0
    for line_struct in infile_structA:
        exp_id = line_struct.split('>')[1].strip()
        exp_id_notnan = exp_ids_train[iter_train]
        probs = np.ones([num_struct_classes, seq_length]) * (1 / num_struct_classes)
        for i in range(5):
            values_line = infile_structA.next().strip()
            values = np.array(map(np.float32, values_line.split('\t')))
            probs[i, 0:values.shape[0]] = values
        if exp_id == exp_id_notnan:
            structures_A.append(probs)
            iter_train = iter_train + 1
    if iter_train < len(exp_ids_train):
        for i in range(iter_train, len(exp_ids_train)):
            structures_A.append(np.ones([num_struct_classes, seq_length]) * (1 / num_struct_classes))

    for line_struct in infile_structB:
        exp_id = line_struct.split('>')[1].strip()
        exp_id_notnan = exp_ids_test[iter_test]
        probs = np.ones([num_struct_classes, seq_length]) * (1 / num_struct_classes)
        for i in range(5):
            values_line = infile_structB.next().strip()
            values = np.array(map(np.float32, values_line.split('\t')))
            probs[i, 0:values.shape[0]] = values
        if exp_id == exp_id_notnan:
            structures_B.append(probs)
            iter_test = iter_test + 1
    if iter_test < len(exp_ids_test):
        for i in range(iter_test, len(exp_ids_test)):
            structures_B.append(np.ones([num_struct_classes, seq_length]) * (1 / num_struct_classes))

    seq_train_enc = []
    for k in range(len(target_id_list)):
        seq_enc = np.ones((len(seq_train), seq_length, 4)) * 0.25
        for i, case in enumerate(seq_train):
            for j, nuc in enumerate(case):
                if nuc == 'A':
                    seq_enc[i, j] = np.array([1, 0, 0, 0])
                elif nuc == 'G':
                    seq_enc[i, j] = np.array([0, 1, 0, 0])
                elif nuc == 'C':
                    seq_enc[i, j] = np.array([0, 0, 1, 0])
                elif nuc == 'U':
                    seq_enc[i, j] = np.array([0, 0, 0, 1])
        seq_enc -= 0.25
        seq_train_enc.append(seq_enc)

    seq_test_enc = []
    for k in range(len(target_id_list)):
        seq_enc = np.ones((len(seq_test), seq_length, 4)) * 0.25
        for i, case in enumerate(seq_test):
            for j, nuc in enumerate(case):
                if nuc == 'A':
                    seq_enc[i, j] = np.array([1, 0, 0, 0])
                elif nuc == 'G':
                    seq_enc[i, j] = np.array([0, 1, 0, 0])
                elif nuc == 'C':
                    seq_enc[i, j] = np.array([0, 0, 1, 0])
                elif nuc == 'U':
                    seq_enc[i, j] = np.array([0, 0, 0, 1])
        seq_enc = seq_enc - 0.25
        seq_test_enc.append(seq_enc)
    data_one_hot_training = np.array(seq_train_enc[0])
    data_one_hot_test = np.array(seq_test_enc[0])
    labels_training = np.array([i[0] for i in target_train], dtype=float)
    labels_test = np.array([i[0] for i in target_test], dtype=float)
    training_cases = data_one_hot_training.shape[0]
    test_cases = data_one_hot_test.shape[0]
    # seq_length = data_one_hot_training.shape[1]

    structures_train = np.array(structures_A, dtype=np.float32)
    structures_test = np.array(structures_B, dtype=np.float32)

    train_remove = np.round(0.0005 * training_cases).astype(int)
    test_remove = np.round(0.0005 * test_cases).astype(int)
    train_ind = np.argpartition(labels_training, -train_remove)[-train_remove:]
    test_ind = np.argpartition(labels_test, -test_remove)[-test_remove:]
    train_clamp = np.min(labels_training[train_ind])
    test_clamp = np.min(labels_test[test_ind])
    labels_training[train_ind] = train_clamp
    labels_test[test_ind] = test_clamp


    save_target = "../data/rnac/npz_archives/" +str(target_id_list[0])
    np.savez(save_target, data_one_hot_training=data_one_hot_training,
             labels_training=labels_training,
             data_one_hot_test=data_one_hot_test,
             labels_test=labels_test, training_cases=training_cases,
             test_cases=test_cases,
             structures_train=structures_train,
             structures_test=structures_test,
             seq_length=seq_length)


def load_data_rnac2009(protein_name):
    data_folder = '../data/rnac_2009/full'
    structure_folder = '../data/rnac_2009/full/structure_annotations'
    training_seqs = []
    training_scores = []
    training_structs = []
    test_structs = []
    num_struct_classes = 5
    test_seqs = []
    test_scores = []
    with open(os.path.join(data_folder, protein_name + '_data_full_A.txt'), 'r') as training_file:
        for line in training_file:
            training_scores.append(line.split('\t')[0])
            training_seqs.append(line.split('\t')[1].strip())

    with open(os.path.join(data_folder, protein_name + '_data_full_B.txt'), 'r') as test_file:
        for line in test_file:
            test_scores.append(line.split('\t')[0])
            test_seqs.append(line.split('\t')[1].strip())

    seq_len_train = max([len(seq) for seq in training_seqs])
    seq_len_test = max([len(seq) for seq in test_seqs])

    with open(os.path.join(structure_folder, protein_name + '_data_full_A_profile'), 'r') as train_struct_file:
        for line in train_struct_file:
            probs = np.ones([num_struct_classes, seq_len_train]) * (1 / num_struct_classes)
            for i in range(5):
                values_line = train_struct_file.next().strip()
                values = np.array(map(np.float32, values_line.split('\t')))
                probs[i, 0:values.shape[0]] = values
            training_structs.append(probs)
    with open(os.path.join(structure_folder, protein_name + '_data_full_B_profile'), 'r') as test_struct_file:
        for line in test_struct_file:
            probs = np.ones([num_struct_classes, seq_len_test]) * (1 / num_struct_classes)
            for i in range(5):
                values_line = test_struct_file.next().strip()
                values = np.array(map(np.float32, values_line.split('\t')))
                probs[i, 0:values.shape[0]] = values
            test_structs.append(probs)



    seq_enc = np.ones((len(training_seqs), seq_len_train, 4)) * 0.25
    for i, case in enumerate(training_seqs):
        for j, nuc in enumerate(case):
            if nuc == 'A':
                seq_enc[i, j] = np.array([1, 0, 0, 0])
            elif nuc == 'G':
                seq_enc[i, j] = np.array([0, 1, 0, 0])
            elif nuc == 'C':
                seq_enc[i, j] = np.array([0, 0, 1, 0])
            elif nuc == 'U':
                seq_enc[i, j] = np.array([0, 0, 0, 1])
            elif nuc == 'T':
                seq_enc[i, j] = np.array([0, 0, 0, 1])
    seq_enc -= 0.25
    data_one_hot_training = np.array(seq_enc)

    seq_enc = np.ones((len(test_seqs), seq_len_test, 4)) * 0.25
    for i, case in enumerate(test_seqs):
        for j, nuc in enumerate(case):
            if nuc == 'A':
                seq_enc[i, j] = np.array([1, 0, 0, 0])
            elif nuc == 'G':
                seq_enc[i, j] = np.array([0, 1, 0, 0])
            elif nuc == 'C':
                seq_enc[i, j] = np.array([0, 0, 1, 0])
            elif nuc == 'U':
                seq_enc[i, j] = np.array([0, 0, 0, 1])
            elif nuc == 'T':
                seq_enc[i, j] = np.array([0, 0, 0, 1])
    seq_enc -= 0.25
    data_one_hot_test = np.array(seq_enc)
    labels_training = np.array(training_scores, dtype=np.float32)
    labels_test = np.array(test_scores, dtype=np.float32)
    training_cases = data_one_hot_training.shape[0]
    test_cases = data_one_hot_test.shape[0]
    save_target = os.path.join('../data/rnac_2009/npz_archives/', protein_name + '.npz')
    np.savez(save_target, data_one_hot_training=data_one_hot_training,
             labels_training=labels_training,
             data_one_hot_test=data_one_hot_test,
             labels_test=labels_test, training_cases=training_cases,
             test_cases=test_cases,
             structures_train=np.array(training_structs, np.float32),
             structures_test=np.array(test_structs, np.float32),
             seq_length=max(seq_len_train, seq_len_test))
    print("[*] Finished loading data for " + protein_name)


def load_data_clipseq(protein_name):
    data_folder = '../data/GraphProt_CLIP_sequences'
    structure_folder = '../data/GraphProt_CLIP_sequences/structure_annotations/' + protein_name
    structs = []
    num_struct_classes = 5
    seqs = []
    labels = []
    with open(os.path.join(data_folder, protein_name + '.ls.positives.fa'), 'r') as pos_file:
        for line in pos_file:
            seqs.append(pos_file.next().strip())
            labels.append(1.0)
    with open(os.path.join(data_folder, protein_name + '.ls.negatives.fa'), 'r') as neg_file:
        for line in neg_file:
            seqs.append(neg_file.next().strip())
            labels.append(0.0)
    seq_len = max([len(seq) for seq in seqs])

    with open(os.path.join(structure_folder, protein_name + '.ls.positives_combined'), 'r') as pos_struct_file:
        for line in pos_struct_file:
            probs = np.ones([num_struct_classes, seq_len]) * (1 / num_struct_classes)
            for i in range(5):
                values_line = pos_struct_file.next().strip()
                values = np.array(map(np.float32, values_line.split('\t')))
                probs[i, 0:values.shape[0]] = values
            structs.append(probs)
    with open(os.path.join(structure_folder, protein_name + '.ls.negatives_combined'), 'r') as neg_struct_file:
        for line in neg_struct_file:
            probs = np.ones([num_struct_classes, seq_len]) * (1 / num_struct_classes)
            for i in range(5):
                values_line = neg_struct_file.next().strip()
                values = np.array(map(np.float32, values_line.split('\t')))
                probs[i, 0:values.shape[0]] = values
            structs.append(probs)

    seq_enc = np.ones((len(seqs), seq_len, 4)) * 0.25
    for i, case in enumerate(seqs):
        for j, nuc in enumerate(case):
            if nuc == 'A' or nuc == 'a':
                seq_enc[i, j] = np.array([1, 0, 0, 0])
            elif nuc == 'G' or nuc == 'g':
                seq_enc[i, j] = np.array([0, 1, 0, 0])
            elif nuc == 'C' or nuc == 'c':
                seq_enc[i, j] = np.array([0, 0, 1, 0])
            elif nuc == 'U' or nuc == 'u':
                seq_enc[i, j] = np.array([0, 0, 0, 1])
            elif nuc == 'T' or nuc == 't':
                seq_enc[i, j] = np.array([0, 0, 0, 1])
    seq_enc -= 0.25
    data_one_hot = np.array(seq_enc, np.float32)
    labels_array = np.array(labels, np.float32)
    total_cases = data_one_hot.shape[0]
    structures = np.array(structs, np.float32)
    save_target = os.path.join('../data/GraphProt_CLIP_sequences/npz_archives', protein_name + '.npz')
    np.savez(save_target, data_one_hot=data_one_hot,
             labels=labels_array,
             seq_length=seq_len,
             structures=structures,
             total_cases=total_cases)
    print("[*] Finished loading data for " + protein_name)

def load_data(protein_name):
    if 'RNCMPT' in protein_name:
        # if True:
        if not (os.path.isfile('../data/rnac/npz_archives/' + str(protein_name) + '.npz')):
            print("[!] Processing input for " + protein_name)
            load_data_rnac2013([protein_name])
        return np.load('../data/rnac/npz_archives/' + str(protein_name) + '.npz')
    else:
        # if True:
        if not (os.path.isfile('../data/rnac_2009/npz_archives/' + str(protein_name) + '.npz')):
            print("[!] Processing input for " + protein_name)
            load_data_rnac2009(protein_name)
        return np.load('../data/rnac_2009/npz_archives/' + str(protein_name) + '.npz')


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


# def generate_configs_RNN(num_calibrations, flag='small'):
#     configs = []
#     for i in range(num_calibrations):
#         eta = np.float32(10 ** (np.random.uniform(-2, -6)))
#         lam = np.float32(10 ** (np.random.uniform(-3, -6)))
#         init_scale = np.float32(10 ** (np.random.uniform(-7, -3)))
#         minib = 100
#         test_interval = 10
#         bidirectional_LSTM = np.random.choice([True, False])
#         num_conv_layers = np.random.choice([2])
#         filter_lengths = [16 // (2 ** i) for i in range(num_conv_layers)]
#         num_filters = [8 * (i + 1) for i in range(num_conv_layers)]
#         strides = np.random.choice([1], size=num_conv_layers)
#         pool_windows = np.random.choice([1], size=num_conv_layers)
#         batchnorm = np.random.choice([True, False])
#         lstm_size = np.random.choice([10, 20, 30])
#         temp_config = {'eta_model': eta, 'lam_model': lam, 'minib': minib,
#                        'test_interval': test_interval, 'filter_lengths': filter_lengths, 'num_filters': num_filters,
#                        'num_conv_layers': num_conv_layers, 'lstm_size': lstm_size, 'strides': strides,
#                        'pool_windows': pool_windows,
#                        'bidirectional_LSTM': bidirectional_LSTM,
#                        'batchnorm': batchnorm,
#                        'init_scale': init_scale, 'flag': flag}
#
#         configs.append(create_config_dict(**temp_config))
#     return configs

def generate_configs_RNN(num_calibrations, flag='small'):
    configs = []
    for i in range(num_calibrations):
        eta = np.float32(10 ** (np.random.uniform(-2, -6)))
        momentum = np.float32(np.random.uniform(0.95, 0.99))
        lam = np.float32(10 ** (np.random.uniform(-3, -6)))
        init_scale = np.float32(10 ** (np.random.uniform(-7, -3)))
        minib = 100
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


def generate_configs(num_calibrations, model_type, flag='small'):
    if model_type=='CNN':
        return generate_configs_CNN(num_calibrations, flag)
    if model_type=='CNN_struct':
        return generate_configs_CNN(num_calibrations, flag)
    if model_type=='RNN_struct':
        return generate_configs_RNN(num_calibrations, flag)
    if model_type == 'RNN':
        return generate_configs_RNN(num_calibrations, flag)

def summarize(save_path='../results_final/'):
    protein_list = ['Fusip', 'HuR', 'PTB', 'RBM4', 'SF2', 'SLM2', 'U1A', 'VTS1', 'YB1',
                    'RNCMPT00100',
                    'RNCMPT00101',
                    'RNCMPT00102',
                    'RNCMPT00103',
                    'RNCMPT00104',
                    'RNCMPT00105',
                    'RNCMPT00106',
                    'RNCMPT00107',
                    'RNCMPT00108',
                    'RNCMPT00109',
                    'RNCMPT00010',
                    'RNCMPT00110',
                    'RNCMPT00111',
                    'RNCMPT00112',
                    'RNCMPT00113',
                    'RNCMPT00114',
                    'RNCMPT00116',
                    'RNCMPT00117',
                    'RNCMPT00118',
                    'RNCMPT00119',
                    'RNCMPT00011',
                    'RNCMPT00120',
                    'RNCMPT00121',
                    'RNCMPT00122',
                    'RNCMPT00123',
                    'RNCMPT00124',
                    'RNCMPT00126',
                    'RNCMPT00127',
                    'RNCMPT00129',
                    'RNCMPT00012',
                    'RNCMPT00131',
                    'RNCMPT00132',
                    'RNCMPT00133',
                    'RNCMPT00134',
                    'RNCMPT00136',
                    'RNCMPT00137',
                    'RNCMPT00138',
                    'RNCMPT00139',
                    'RNCMPT00013',
                    'RNCMPT00140',
                    'RNCMPT00141',
                    'RNCMPT00142',
                    'RNCMPT00143',
                    'RNCMPT00144',
                    'RNCMPT00145',
                    'RNCMPT00146',
                    'RNCMPT00147',
                    'RNCMPT00148',
                    'RNCMPT00149',
                    'RNCMPT00014',
                    'RNCMPT00150',
                    'RNCMPT00151',
                    'RNCMPT00152',
                    'RNCMPT00153',
                    'RNCMPT00154',
                    'RNCMPT00155',
                    'RNCMPT00156',
                    'RNCMPT00157',
                    'RNCMPT00158',
                    'RNCMPT00159',
                    'RNCMPT00015',
                    'RNCMPT00160',
                    'RNCMPT00161',
                    'RNCMPT00162',
                    'RNCMPT00163',
                    'RNCMPT00164',
                    'RNCMPT00165',
                    'RNCMPT00166',
                    'RNCMPT00167',
                    'RNCMPT00168',
                    'RNCMPT00169',
                    'RNCMPT00016',
                    'RNCMPT00170',
                    'RNCMPT00171',
                    'RNCMPT00172',
                    'RNCMPT00173',
                    'RNCMPT00174',
                    'RNCMPT00175',
                    'RNCMPT00176',
                    'RNCMPT00177',
                    'RNCMPT00178',
                    'RNCMPT00179',
                    'RNCMPT00017',
                    'RNCMPT00180',
                    'RNCMPT00181',
                    'RNCMPT00182',
                    'RNCMPT00183',
                    'RNCMPT00184',
                    'RNCMPT00185',
                    'RNCMPT00186',
                    'RNCMPT00187',
                    'RNCMPT00018',
                    'RNCMPT00197',
                    'RNCMPT00199',
                    'RNCMPT00019',
                    'RNCMPT00001',
                    'RNCMPT00200',
                    'RNCMPT00202',
                    'RNCMPT00203',
                    'RNCMPT00205',
                    'RNCMPT00206',
                    'RNCMPT00209',
                    'RNCMPT00020',
                    'RNCMPT00212',
                    'RNCMPT00215',
                    'RNCMPT00216',
                    'RNCMPT00217',
                    'RNCMPT00218',
                    'RNCMPT00219',
                    'RNCMPT00021',
                    'RNCMPT00220',
                    'RNCMPT00223',
                    'RNCMPT00224',
                    'RNCMPT00225',
                    'RNCMPT00226',
                    'RNCMPT00228',
                    'RNCMPT00229',
                    'RNCMPT00022',
                    'RNCMPT00230',
                    'RNCMPT00232',
                    'RNCMPT00234',
                    'RNCMPT00235',
                    'RNCMPT00236',
                    'RNCMPT00237',
                    'RNCMPT00238',
                    'RNCMPT00239',
                    'RNCMPT00023',
                    'RNCMPT00240',
                    'RNCMPT00241',
                    'RNCMPT00245',
                    'RNCMPT00246',
                    'RNCMPT00248',
                    'RNCMPT00249',
                    'RNCMPT00024',
                    'RNCMPT00251',
                    'RNCMPT00252',
                    'RNCMPT00253',
                    'RNCMPT00254',
                    'RNCMPT00255',
                    'RNCMPT00256',
                    'RNCMPT00257',
                    'RNCMPT00258',
                    'RNCMPT00259',
                    'RNCMPT00025',
                    'RNCMPT00261',
                    'RNCMPT00262',
                    'RNCMPT00263',
                    'RNCMPT00265',
                    'RNCMPT00268',
                    'RNCMPT00269',
                    'RNCMPT00026',
                    'RNCMPT00270',
                    'RNCMPT00272',
                    'RNCMPT00273',
                    'RNCMPT00274',
                    'RNCMPT00278',
                    'RNCMPT00279',
                    'RNCMPT00027',
                    'RNCMPT00280',
                    'RNCMPT00281',
                    'RNCMPT00282',
                    'RNCMPT00283',
                    'RNCMPT00284',
                    'RNCMPT00285',
                    'RNCMPT00287',
                    'RNCMPT00288',
                    'RNCMPT00289',
                    'RNCMPT00028',
                    'RNCMPT00291',
                    'RNCMPT00029',
                    'RNCMPT00002',
                    'RNCMPT00031',
                    'RNCMPT00032',
                    'RNCMPT00033',
                    'RNCMPT00034',
                    'RNCMPT00035',
                    'RNCMPT00036',
                    'RNCMPT00037',
                    'RNCMPT00038',
                    'RNCMPT00039',
                    'RNCMPT00003',
                    'RNCMPT00040',
                    'RNCMPT00041',
                    'RNCMPT00042',
                    'RNCMPT00043',
                    'RNCMPT00044',
                    'RNCMPT00045',
                    'RNCMPT00046',
                    'RNCMPT00047',
                    'RNCMPT00049',
                    'RNCMPT00004',
                    'RNCMPT00050',
                    'RNCMPT00051',
                    'RNCMPT00052',
                    'RNCMPT00053',
                    'RNCMPT00054',
                    'RNCMPT00055',
                    'RNCMPT00056',
                    'RNCMPT00057',
                    'RNCMPT00058',
                    'RNCMPT00059',
                    'RNCMPT00005',
                    'RNCMPT00060',
                    'RNCMPT00061',
                    'RNCMPT00062',
                    'RNCMPT00063',
                    'RNCMPT00064',
                    'RNCMPT00065',
                    'RNCMPT00066',
                    'RNCMPT00067',
                    'RNCMPT00068',
                    'RNCMPT00069',
                    'RNCMPT00006',
                    'RNCMPT00070',
                    'RNCMPT00071',
                    'RNCMPT00072',
                    'RNCMPT00073',
                    'RNCMPT00074',
                    'RNCMPT00075',
                    'RNCMPT00076',
                    'RNCMPT00077',
                    'RNCMPT00078',
                    'RNCMPT00079',
                    'RNCMPT00007',
                    'RNCMPT00080',
                    'RNCMPT00081',
                    'RNCMPT00082',
                    'RNCMPT00083',
                    'RNCMPT00084',
                    'RNCMPT00085',
                    'RNCMPT00086',
                    'RNCMPT00087',
                    'RNCMPT00088',
                    'RNCMPT00089',
                    'RNCMPT00008',
                    'RNCMPT00090',
                    'RNCMPT00091',
                    'RNCMPT00093',
                    'RNCMPT00094',
                    'RNCMPT00095',
                    'RNCMPT00096',
                    'RNCMPT00097',
                    'RNCMPT00099',
                    'RNCMPT00009']
    print("[*] Updating result summary")
    model_list = ['CNN_struct', 'CNN', 'RNN_struct', 'RNN']
    result_file = open(save_path+'summary.tsv', 'w')
    heading = 'Protein\t' + '\t'.join(model_list) + '\n'
    result_file.write(heading)
    count = 0
    for protein in protein_list:
        result_file.write(protein)
        for model in model_list:
            if os.path.isfile(save_path + protein + '_' + model + '_large.npz'):
                read_file = np.load(save_path + protein + '_' + model + '_large.npz')
                result_file.write('\t' + str(read_file['pearson']))
                count += 1
            else:
                result_file.write('\t')
        result_file.write('\n')
    print("[*] Update complete, %d records updated" % (count))

def summarize2(model_path):
    # proteins = os.listdir(model_path)
    proteins = new_listdir(model_path)
    values = {}
    for protein in proteins:
        values[protein] = {}
        models = new_listdir(os.path.join(model_path, protein))
        for model in models:
            values[protein][model]=0
            trials =new_listdir(os.path.join(model_path,protein,model))
            for trial in trials:
                result_file = os.path.join(model_path,protein,model,trial,'results_final')
                if os.path.exists(result_file):
                    values[protein]['complete']=True
                else:
                    values[protein]['complete']=False
                if values[protein]['complete']:
                    val = np.load(result_file+'/'+protein+model+'.npz')['pearson']
                    if val >=values[protein][model]:
                        values[protein][model] = val
    result_file = open(model_path + '/summary.tsv', 'w')
    models = ['RNN_struct','CNN_struct','CNN']

    heading = 'Protein\t' + '\t'.join(models) + '\n'
    print(heading)
    result_file.write(heading)
    for protein in proteins:
        if values[protein]['complete']:
            line = protein+ '\t' +'\t'.join([str(values[protein].get(model,' ')) for model in models])+'\n'
            result_file.write(line)
            print(line)

def new_listdir(path):
    dir_list = os.listdir(path)
    dir_list_new = []
    for dir in dir_list:
        if os.path.isdir(os.path.join(path,dir)):
            dir_list_new.append(dir)
    return dir_list_new
