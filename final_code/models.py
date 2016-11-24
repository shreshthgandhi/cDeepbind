import numpy as np
import tensorflow as tf
from sklearn import cross_validation
import scipy.stats as stats
import time
from sklearn import metrics



class Deepbind_CNN_input(object):
    """The deepbind_CNN model input without structure"""
    def __init__(self, config, inf, validation=False, fold_id=1):
        # self.batch_size = batch_size = config.batch_size
        # self.epochs = epochs = config.epochs
        # self.momentum_model = momentum_model = config.momentum_model
        # self.eta_model = eta_model = config.eta_model
        # self.lam_model = lam_model = config.lam_model

        # self.minib = minib = config.minib
        # self.init_scale = init_scale = config.init_scale
        # self.motif_len = motif_len = config.motif_len
        # self.num_motifs = num_motifs = config.num_motifs
        self.folds = folds = config.folds

        # with np.load("deepbind_RNAC.npz") as inf:
        (data_one_hot_training, labels_training,
         data_one_hot_test, labels_test,
         training_cases, test_cases,
         seq_length) = (inf["data_one_hot_training"], inf["labels_training"],
                        inf["data_one_hot_test"], inf["labels_test"],
                        inf["training_cases"], inf["test_cases"],
                        inf["seq_length"])
        self.training_cases = int(training_cases * config.training_frac)
        self.test_cases = int(test_cases * config.test_frac)
        train_index = range(self.training_cases)
        validation_index = range(self.test_cases)

        if validation:
            kf = cross_validation.KFold(self.training_cases, n_folds=folds)
            check = 1
            for train_idx, val_idx in kf:
                if(check == fold_id):
                    train_index = train_idx
                    validation_index = val_idx
                    break
                check = check + 1
        if validation:
            self.training_data = data_one_hot_training[train_index]
            self.test_data = data_one_hot_training[validation_index]
            self.training_labels = labels_training[train_index]
            self.test_labels = labels_training[validation_index]
        else:
            self.training_data = data_one_hot_training[0:self.training_cases]
            self.test_data = data_one_hot_test[0:self.test_cases]
            self.training_labels = labels_training[0:self.training_cases]
            self.test_labels = labels_test[0:self.test_cases]
#         self.training_struct = np.transpose(structures_training[0:config.training_cases],[0,2,1])
#         self.test_struct = np.transpose(structures_test[0:config.test_cases],[0,2,1])
        
#         self.training_data=np.append(self.training_data,self.training_struct,axis=2)
#         self.test_data=np.append(self.test_data,self.test_struct,axis=2)
        
        self.seq_length = int(seq_length)
        self.training_cases = self.training_data.shape[0]
        self.test_cases = self.test_data.shape[0]


class Deepbind_CNN_struct_input(object):
    """The deepbind_CNN model input with structure"""
    def __init__(self, config, inf, validation=False, fold_id=1):
        # self.batch_size = batch_size = config.batch_size
        # self.epochs = epochs = config.epochs
        # self.momentum_model = momentum_model = config.momentum_model
        # self.eta_model = eta_model = config.eta_model
        # self.lam_model = lam_model = config.lam_model

        # self.minib = minib = config.minib
        # self.init_scale = init_scale = config.init_scale
        # self.motif_len = motif_len = config.motif_len
        # self.num_motifs = num_motifs = config.num_motifs
        self.folds = folds = config.folds

        # with np.load("deepbind_RNAC.npz") as inf:
        (data_one_hot_training, labels_training,
         data_one_hot_test, labels_test,
         structures_training, structures_test,
         training_cases, test_cases,
         seq_length) = (inf["data_one_hot_training"], inf["labels_training"],
                        inf["data_one_hot_test"], inf["labels_test"],
                        inf["structures_train"], inf["structures_test"],
                        inf["training_cases"], inf["test_cases"],
                        inf["seq_length"])
        self.training_cases = int(training_cases * config.training_frac)
        self.test_cases = int(test_cases * config.test_frac)

        train_index = range(self.training_cases)
        validation_index = range(self.test_cases)
        if validation:
            kf = cross_validation.KFold(self.training_cases, n_folds=folds)
            check = 1
            for train_idx, val_idx in kf:
                if(check == fold_id):
                    train_index = train_idx
                    validation_index = val_idx
                    break
                check += 1
        if validation:
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
#         self.training_struct = np.transpose(structures_training[0:config.training_cases],[0,2,1])
#         self.test_struct = np.transpose(structures_test[0:config.test_cases],[0,2,1])
        
#         self.training_data=np.append(self.training_data,self.training_struct,axis=2)
#         self.test_data=np.append(self.test_data,self.test_struct,axis=2)
        self.training_data=np.append(self.training_data,self.training_struct,axis=2)
        self.test_data=np.append(self.test_data,self.test_struct,axis=2)
        self.seq_length = int(seq_length)
        self.training_cases = self.training_data.shape[0]
        self.test_cases = self.test_data.shape[0]

def Deepbind_input(config,inf,model,validation=False,fold_id=1):
    if model == 'CNN':
        return Deepbind_CNN_input(config, inf, validation, fold_id)
    elif model == 'CNN_struct':
        return Deepbind_CNN_struct_input(config, inf, validation, fold_id)


class Deepbind_CNN_struct_model(object):
    """The deepbind_CNN model with structure"""
    def __init__(self, config, input_):
        # type: (object, object) -> object
        self._input = input_
        self._config = config
#         batch_size = input_.batch_size
        eta_model = config.eta_model
        momentum_model = config.momentum_model
        lam_model = config.lam_model
        epochs = config.epochs
        training_cases = input_.training_cases
        test_cases = input_.test_cases
        minib = config.minib
        seq_length = input_.seq_length
        
        m = config.motif_len  # Tunable Motif length
        d = config.num_motifs  # Number of tunable motifs
        m2 = 4  # Filter size for 2 conv net
        self._init_op = tf.initialize_all_variables()

        self._x = x = tf.placeholder(tf.float32, shape=[None, seq_length, 9], name='One_hot_data')
        self._y_true = y_true = tf.placeholder(tf.float32, shape=[None], name='Labels')

        x_image = tf.reshape(x, [-1, seq_length, 1, 9])

        W_conv1 = tf.Variable(tf.random_normal([m, 1, 9, d], stddev=0.01), name='W_Conv1')
        b_conv1 = tf.Variable(tf.constant(0.001, shape=[d]), name='b_conv1')

        h_conv1 = tf.nn.conv2d(x_image, W_conv1,
                       strides=[1, 1, 1, 1], padding='SAME')
        h_relu_conv1 = tf.nn.relu(h_conv1 + b_conv1, name='First_layer_output')
        W_conv2 = tf.Variable(tf.random_normal([m2, 1, d, 1]), name='W_conv2')
        b_conv2 = tf.Variable(tf.constant(0.001, shape=[1]), name= 'b_conv2')
        h_conv2 = tf.nn.conv2d(h_relu_conv1, W_conv2,
                               strides=[1, 1, 1, 1], padding='SAME')

        h_relu_conv2 = tf.nn.relu(h_conv2 + b_conv2)
        # h_max=tf.reduce_max(h_relu_conv2,reduction_indices=[1,2,3]) 
        #Taking max of rectified output was giving poor performance
        h_max = tf.reduce_max(h_conv2+b_conv2, reduction_indices=[1, 2, 3], name='h_max')
        h_avg = tf.reduce_mean(h_conv2+b_conv2, reduction_indices=[1, 2, 3], name='h_avg')
        W_final = tf.Variable(tf.random_normal([2], stddev=0.01))
        b_final = tf.Variable(tf.constant(0.001, shape=[]))
        h_final = tf.mul(tf.pack([h_max,h_avg]),W_final) + b_final
        # Output has shape None and is a vector of length minib

        cost_batch = tf.square(h_max - y_true)
        # cost_batch = tf.square(h_final - y_true)
        self._cost = cost = tf.reduce_mean(cost_batch)
        # tf.scalar_summary("Training Loss", cost)
        norm_w = (tf.reduce_sum(tf.abs(W_conv1)) +tf.reduce_sum(tf.abs(W_conv2)))                  
        optimizer = tf.train.MomentumOptimizer(learning_rate=eta_model,
                                               momentum=momentum_model)
        # optimizer = tf.train.AdamOptimizer(learning_rate=eta_model)

        self._train_op = optimizer.minimize(cost + norm_w * lam_model)
        self._predict_op = h_max
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
#     @property
#     def init_op(self):
#         return self._init_op
    @property
    def x(self):
        return self._x
    @property
    def y_true(self):
        return self._y_true

class Deepbind_CNN_model(object):
    """The deepbind_CNN model without structure"""
    def __init__(self, config, input_):
        self._input = input_
#         batch_size = input_.batch_size
        self._config = config
        eta_model = config.eta_model
        momentum_model = config.momentum_model
        lam_model = config.lam_model
        epochs = config.epochs
        training_cases = input_.training_cases
        test_cases = input_.test_cases
        minib = config.minib
        seq_length = input_.seq_length

        m = 16  # Tunable Motif length
        d = 10  # Number of tunable motifs
        m2 = 4  # Filter size for 2 conv net
        
        self._init_op = tf.initialize_all_variables()

        self._x = x = tf.placeholder(tf.float32, shape=[None, seq_length, 4], name='One_hot_data')
        self._y_true = y_true = tf.placeholder(tf.float32, shape=[None], name='Labels')

        x_image = tf.reshape(x, [-1, seq_length, 1, 4])

        W_conv1 = tf.Variable(tf.random_normal([m, 1, 4, d], stddev=0.01), name='W_Conv1')
        b_conv1 = tf.Variable(tf.constant(0.001, shape=[d]), name='b_conv1')

        h_conv1 = tf.nn.conv2d(x_image, W_conv1,
                       strides=[1, 1, 1, 1], padding='SAME')
        h_relu_conv1 = tf.nn.relu(h_conv1 + b_conv1, name='First_layer_output')
        W_conv2 = tf.Variable(tf.random_normal([m2, 1, d, 1]), name='W_conv2')
        b_conv2 = tf.Variable(tf.constant(0.001, shape=[1]), name= 'b_conv2')
        h_conv2 = tf.nn.conv2d(h_relu_conv1, W_conv2,
                               strides=[1, 1, 1, 1], padding='SAME')

        h_relu_conv2 = tf.nn.relu(h_conv2 + b_conv2)
        # h_max=tf.reduce_max(h_relu_conv2,reduction_indices=[1,2,3]) 
        #Taking max of rectified output was giving poor performance
        h_max = tf.reduce_max(h_conv2+b_conv2, reduction_indices=[1, 2, 3], name='h_max')
        h_avg = tf.reduce_mean(h_conv2+b_conv2, reduction_indices=[1, 2, 3], name='h_avg')
        W_final = tf.Variable(tf.random_normal([2], stddev=0.01))
        b_final = tf.Variable(tf.constant(0.001, shape=[]))

        h_final = tf.mul(tf.pack([h_max,h_avg]),W_final) + b_final


        # Output has shape None and is a vector of length minib

        cost_batch = tf.square(h_max - y_true)
        # cost_batch = tf.square(h_final - y_true)
        self._cost = cost = tf.reduce_mean(cost_batch)
        # tf.scalar_summary("Training Loss", cost)
        norm_w = (tf.reduce_sum(tf.abs(W_conv1)) +tf.reduce_sum(tf.abs(W_conv2)))
                  
        optimizer = tf.train.MomentumOptimizer(learning_rate=eta_model,
                                               momentum=momentum_model)
        # optimizer = tf.train.AdamOptimizer(learning_rate=eta_model)

        
        self._train_op = optimizer.minimize(cost + norm_w * lam_model)
        self._predict_op = h_max
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
#     @property
#     def init_op(self):
#         return self._init_op
    @property
    def x(self):
        return self._x
    @property
    def y_true(self):
        return self._y_true

def Deepbind_model(config, input, model_type):
    if model_type == 'CNN':
        return Deepbind_CNN_model(config, input)
    elif model_type == 'CNN_struct':
        return Deepbind_CNN_struct_model(config, input)

def run_epoch(session, model, epoch, eval_op=None, verbose=False, testing=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    # print("Running epoch")

    fetches = {"cost":model.cost
               }
    if eval_op is not None:
        fetches["eval_op"] = eval_op
    Nbatch_train = model.input.training_cases // model.config.minib
    Nbatch_test =  model.input.test_cases // model.config.minib
    minib = model.config.minib
    cost_temp = 0

    for i in range(Nbatch_train):
        mbatchX_train = model.input.training_data[(minib * i): (minib * (i + 1)), :, :]
        mbatchY_train = model.input.training_labels[(minib * i): (minib * (i + 1))]
        # print(mbatchY_train.shape)
        # print(mbatchY_train[-1])
        feed_dict = {model.x:mbatchX_train, model.y_true: mbatchY_train}
        vals = session.run(fetches, feed_dict)
        cost_temp = cost_temp + vals["cost"]
    cost_train = cost_temp / Nbatch_train

    if testing:
        fetches = {"cost":model.cost,
               "predictions":model.predict_op}
        feed_dict = {model._x:model.input.test_data, model._y_true:model.input.test_labels }
        vals = session.run(fetches, feed_dict)
        pearson_test = stats.pearsonr(model.input.test_labels, vals["predictions"])[0]
        cost_test = vals["cost"]
        if verbose:
            print ("Epoch:%04d, Train cost=%0.4f, Test cost=%0.4f, Test Pearson=%0.4f" %
                   (epoch + 1, cost_train, cost_test, pearson_test))
        return(cost_train, cost_test, pearson_test)
    return cost_train

def train_model(session, config, model, early_stop=False):
    # with tf.Graph().as_default():
    print("Training model")
    print_config(config)
    if early_stop:
        epochs = config.early_stop_epochs
    else:
        epochs = config.epochs
    test_epochs = epochs // config.test_interval
    cost_train = np.zeros([test_epochs])
    cost_test = np.zeros([test_epochs])
    pearson_test = np.zeros([test_epochs])
    # with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    for i in range(epochs):
        _ = run_epoch(session, model, i, eval_op=model.train_op)
        if i % config.test_interval == 0:
            step = i // config.test_interval
            (cost_train[step], cost_test[step], pearson_test[step]) = \
                run_epoch(session, model, i, verbose=True, testing=True)
    best_epoch = int(np.argmax(pearson_test) * config.test_interval)
    best_pearson = np.max(pearson_test)
    last_pearson = pearson_test[-1]
    return (best_pearson, last_pearson, best_epoch)

def print_config(config):
    print("eta = %.4f, momentum =%.2f, lambda =10^%.2f "%(config.eta_model,
                                                          config.momentum_model,
                                                          np.log10(config.lam_model)))

class Config_class(object):
    """Generates configuration"""
    def __init__(self, eta=0.01, momentum=0.9, lam=0.00001,
                 minib=100, test_interval=10,
                 motif_len=16, num_motifs=16, init_scale=0.01, flag='small'):
        self.eta_model = eta
        self.momentum_model = momentum
        self.lam_model = lam
        # self.epochs = epochs
        self.minib = minib
        self.test_interval = test_interval
        self.motif_len = motif_len
        self.num_motifs = num_motifs
        self.init_scale = init_scale
        self.folds = 3
        if flag == 'large':
            self.training_frac = 1

            self.test_frac = 1
            self.epochs = 100
            self.early_stop_epochs = 100
            self.test_interval = 10
        elif flag == 'medium':
            self.training_frac = 0.5
            self.test_frac = 0.5
            self.epochs = 50
            self.early_stop_epochs = 50
            self.test_interval = 5

        else:
            self.training_frac = 0.1
            self.test_frac = 0.1
            self.epochs = 20
            self.early_stop_epochs = 20
            self.test_interval = 2

class input_config(object):
    """Generates configuration for processing input to model"""
    def __init__(self, flag):
        self.folds = 3
        if flag == 'large':
            self.training_frac = 1
            self.test_frac  = 1
        elif flag == 'medium':
            self.training_frac = 0.5
            self.test_frac = 0.5
        else:
            self.training_frac = 0.1
            self.test_frac = 0.1

def load_data(target_id_list=None, fold_filter='A'):
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

    train_remove = np.round(0.05 * training_cases).astype(int)
    test_remove = np.round(0.05 * test_cases).astype(int)
    train_ind = np.argpartition(labels_training, -train_remove)[-train_remove:]
    test_ind = np.argpartition(labels_test, -test_remove)[-test_remove:]
    train_clamp = np.min(labels_training[train_ind])
    test_clamp = np.min(labels_test[test_ind])
    labels_training[train_ind] = train_clamp
    labels_test[test_ind] = test_clamp

    # return (data_one_hot_training, data_one_hot_test,
    #         labels_training, labels_test,
    #         training_cases, test_cases, seq_length)
    save_target = "../data/rnac/npz_archives/" +str(target_id_list[0])
    np.savez(save_target, data_one_hot_training=data_one_hot_training,
             labels_training=labels_training,
             data_one_hot_test=data_one_hot_test,
             labels_test=labels_test, training_cases=training_cases,
             test_cases=test_cases,
             structures_train=structures_train,
             structures_test=structures_test,
             seq_length=seq_length)


def generate_configs_CNN(num_calibrations, flag='small'):
    configs = []
    for i in range(num_calibrations):
        eta = np.float32(10**(np.random.uniform(-4,-6)))
        momentum = np.float32(np.random.uniform(0.95,0.99))
        lam = np.float32(10**(np.random.uniform(-3,-10)))
        init_scale = np.float32(10**(np.random.uniform(-7,-3)))
        minib = 100
        test_interval = 10
        motif_len = 16
        num_motifs = 16
        configs.append(Config_class(eta,momentum,
                                    lam,minib,
                                    test_interval,
                                    motif_len,num_motifs,
                                    init_scale,flag))
    return configs

def generate_configs_CNN_struct(num_calibrations, flag='small'):
    configs = []
    for i in range(num_calibrations):
        eta = np.float32(10**(np.random.uniform(-4,-6)))
        momentum = np.float32(np.random.uniform(0.95,0.99))
        lam = np.float32(10**(np.random.uniform(-3,-10)))
        init_scale = np.float32(10**(np.random.uniform(-7,-3)))
        minib = 100
        test_interval = 10
        motif_len = 16
        num_motifs = 16
        configs.append(Config_class(eta,momentum,
                                    lam,minib,
                                    test_interval,
                                    motif_len,num_motifs,
                                    init_scale,flag))
    return configs

def generate_configs(num_calibrations, model_type, flag='small'):
    if model_type=='CNN':
        return generate_configs_CNN(num_calibrations, flag)
    if model_type=='CNN_struct':
        return generate_configs_CNN_struct(num_calibrations, flag)