import numpy as np 
import tensorflow as tf
import time
import scipy as Sci

class SmallConfig(object):
    """Small config."""
    training_cases=10000
    test_cases=10000
    eta_model=0.01
    momentum_model=0.9
    lam_model=0.00001
    epochs=100
    minib=100
    test_interval = 10


class MediumConfig(object):
    """Medium config."""
    training_cases=50000
    test_cases=50000
    eta_model=0.01
    momentum_model=0.9
    lam_model=0.00001
    epochs=200
    minib=100
    test_interval = 20


class LargeConfig(object):
    """Large config."""
    training_cases=100000
    test_cases=100000
    eta_model=0.01
    momentum_model=0.9
    lam_model=0.00001
    epochs=300
    minib=100
    test_interval = 30

def get_config(flag):
    if flag == "small":
        return SmallConfig()
    elif flag == "medium":
        return MediumConfig()
    elif flag == "large":
        return LargeConfig()
    elif flag == "test":
        return TestConfig()


class Deepbind_CNN_input(object):
    """The deepbind_CNN model input with structure"""
    def __init__(self, config, inf):
        self.batch_size = batch_size = config.batch_size
        self.epochs = epochs = config.epochs
        self.momentum_model = momentum_model = config.momentum_model
        self.eta_model = eta_model = config.eta_model
        self.lam_model = lam_model = config.lam_model
        self.training_cases = training_cases = config.training_cases
        self.test_cases = test_cases = config.test_cases
        self.minib = minib = config.minib
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
        self.training_data = data_one_hot_training[0:config.training_cases]
        self.test_data = data_one_hot_test[0:config.test_cases]
        self.training_labels = labels_training[0:config.training_cases]
        self.test_labels = labels_test[0:config.test_cases]
        self.training_struct = np.transpose(structures_training[0:config.training_cases],[0,2,1])
        self.test_struct = np.transpose(structures_test[0:config.test_cases],[0,2,1])
        self.seq_length = int(seq_length)

class Deepbind_CNN_model(object):
    """The deepbind_CNN model with structure"""
    def __init__(self, config, input_):
        self._input = input_
        batch_size = input_.batch_size
        eta_model = input_.eta_model
        momentum_model = input_.momentum_model
        lam_model = input_.lam_model
        epochs = input_.epochs
        training_cases = input_.training_cases
        test_cases = input_.test_cases
        minib = input_.minib
        m = 16  # Tunable Motif length
        d = 10  # Number of tunable motifs
        m2 = 4  # Filter size for 2 conv net

        x = tf.placeholder(tf.float32, shape=[None, seq_length, 9], name='One_hot_data')
        y_true = tf.placeholder(tf.float32, shape=[None], name='Labels')

        x_image = tf.reshape(x, [-1, seq_length, 1, 9])

        W_conv1 = tf.Variable(tf.random_normal([m, 1, 9, d], stddev=0.01))
        b_conv1 = tf.Variable(tf.constant(0.001, shape=[d]))

        h_conv1 = tf.nn.conv2d(x_image, W_conv1,
                       strides=[1, 1, 1, 1], padding='SAME')
        h_relu_conv1 = tf.nn.relu(h_conv1 + b_conv1, name='First layer output')
        W_conv2 = tf.Variable(tf.random_normal([m2, 1, d, 1]))
        b_conv2 = tf.Variable(tf.constant(0.001, shape=[1]))
        h_conv2 = tf.nn.conv2d(h_relu_conv1, W_conv2,
                               strides=[1, 1, 1, 1], padding='SAME')

        h_relu_conv2 = tf.nn.relu(h_conv2 + b_conv2)
        # h_max=tf.reduce_max(h_relu_conv2,reduction_indices=[1,2,3]) 
        #Taking max of rectified output was giving poor performance
        h_max = tf.reduce_max(h_conv2+b_conv2, reduction_indices=[1, 2, 3])
        h_avg = tf.reduce_mean(h_conv2+b_conv2, reduction_indices=[1, 2, 3])
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

        _init_op = tf.initialize_all_variables()
        self._train_op = optimizer.minimize(cost + norm_w * lam_model)
        self._predict_op = h_max
    @property
    def input(self):
        return self._input

    
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
    def init(self):
        return self._init_op


def run_epoch(session, model, epoch, eval_op=None, verbose=False, testing=False):
"""Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    fetches = {"cost":model.cost
               }
    if eval_op is not None:
        fetches["eval_op"] = eval_op
    Nbatch_train = model.training_cases // model.minib
    Nbatch_test =  model.test_cases // model.minib
    minib = model.minib
    cost_temp = 0

    for i in range(Nbatch_train):
        mbatchX_train = model.input.training_data[(minib * i): (minib * (i + 1)), :, :]
        mbatchY_train = model.input.training_labels[(minib * i): (minib * (i + 1))]
        feed_dict = {x:mbatchX_train, y_true: mbatchY_train}
        vals = session.run(fetches, feed_dict)
        cost_temp = cost_temp + vals["cost"]
    cost_train = cost_temp / Nbatch_train

    if testing:
        fetches = {"cost":model.cost,
               "predictions":model.predict_op}
        feed_dict = {x:model.input.test_data, y_true:model.input.test_labels }
        vals = session.run(fetches, feed_dict)
        pearson_test = Sci.stats.pearsonr(model.input.test_labels, vals["predictions"])
        cost_test = vals["cost"]
        if verbose:
            print ("Epoch:%04d, Train cost=%0.4f, Test cost=%0.4f, Test Pearson=%0.4f" %
                   (epoch + 1, cost_train, cost_test, pearson_test))
        return(cost_train, cost_test, pearson_test)
    return cost_train

def main(config_flag='medium'):
    # add line for loader
    config=get_config(config_flag)
    inf=np.load("deepbind_RNAC.npz")
    with tf.Graph().as_default():
        input_data = Deepbind_CNN_input(config,inf)
        m = Deepbind_CNN_model(config, input_data)
        tf.scalar_summary("Loss", m.cost)

        sv=tf.train.Supervisor(logdir='models')
        with sv.managed_session() as session:
            for i in range(config.epochs):
                _ = run_epoch(session,m,i, eval_op=m.train_op)
                if i%config.test_interval == 0:
                    (cost_train, cost_test, pearson_test) = run_epoch(session,m,i,verbose=True,testing=True)
            print("Saving model")
            sv.saver.save(session, 'models', global_step = sv.global_step)

if __name__ =="__main__":
    main()


