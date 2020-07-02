"""
Badam regrssions with sampling and linearised laplace
"""
import os.path
import math
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg') # for server
import matplotlib.pyplot as plt
from visualise import predictive_dist_plot, predictive_dist_plot_sampling
from optimizers import BayesAdam

from data import reg, reg_iter

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

eps = 1e-8

class FullyConnectedNN():

    def __init__(self, hparams):
        self.hparams = hparams
        self.name = hparams.name
        self.n_in = hparams.n_in
        self.d = hparams.d
        self.num_layers = hparams.num_layers
        self.n_out = hparams.n_out
        self.dropout_rate = hparams.dropout_rate
        self.activation = hparams.activation
        self.sigma_prior = getattr(self.hparams, "sigma_prior", 0.1)
        self.noise_var = getattr(self.hparams, "noise_var", 0.1)
        self.reg = getattr(self.hparams, "l2_reg", 0.0)

        if hparams.activation == 'relu':
            self.activation = tf.nn.relu
        elif hparams.activation == 'tanh':
            self.activation = tf.nn.tanh
        elif hparams.activation == 'sigmoid':
            self.activation = tf.nn.sigmoid
        elif hparams.activation == 'lrelu':
            self.activation = tf.nn.leaky_relu
        elif hparams.activation == 'elu':
            self.activation = tf.nn.elu
        else:
            raise ValueError

        self.log_folder = os.path.join(self.hparams.tensorboard_dir, "graph_{}".format(self.name))

        self._make_graph()

    def _make_graph(self):
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(config=config)
            self.global_step = tf.train.get_or_create_global_step()
            self._init_placeholders()
            self._init_model(activation=self.activation)
            self.sess.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()

            if self.hparams.badam:
                self.sample_weights()

    def _init_placeholders(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.n_in], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, self.n_out], name='Y')
        self.N = tf.placeholder(tf.float32, shape=(), name='N')
        self.training = tf.placeholder(tf.bool, shape=(), name='training_flag')

    def _init_model(self, activation):
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg) if self.reg != 0 else None
        layer = self.X
        for j in range(self.num_layers):
            layer = tf.layers.dropout(
                tf.layers.dense(layer,
                                self.d,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=regularizer,
                                bias_regularizer=regularizer,
                                name='dense_l{}'.format(j)),
                rate=self.dropout_rate,
                training=self.training)

            # if j == self.num_layers - 1:
            #     layer = tf.nn.tanh(layer)
            # else:
            #     layer = activation(layer)
            layer = activation(layer)

        self.preds = tf.layers.dense(layer,
                                     self.n_out,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=regularizer,
                                     bias_regularizer=regularizer,
                                     name='out') # (?, n_out)


        self.loss = tf.reduce_mean(tf.square(self.preds - self.Y)) + tf.losses.get_regularization_loss()

        if self.hparams.badam:
            self.optim = BayesAdam(learning_rate=self.hparams.lr,
                                   laplace_mle=self.hparams.laplace_mle,
                                   N = self.hparams.N,
                                   no_bias_init=self.hparams.no_bias_init,
                                   params = {'beta_1': 0.9,
                                             'beta_2': 0.999,
                                             'prec': 1.0/(self.hparams.sigma_prior**2)})
        else:
            self.optim = tf.train.AdamOptimizer(learning_rate=self.hparams.lr)

        self.grads_loss = self.optim.compute_gradients(loss=self.loss)
        self.train_op = self.optim.apply_gradients(grads_and_vars=self.grads_loss)

        # Jacobian
        self.grads = tf.gradients(self.preds, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    def train(self, tag, n, num_epochs, batch_size, N, return_batch=False, verbose=True, root=''):
        """
        Trains for num_epochs
        :param tag: str
        :param num_epochs: int
        :param data_batching_size: int
        :return:
        """
        with self.graph.as_default():
            self.reg_train_x, self.reg_train_y = reg(n=n, train=True)
            for epoch in range(num_epochs):
                num_batches = int(math.ceil(self.reg_train_x.shape[0] / float(batch_size)))
                for i in range(num_batches):
                    inds = np.random.permutation(self.reg_train_x.shape[0])
                    x = self.reg_train_x[inds[int(i * batch_size):int((i + 1) * batch_size)]].reshape(-1, 1)
                    y = self.reg_train_y[inds[int(i * batch_size):int((i + 1) * batch_size)]].reshape(-1, 1)
                    _, loss, preds = self.sess.run([self.train_op, self.loss, self.preds], feed_dict=self.make_inputs_to_graph(x, y, N, train=True))
                if verbose and epoch % 500 == 0:
                    print("Epoch: {0}, val loss: {1:.3f}".format(epoch, loss))
            self.save(self.log_folder)

        # predictive distirbution
        if verbose:
            fig, ax = plt.subplots(1, 1)
            ax.scatter(x, y, alpha=0.3)
            ax.scatter(x, preds, alpha=0.2)
            ax.set_xticks([0.0, 0.5], minor=True)
            ax.xaxis.grid(True, which='minor')
            plt.tight_layout()
            plt.savefig("{2}plots/train_dist_{0}_{1}.png".format('badam' if self.hparams.badam else 'variational_dropout', tag, root))
            plt.clf()

        if return_batch:
            return x, y

    def debug_weights(self):
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print("{0}: {1}".format(v.name, self.sess.run(v)))

    def debug_mean_coef(self):
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print("{0}: {1}".format(v.name, self.sess.run(self.optim.get_slot(v, "mean_coef"))))

    def test(self, n, batch_size, N, tag, verbose=True, root=''):
        """
        :param iters: number of iterations of testing to make
        :return: test loss
        """
        losses = []
        predictions = []
        xx = []
        yy = []
        with self.graph.as_default():
            for x, y in reg_iter(n, train=False, batch_size=batch_size):
                xx += x[:, 0].tolist()
                yy += y[:, 0].tolist()
                pred, loss = self.sess.run([self.preds,  self.loss], feed_dict=self.make_inputs_to_graph(x,
                                                                                    y, N,
                                                                                    train=False))
                predictions += pred[:, 0].tolist()
                losses.append(loss)

        # predictive distribution
        if verbose:
            fig, ax = plt.subplots(1, 1)
            ax.scatter(xx, yy, alpha=0.5)
            ax.scatter(xx, predictions, alpha=0.05)
            ax.set_xticks([0.0, 0.5], minor=True)
            ax.xaxis.grid(True, which='minor')
            plt.tight_layout()
            plt.savefig("{2}plots/test_dist_{0}_{1}.png".format('badam' if self.hparams.badam else 'variational_dropout', tag, root))
            plt.clf()

        return np.mean(losses)

    def sample_weights(self):
        """
        Samples weights according the BADAM distribution and places them into the graph
        :return: None
        """
        with self.graph.as_default():
            weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            weights = [v for v in weights if 'out' not in v.name]
            # sigmas = [tf.sqrt(self.optim.get_slot(v, "sigma"), name="ws_sqrt_{}".format(i)) for i, v in enumerate(weights)]
            if self.hparams.laplace_mle:
                sigmas = [tf.sqrt(self.optim.get_slot(v, "sigma")) for v in weights]  # the slot sigma is actually the variance
                mean_coefs = [self.optim.get_slot(v, "mean_coef") for v in weights]
                store = []
                for _ in range(self.hparams.n_samples):
                    vv = [coef * v + sigma * tf.random_normal(v.get_shape()) for v, coef, sigma in zip(weights, mean_coefs, sigmas)]
                    store.append(vv)
                y = list(zip(*store)) # [(<tf.Tensor 'add:0' shape=(1, 100) dtype=float32>,), (<tf.Tensor 'add_1:0' shape=(100,) dtype=float32>,), (<tf.Tensor 'add_2:0' shape=(100, 100) dtype=float32>,), (<tf.Tensor 'add_3:0' shape=(100,) dtype=float32>,)]
                self.new_vars = [tf.add_n(v) / len(v) for v in y]
            else:
                sigmas = [tf.sqrt(self.optim.get_slot(v, "sigma") / self.N) for v in weights]  # the slot sigma is actually the variance
                self.new_vars = [v + sigma * tf.random_normal(v.get_shape()) for v, sigma in zip(weights, sigmas)]
            # assignment
            self.assignment_op = [v.assign(v_new) for v, v_new in zip(weights, self.new_vars)]

    def predictive_dist_sampling(self, x, y, N, samples=100):
        """
        Plots the predictive distributions for Variational Dropout and BADAM
        :param d: number of points
        :param data_batching_size: batch_size for use in outputing the predictive distributions
        :return: None
        """
        xx, yy, predictions = [], [], []
        with self.graph.as_default():
            if self.hparams.badam:
                for i in range(samples):
                    self.sess.run(self.assignment_op, feed_dict=self.make_inputs_to_graph(x, y, N, train=False))
                    pred = self.sess.run(self.preds, feed_dict=self.make_inputs_to_graph(x, y, N, train=False))
                    yy += y[:, 0].tolist()
                    xx += x[:, 0].tolist()
                    predictions += pred[:, 0].tolist()
                    self.restore(self.log_folder)
            # MC dropout
            else:
                for _ in range(samples):
                    pred = self.sess.run(self.preds, feed_dict=self.make_inputs_to_graph(x, y, N, train=True))
                    predictions += pred[:, 0].tolist()
                    yy += y[:, 0].tolist()
                    xx += x[:, 0].tolist()

        return predictions, yy, xx

    def gradients(self, d, data, targets, N):
        no_train = data.shape[0]
        S = np.zeros((no_train, d))
        with self.graph.as_default():
            for i in range(no_train):
                gg = self.sess.run(self.grads, feed_dict=self.make_inputs_to_graph(data[i].reshape(-1, 1),
                                                                                   targets[i].reshape(-1, 1),
                                                                                   N,
                                                                                   train=False))  # gradients: a list of (gradient, variable) pairs.

                gg_flat = np.hstack(np.array([g.flatten() for g in gg]))
                S[i, :] = gg_flat
        return S

    def predictive_dist_linear(self, x, y, N, data_var = 0.0, ggn_approx=False):
        y_preds, pred_sigmas = [], []
        with self.graph.as_default():
            weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
            d = sum([self.sess.run(w).size for w in weights])
            if ggn_approx:
                raise NotImplementedError
                # for i in range(x.shape[0]):
                #     y_pred = self.sess.run([self.preds], feed_dict=self.make_inputs_to_graph(x[i, :].reshape(1, 1), y[i, :].reshape(1, 1), N, train=False))[0] # gradients: a list of (gradient, variable) pairs.
                #     y_preds.append(y_pred)
                # d = sum([self.sess.run(w).size for w in weights])
                # Z = self.gradients(d, self.reg_train_x, self.reg_train_y) # (no_train x d) max = 1, min-1
                # N = Z.shape[0]
                # G = self.gradients(d, x, y) # (no_test x d) max =1, min=-1
                # Gunsq = np.expand_dims(G, 2)
                # Pinv = 1/self.sigma_prior**2 # 100 <--
                # PinvG = Pinv * Gunsq # max = 100
                # ZPinvG = np.matmul(Z, np.transpose(np.squeeze(PinvG))) # (no_train, no_test) max= 800, min 500, foong max is 200
                # PinvZt = Pinv * np.transpose(Z) # (d, no_train)
                # M = np.eye(N) + (1/data_var) * np.matmul(Z, PinvZt) # (no_train, no_train), max = 80000 <-- order of mag smaller
                # M = (M + np.transpose(M)) / 2
                # M = M + np.eye(N)* M[0,0]*1e-6
                # U = scipy.linalg.cholesky(M) # (no_train, no_train) # max = 200
                # V = scipy.linalg.solve_triangular(U, ZPinvG) # (no_train, no_test) # min=-4000, 800
                # v_tmp = np.zeros((x.shape[0], 1, 1))
                # for i in range(x.shape[0]):
                #     v_tmp[i, 0, 0] = np.expand_dims(np.transpose(V), 1)[i, :, :] @ np.expand_dims(np.transpose(V), 2)[i, :, :]
                # v = (-1 /data_var) * np.einsum('ijk,ikl->ijl', np.expand_dims(np.transpose(V), 1), np.expand_dims(np.transpose(V), 2)) # (no_test, 1, 1) # foong max = -7 min = -1000
                #
                # G *= PinvG
                # prior_term = np.sum(G, 1)
                # pred_sigmas = np.sqrt(data_var + prior_term + np.squeeze(v))
            else:
                print([w.get_shape() for w in weights])
                # Checking that gradients are doing what we think they are doing.
                g_l_1 = self.sess.run(tf.gradients(self.loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)),
                                      feed_dict=self.make_inputs_to_graph(x[0, :].reshape(1, 1),
                                                                          y[0, :].reshape(1, 1),
                                                                          N, train=False))
                g_l_2 = self.sess.run(self.grads_loss, feed_dict=self.make_inputs_to_graph(x[0, :].reshape(1, 1),
                                                                                           y[0, :].reshape(1, 1),
                                                                                           N, train=False))
                # tf.compute gradiens returns (grad, var) pairs
                g_l_2 = [g[0] for g in g_l_2]
                for a, b in zip(g_l_1, g_l_2):
                    assert np.allclose(a, b)

                # Checking that gradients are doing what we think they are doing.
                g_l_1 = self.sess.run(tf.gradients(self.loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)),
                                      feed_dict=self.make_inputs_to_graph(x[1, :].reshape(1, 1),
                                                                          y[1, :].reshape(1, 1),
                                                                          N, train=False))
                g_l_2 = self.sess.run(self.grads_loss, feed_dict=self.make_inputs_to_graph(x[1, :].reshape(1, 1),
                                                                                           y[1, :].reshape(1, 1),
                                                                                           N, train=False))
                g_l_2 = [g[0] for g in g_l_2]
                for a, b in zip(g_l_1, g_l_2):
                    assert np.allclose(a, b)

                for i in range(x.shape[0]):
                    y_pred, gg = self.sess.run([self.preds, self.grads], feed_dict=self.make_inputs_to_graph(x[i, :].reshape(1, 1), y[i, :].reshape(1, 1), N, train=False)) # gradients: a list of (gradient, variable) pairs.
                    y_preds.append(y_pred)
                    sigmas = [self.optim.get_slot(v, "sigma") for v in weights]
                    S = np.diag(np.hstack(np.array([self.sess.run(s).flatten() for s in sigmas])))
                    S *= (1.0/N)
                    gg_flat = np.hstack(np.array([g.flatten() for g in gg]))
                    vv = data_var + np.matmul(np.transpose(gg_flat), np.matmul(S, gg_flat))
                    pred_sigmas.append(np.sqrt(vv + eps))
        return y_preds, pred_sigmas, d

    def save(self, model_dir):
        self.saver.save(self.sess, os.path.join(model_dir, "model.ckpt"))

    def restore(self, model_dir):
        self.saver.restore(self.sess, os.path.join(model_dir, "model.ckpt"))

    def make_inputs_to_graph(self, batch_x, batch_y, N, train):
        return {self.X: batch_x, self.Y: batch_y, self.N: N, self.training: train}

if __name__ == "__main__":
    t = 'badam'
    n = 200
    no_test = 400
    n_epochs = 20000
    learning_rate = 0.001
    batch_size = 128
    root_model_folder = '../results/'
    pickle_file = 'reg_pd.pickle'
    root_data_folder = '../results/'
    data_var = 0.00
    dr = 0.0
    num_layers = 2
    dims = 50
    Ns = [200, 400, 1000]
    l2_reg = [0.00001, 0.0001]
    for r in l2_reg:
        ###########
        ## Train ##
        ###########
        tag = "{0}_{1}_{2}_{3}".format(t, num_layers, dims, r)
        hparams_badam = tf.contrib.training.HParams(name=tag, n_in=1, d=dims,
                                                    num_layers=num_layers, n_out=1,
                                                    dropout_rate=dr, activation='tanh',
                                                    sigma_prior=0.1, noise_var=data_var, l2_reg=r,
                                                    badam=True, laplace_mle=False, N=n,
                                                    output_tb_gradients=False, val_time=1e2,
                                                    tensorboard_dir='../logs', lr=learning_rate,
                                                    model_folder=os.path.join(root_model_folder, 'badam' + tag),
                                                    n_samples=1, no_bias_init=True,
                                                    badam_no_out=False)

        model = FullyConnectedNN(hparams_badam)
        x_train_batch, y_train_batch = model.train(tag, n, n_epochs, batch_size, n, return_batch=True, root='../')
        test_loss = model.test(no_test, batch_size, n, tag, root='../')
        print("Test loss: {}".format(test_loss))
        #########
        ## Val ##
        #########
        val_mse = []
        x, y = reg(n, True, seed=123)
        for N in Ns:
            preds, yy, _ = model.predictive_dist_sampling(x.reshape(-1, 1), y.reshape(-1, 1), N, samples=100)
            val_mse.append(np.mean([(p-target)**2 for p, target in zip(preds, yy)]))
        ##########
        ## Test ##
        ##########
        mipw_sampling, mipw_ll = [], []
        x, y = reg(no_test, False, seed=123)
        for N in Ns:
            preds, yy, xx = model.predictive_dist_sampling(x.reshape(-1, 1), y.reshape(-1, 1), N, samples=100)
            mipw = predictive_dist_plot_sampling(x_train_batch, y_train_batch, xx, yy, preds, tag="sampling_{0}_{1}".format(tag, N), root='../')
            mipw_sampling.append(mipw)
        for N in Ns:
            mus, sigmas, no_params = model.predictive_dist_linear(x.reshape(-1, 1), y.reshape(-1, 1), N, data_var=0.0, ggn_approx=False) # no_params won't change for each iter
            mipw = predictive_dist_plot(x_train_batch, y_train_batch, x, mus, sigmas, tag="ll_{0}_{1}".format(tag, N), root='../')
            mipw_ll.append(mipw)
