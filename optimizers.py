"""
Bayesian Adam class for TF
"""

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.eager import context

import tensorflow as tf


class BayesAdam(optimizer.Optimizer):
    """Custom class for BADAM.
    """
    def __init__(self, laplace_mle, N, no_bias_init=False, learning_rate=0.0001, epsilon=1e-8,
                 params={},
                 use_locking=False, name="BayesAdam"):
        super(BayesAdam, self).__init__(use_locking, name)
        self.laplace_mle = laplace_mle
        self.size_t = N
        self.no_bias_init = no_bias_init
        self._eta = learning_rate
        self._epsilon = epsilon
        self.params = params

    def _prepare(self):
        self._eta_t = ops.convert_to_tensor(self._eta, name="learning_rate")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")
        for key, value in self.params.items():
            setattr(self, key + "_t", ops.convert_to_tensor(value, name=key + "_t"))

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "sigma", self._name)  # variance
            if self.laplace_mle:
                self._zeros_slot(v, "mean_coef", self._name)  # coefficeint on the Gaussian mean of the posterior

    def _apply_dense(self, grad, var):
        eta_t = math_ops.cast(self._eta_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
        beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
        size_t = math_ops.cast(self.size_t, var.dtype.base_dtype)
        if self.laplace_mle:
            prec_t = math_ops.cast(self.prec_t, var.dtype.base_dtype)

        self.t = tf.to_float(tf.train.get_or_create_global_step()) + 1.0

        eta = eta_t * math_ops.sqrt(1 - tf.pow(beta_2_t, self.t)) / (1 - tf.pow(beta_1_t, self.t))

        # Equation (1):
        m = self.get_slot(var, "m")
        m_t = state_ops.assign(m, (m * beta_1_t + grad * (1 - beta_1_t)))

        # Equation (2):
        v = self.get_slot(var, "v")
        v_t = state_ops.assign(v, (v * beta_2_t + (grad * grad) * (1 - beta_2_t)))

        # Equation (3):
        v_sqrt = math_ops.sqrt(v_t)
        sigma = self.get_slot(var, "sigma")
        if self.laplace_mle:
            sigma_t = state_ops.assign(sigma, 1 / (size_t * v_sqrt + prec_t + epsilon_t))
            mean_coef = self.get_slot(var, "mean_coef")
            mean_coef_t = state_ops.assign(mean_coef, v_sqrt / (v_sqrt + (prec_t / size_t)))
        else:
            sigma_t = state_ops.assign(sigma, 1 / (size_t * v_sqrt + epsilon_t))

        if self.no_bias_init:
            var_update = state_ops.assign_sub(var, eta_t * m_t / (v_sqrt + epsilon_t))
        else:
            var_update = state_ops.assign_sub(var, eta * m_t / (v_sqrt + epsilon_t))

        if self.laplace_mle:
            return control_flow_ops.group(*[var_update, m_t, v_t, sigma_t, mean_coef_t])
        else:
            return control_flow_ops.group(*[var_update, m_t, v_t, sigma_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient not supported.")