# BADAM
Code which implements [BADAM](https://arxiv.org/abs/1811.03679).

The Adam algorithm uses a preconditioner which scales parameter updates by the square root of an  exponentially weighted moving average (EWMA) estimate of the squared gradients. Larger curvature in the loss landscape will mean that the step-sizes are very small and whether "there is greater uncertainty about the direction of the estimated gradient corresponds to the direction of the true gradient." Kingma et al. 

Likewise, the diagonal Hessian of the log-likelihood required for Laplace approximation around the optimal parameters of a NN, uses the squared gradients (Gauss-Newton). But importantly no square-root or EWMA unlike Adam. By interpreting adaptive subgradient methods in a Bayesian manner can we construct approximate posteriors by leveraging the preconditioners from the optimizers for cheaply and for free?
