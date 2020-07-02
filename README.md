# BADAM
Code which implements [BADAM](https://arxiv.org/abs/1811.03679).

The Adam algorithm uses a preconditioner which scales parameter updates by the square root of an exponentially weighted moving average (EWMA) estimate of the squared gradients. This can be readily translated into a geometric notion of uncertainty, if this uncertainty is 
 large then the stepsize be close to zero. "This is a desirable property, since a smaller SNR means that
there is greater uncertainty about whether the direction of $\hat{m}_t$ corresponds to the direction of the true
gradient. For example, the SNR value typically becomes closer to 0 towards an optimum, leading
to smaller effective steps in parameter space" Kingma et al. 2014.  

Likewise, the diagonal Hessian of the log-likelihood required for Laplace approximation around the optimal parameters of a NN, 
uses the squared gradients (Generalized Gauss-Newton approximation). But importantly no square-root or EWMA unlike Adam. 
By interpreting adaptive subgradient methods in a Bayesian manner can we construct approximate posteriors by leveraging the
 preconditioners from these optimizers, cheaply and for free?
 
 ## Setup
 
 Required packages are listed in `requirements.txt` which contains instructions to create a conda env with the correct versions. Run:
 
 `mkdir logs`
 
 `mkdir plots`
 
 ## Method
 
 There are a couple different ways to use the curvature estimates from Adam as an estimate of uncertainty in the loss landscape.
 One can perform a Laplace approximation of the likelihood and then introduce a Gaussian prior and hence the posterior will be Gaussian: final weight distribution
 <img src="https://render.githubusercontent.com/render/math?math=\mathcal{N}\left( \theta \,\bigg|\, \left( \frac{N \mathrm{diag}(v_t^{1/2})}{N \mathrm{diag}(v_t^{1/2}) %2B 1/ \sigma^2 } \right) \theta_{t}, \, \frac{1}{N \mathrm{diag}(v_t^{1/2}) %2B 1 / \sigma^2} \right)">
 
 Alternatively one can introduce an explicit L^2 regularisation for thr weights of a NN and then the curvature estimates
 can be used for a Laplace approximation around the posterior: <img src="https://render.githubusercontent.com/render/math?math=\mathcal{N}\left( \theta \,|\, \theta_{t}, \, (N \mathrm{diag}(v_t^{1/2}))^{-1} \right)">
 
 For pruning the exact version doesn't matter too much since the signal to noise ratios are equal. On the other hand for the
 regression experiments the Laplace around the posterior version of Badam works better. The mean 
 coefficient of the the first version can zero out some important weights in small NN models used for regression experiments.
 
 ## Disclaimer
 
 I haven't had time to tidy up the code, so please get in touch if you have any questions.
