# Bayesian Machine Learning

My experiments in Bayesian Machine Learning.
I make use of [pyro](http://pyro.ai/)'s ppl constructs for building models and inference (`pyro.infer.SVI`).
Every experiment is in a separate folder with documentation in `README`.
The list of experiments are given below.

- [x] [Deconstructing Pyro](minipyro/README.md)
- [x] [Bayesian Linear Regression](bayesian-linear-regression)
  - [x] [Multivariate Linear Regression](bayesian-linear-regression/README.md#multivariate-linear-regression)
- [x] [Bayesian Logistic Regression](bayesian-logistic-regression)
  - [x] [Binary Classification](bayesian-logistic-regression)
  - [x] [Multiclass Classification](bayesian-logistic-regression/README.md#multi-class-logistic-regression)
  - [x] [MNIST Classification](bayesian-logistic-regression/README.md#mnist)
- [ ] [Bayes by Backprop](bayes-by-backprop/)
  - [x] [Bayesian Neural Network for MNIST Classification](bayes-by-backprop/b3.py)
  - [ ] Out-of-domain Uncertainty
- [ ] [Bayesian Neural Network](bayesian-neural-network/)
- [ ] [Bayesian Convolutional Neural Network](#)
- [x] [Gaussian Mixture Model](gmm/)
- [ ] [Pyro Demystified](on-pyro/)
- [ ] [Introduction to PyMC3](pymc3-intro)
- [x] [Gaussian Processes](gaussian-processes/)
  - [x] [GPytorch](gaussian-processes/GPytorch.ipynb)
  - [ ] [Simulation of Bose Einstein Condensates with Gaussian Processes](gaussian-processes/BEC.ipynb)

## Related Work

- [Anatomy of Probabilistic Programming Languages](https://github.com/suriyadeepan/anatomy-of-ppl)

## How to Contribute?

Would you like a question (relevant to Bayes ML) answered? Rise an issue and I'll add an experiment to answer it (if necessary).
