import torch
import utils

import numpy as np


class GMM:

  def __init__(self, data, K):
    self.K = K
    self.data = data
    self.init_params()

  def init_params(self, var=1):
    # get len of dataset
    N = self.data.size(0)  # [2000, 2] -> 2000
    # num components
    K = self.K
    # Set mu as randomly selected K datapoints
    indices = torch.from_numpy(np.random.choice(N, K, replace=False))
    self.mu = self.data[indices]
    # set variances
    self.var = torch.Tensor(K, 2).fill_(var)
    # uniform prior >> 1/K
    self.pi = torch.Tensor(K).fill_(1. / K)

    return self.mu, self.var, self.pi

  def log_gaussian(self, data, mu, logvar):
    # norm constant
    log_norm_constant = -0.5 * np.log(2 * np.pi)
    # calc logp
    a = (data - mu) ** 2
    log_p = -0.5 * (logvar + a / logvar.exp())
    log_p = log_p + log_norm_constant

    return log_p

  def get_likelihood(self, data, mu, logvar, log=True):
    # reshape data and parameters for likelihood estimation
    data = data[None, :, :]
    mu = mu[:, None, :]
    logvar = logvar[:, None, :]
    # get likelihood
    log_likelihoods = self.log_gaussian(data, mu, logvar)
    # sum up log-likelihoods along feature-axis
    log_likelihoods = log_likelihoods.sum(-1)
    # do we apply log?
    if not log:
      log_likelihoods.exp_()

    return log_likelihoods

  def get_posterior(self, log_likelihoods):
    return log_likelihoods - utils.logsumexp(log_likelihoods, dim=0, keepdim=True)

  def step(self):
    # calculate likelihood
    log_likelihoods = self.get_likelihood(self.data, self.mu, self.var.log())
    # get posteriors
    posteriors = self.get_posterior(log_likelihoods)
    # update parameters
    self.update(posteriors.exp())

    return log_likelihoods.mean()

  def update(self, posteriors, eps=1e-6, min_var=1e-6):
    K = self.K
    # sum up posteriors along feature dimension
    p_k = torch.sum(posteriors, dim=1)
    p_k = p_k.view(K, 1, 1)
    # get the means by taking the weighted combination of points
    # (K, 1, examples) @ (1, examples, features) -> (K, 1, features)
    mu = posteriors[:, None] @ self.data[None,]
    mu = mu / (p_k + eps)

    # compute the diagonal covar. matrix, by taking a weighted combination of
    # the each point's square distance from the mean
    A = self.data - mu
    var = posteriors[:, None] @ (A ** 2)  # (K, 1, features)
    var = var / (p_k + eps)
    logvar = torch.clamp(var, min=min_var).log()

    # recompute the mixing probabilities
    self.pi = (p_k / p_k.sum()).squeeze()
    self.mu = mu.squeeze(1)
    logvar.squeeze(1)
