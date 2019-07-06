import torch

import numpy as np
from data import load
from random import randint

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from matplotlib import pyplot as plt
plt.style.use('ggplot')

def to_np(t): return t.data.numpy()  # torch to numpy
def to_to(a): return torch.tensor(a).float()  # numpy to torch


class DataLoader():

  def __init__(self, filename='bec1d.data', num_samples=None):
    # read from file
    self.data_g, self.data_y, reference = load(filename)
    self.x = reference['x']
    self.x_size = len(self.x)
    # prepare training data
    if num_samples:
      self.prepare_train_data(num_samples)

  def prepare_train_data(self, num_samples=None):
    # set sizes
    N = len(self.data_g)
    M = num_samples if num_samples else N
    assert N == 50000
    assert self.x_size == 512
    # keep track of sizes
    self.N, self.M = N, M
    # convert to torch tensors
    g = torch.tensor(self.data_g).float()
    x = torch.tensor(self.x).float()
    y = torch.tensor(self.data_y).float()
    # combine g and x
    gx = torch.stack(
        [g.view(1, -1).repeat(self.x_size, 1).t(), x.repeat(N, 1)]
    ).permute(1, 0, 2)
    gxy = torch.cat([gx, y.view(N, 1, -1)], dim=1)
    gxy_tensor_short = gxy.permute(1, 0, 2)[:M].contiguous().view(3, -1).t()
    # sample from gx tensor
    gx_samples = gxy_tensor_short[torch.randint(0, gxy_tensor_short.size(0), (M,))]
    X = gx_samples[:, :2]
    y = gx_samples[:, -1]
    # size check
    assert X.size() == (M, 2)
    assert y.size() == (M, )

    self.train_X, self.train_y = to_np(X), to_np(y)

    return self.train_X, self.train_y

  def prepare_prediction_data(self, idx=None):
    # choose a random row if idx is None
    idx = idx if idx else randint(0, self.N)
    # select row
    x_pred = self.x
    g_pred = to_to(self.data_g[idx])
    y_pred = self.data_y[idx]
    # combine g and x
    g_pred = g_pred.repeat(self.x_size)
    gx_pred = torch.stack([g_pred, to_to(x_pred)]).t()
    # shape check
    assert len(x_pred) == self.x_size
    assert gx_pred.shape == (self.x_size, 2)
    assert y_pred.shape[-1] == self.x_size

    self.pred_gx = gx_pred
    self.pred_y = y_pred

    return x_pred, gx_pred, y_pred


class GPWrapper():

  def __init__(self):
    # setup kernel
    self.kernel = C(1.0, (1e-3, 1e3)) * RBF([5, 5], (1e-2, 1e2))
    self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=15)

  def fit(self, X, y):
    assert X.shape[-1] == 2
    assert X.shape[0] == y.shape[-1]
    self.gp.fit(X, y)

  def predict(self, x, *args, **kwargs):
    assert x.shape[-1] == 2
    return self.gp.predict(x, *args, **kwargs)


def plot_gp_prediction(x, y, y_pred_gp, sigma):
  # Plot the function, the prediction and the 95% confidence interval based on
  # the MSE
  plt.figure()
  plt.plot(x, y, 'r:', label='waveform')

  # plt.plot(x, y, 'r.', markersize=10, label='Observations')
  plt.plot(x, y_pred_gp, 'b-', label='Prediction')

  plt.fill(np.concatenate(
    [x, x[::-1]]),
    np.concatenate(
      [y_pred_gp - 1.9600 * sigma, (y_pred_gp + 1.9600 * sigma)[::-1]]
      ),
    alpha=.5, fc='b', ec='None', label='95% confidence interval')

  plt.xlabel('$x$')
  plt.ylabel('$\psi\ (x)$')
  plt.ylim(-0.1, 1.1)
  plt.legend(loc='upper left')
  plt.show()


if __name__ == '__main__':

  # create data loader
  dldr = DataLoader('bec1d.data')
  # prepare training data
  NUM_TRAINING_SAMPLES = 300
  trainX, trainY = dldr.prepare_train_data(NUM_TRAINING_SAMPLES)
  # create GP wrapper
  gpwrapper = GPWrapper()
  # fit GP
  gpwrapper.fit(trainX, trainY)
  # prepare prediction data for a random `g`
  predX, predgX, predY = dldr.prepare_prediction_data()
  # predict with GP
  y_pred_gp, sigma = gpwrapper.predict(predgX, return_std=True)
  # plot predictions
  plot_gp_prediction(dldr.x, predY, y_pred_gp, sigma)
