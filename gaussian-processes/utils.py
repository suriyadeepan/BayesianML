import pickle
import logging

import numpy as np
import torch.distributions as dist


def save(d, filename):
  pickle.dump(d, open(filename, 'wb'))


def get_logger(name):
  # setup logger
  logging.basicConfig(level=logging.INFO)
  return logging.getLogger(name)


def split_dataset(dataset, ratio=0.8):
  """Split dataset into train set and test set

  Parameters
  ----------
  dataset : list
    List of data points
  ratio : float
    train/test split ratio

  Returns
  -------
  tuple
    train and test set lists
  """
  inputs, outputs = dataset
  n = len(inputs)
  m = int(n * ratio)
  return ( (inputs[:m], outputs[:m]), (inputs[m:], outputs[m:]) )


def shuffle(dataset):
  """Shuffle data points

  Parameters
  ----------
  dataset : tuple
    ( inputs, outputs ) tuple of list of inputs and outputs

  Returns
  -------
  tuple
    Tuple of shuffled lists of inputs and outputs
  """
  inputs, outputs = dataset
  shuffled_indices = np.arange(len(inputs))
  np.random.shuffle(shuffled_indices)
  return ( np.array(inputs, dtype='float32')[shuffled_indices],
      np.array(outputs, dtype='float32')[shuffled_indices]
      )


def bell(mu, sigma, x):
  # create normal
  norm = dist.Normal(mu, sigma)
  # bell curve from pdf
  psi = norm.log_prob(x).float().exp().numpy()
  # normalize
  psi = psi / psi.max()
  return psi
