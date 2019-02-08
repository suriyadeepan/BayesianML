import pyro
import pyro.distributions as pdist
import pyro.optim as optim

import torch
import torch.nn.functional as F

from pyro.infer import Trace_ELBO, SVI

from random import shuffle


def iris(datafile='./iris.data'):
  # label to index lookup
  # label2idx = { 'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2 }
  label2idx = { 'Iris-setosa' : 0, 'Iris-versicolor' : 1 }
  lines = [ l.replace('\n', '').strip() for l in open(datafile).readlines() ]
  # shuffle lines
  shuffle(lines)
  features, labels = [], []
  for line in lines:
    # super-annoying empty last line
    if not line:
      break

    items = line.split(',')
    label = items[-1]

    # check if label is in label2idx
    if label not in label2idx:
      continue

    features.append([ float(i) for i in items[:-1] ])
    labels.append(label2idx[label])

  # train/test separation
  k = int(0.8 * len(features))
  train_x, train_y = features[:k], labels[:k]
  test_x, test_y = features[k:], labels[k:]
  # convenience
  t = torch.tensor

  return (
      ( t(train_x), t(train_y).float() ),
      ( t(test_x), t(test_y).float() )
      )


def model(x, y):
  w = pyro.sample('w', pdist.Normal(torch.zeros(4), torch.ones(4)))
  b = pyro.sample('b', pdist.Normal(0., 1.))

  # define logistic regression model
  y_hat = torch.sigmoid((w * x).sum(dim=1) + b)

  # variance of distribution centered around y
  sigma = pyro.sample('sigma', pdist.Normal(0.5, 1.))

  # observe data
  with pyro.iarange('data', len(x)):
    # notice the Bernoulli distribution
    pyro.sample('obs', pdist.Bernoulli(y_hat), obs=y)


def guide(x, y):
  # parameters of (w : weight)
  w_loc = pyro.param('w_loc', torch.zeros(4))
  w_scale = F.softplus(pyro.param('w_scale', torch.ones(4)))

  # parameters of (b : bias)
  b_loc = pyro.param('b_loc', torch.tensor(0.))
  b_scale = F.softplus(pyro.param('b_scale', torch.tensor(1.)))
  # parameters of (sigma)
  sigma_loc = pyro.param('sigma_loc', torch.tensor(1.))
  sigma_scale = pyro.param('sigma_scale', torch.tensor(0.05))

  # sample (w, b, sigma)
  w = pyro.sample('w', pdist.Normal(w_loc, w_scale))
  b = pyro.sample('b', pdist.Normal(b_loc, b_scale))
  sigma = pyro.sample('sigma', pdist.Normal(sigma_loc, sigma_scale))


def inference(train_x, train_y, num_epochs=10000):
  svi = SVI(model, guide, optim.Adam({'lr' : 0.001}),
      loss=Trace_ELBO(),
      num_samples=len(train_x)
      )

  for i in range(num_epochs):
    elbo = svi.step(train_x, train_y)
    if i % 1000 == 0:
      print('Elbo loss : {}'.format(elbo))

  print('pyro\'s Param Store')
  for k, v in pyro.get_param_store().items():
    print(k, v)


def get_param(name):
  return pyro.get_param_store()[name]


if __name__ == '__main__':
  # get data
  (train_x, train_y), (test_x, test_y) = iris()
  # infer params
  inference(train_x, train_y)

  # parameters
  w, b = [ get_param(name) for name in ['w_loc', 'b_loc'] ]

  predict = lambda x : torch.sigmoid((w * x).sum(dim=1) + b)

  for xi, yi in zip(test_x, test_y):
    print('x : {}, y vs y_hat : {}/{}'.format(
      xi, yi, int(predict(xi.view(1, -1)).item() > 0.5)
      ))
