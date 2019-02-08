import pyro
import pyro.distributions as pdist
import pyro.optim as optim

import torch
import torch.nn.functional as F

from pyro.infer import Trace_ELBO, SVI

from random import shuffle


def iris(datafile='./iris.data'):
  # label to index lookup
  label2idx = { 'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2 }
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
  w = pyro.sample('w', pdist.Normal(torch.zeros(4, 3), torch.ones(4, 3)))
  b = pyro.sample('b', pdist.Normal(torch.zeros(1, 3), torch.ones(3)))

  # define model [1, 3] x [4, 3] + [1, 3] = [1, 3]
  y_hat = torch.matmul(x, w) + b  # use `logits` directly in `Categorical()`

  # observe data
  with pyro.iarange('data', len(x)):
    # notice the Bernoulli distribution
    pyro.sample('obs', pdist.Categorical(logits=y_hat), obs=y)


def guide(x, y):
  # parameters of (w : weight)
  w_loc = pyro.param('w_loc', torch.zeros(4, 3))
  w_scale = F.softplus(pyro.param('w_scale', torch.ones(4, 3)))

  # parameters of (b : bias)
  b_loc = pyro.param('b_loc', torch.zeros(1, 3))
  b_scale = F.softplus(pyro.param('b_scale', torch.ones(1, 3)))

  # sample (w, b)
  w = pyro.sample('w', pdist.Normal(w_loc, w_scale))
  b = pyro.sample('b', pdist.Normal(b_loc, b_scale))


def inference(train_x, train_y, num_epochs=20000):
  svi = SVI(model, guide, optim.Adam({'lr' : 0.0008}),
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

  predict = lambda x : torch.argmax(torch.softmax(torch.matmul(x, w) + b, dim=-1))

  for xi, yi in zip(test_x, test_y):
    print('x : {}, y vs y_hat : {}/{}'.format(
      xi, int(yi), predict(xi.view(1, -1)).item()
      # xi, yi, int(predict(xi.view(1, -1)).item() > 0.5)
      ))
