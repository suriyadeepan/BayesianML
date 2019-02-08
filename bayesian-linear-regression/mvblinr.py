import pyro
import torch
import pyro.distributions as pdist
import pyro.optim as optim

from pyro.infer import Trace_ELBO, SVI
from torch.distributions import constraints


def shuffle(t):
  rand_indices = torch.randperm(len(t))
  return t[rand_indices]


def noisy():
  x0 = shuffle(torch.arange(-5, 5, 0.1).view(-1, 1))
  x1 = shuffle(torch.arange(-5, 5, 0.1).view(-1, 1))
  x2 = shuffle(torch.arange(-5, 5, 0.1).view(-1, 1))
  x = torch.cat([x0, x1, x2], 1)
  w = torch.tensor([ 1., 2., 3. ])
  return (x,
      (w * x).sum(dim=1) + .7  + torch.normal(torch.tensor(0.), torch.tensor(0.001))
      )


def model(x, y):
  w = pyro.sample('w', pdist.Normal(torch.zeros(3), torch.ones(3)))
  b = pyro.sample('b', pdist.Normal(0., 1.))
  # define model
  y_hat = (w * x).sum(dim=1) + b
  # variance of distribution centered around y
  sigma = pyro.sample('sigma', pdist.Normal(0.5, 1.))
  with pyro.iarange('data', len(x)):
    pyro.sample('obs', pdist.Normal(y_hat, sigma), obs=y)


def guide(x, y):
  # parameters of (w : weight)
  w_loc = pyro.param('w_loc', torch.zeros(3))
  w_scale = pyro.param('w_scale', torch.ones(3))

  # parameters of (b : bias)
  b_loc = pyro.param('b_loc', torch.tensor(0.))
  b_scale = pyro.param('b_scale', torch.tensor(1.))
  # parameters of (sigma)
  sigma_loc = pyro.param('sigma_loc', torch.tensor(1.))
  sigma_scale = pyro.param('sigma_scale', torch.tensor(0.05))

  # sample (w, b, sigma)
  w = pyro.sample('w', pdist.Normal(w_loc, w_scale))
  b = pyro.sample('b', pdist.Normal(b_loc, b_scale))
  sigma = pyro.sample('sigma', pdist.Normal(sigma_loc, sigma_scale))

  # build model
  # y_hat = w * x + b


def inference(train_x, train_y, num_epochs=10000):
  svi = SVI(model, guide, optim.Adam({'lr' : 0.001}),
      loss=Trace_ELBO(),
      num_samples=1000
      )

  for i in range(num_epochs):
    elbo = svi.step(train_x, train_y)
    if i % 1000 == 0:
      print('Elbo loss : {}'.format(elbo))

  print('pyro\'s Param Store')
  for k, v in pyro.get_param_store().items():
    print(k, v)


if __name__ == '__main__':
  # get data
  train_x, train_y = noisy()

  # for x, y in zip(train_x, train_y):
  #  print('| {:.2f} | {:.2f} | {:.2f} | {:.3f} |'.format(x[0].item(), x[1].item(), x[2].item(), y.item()))

  inference(train_x, train_y)
