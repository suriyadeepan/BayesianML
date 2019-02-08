import pyro
import torch
import pyro.distributions as pdist
import pyro.optim as optim

from pyro.infer import Trace_ELBO, SVI

import pandas as pd
from pyro.infer import EmpiricalMarginal
from functools import partial


def noisy():
  x = torch.arange(-5, 5, 0.1)
  return x, 2. * x + .3 + torch.normal(torch.tensor(0.), torch.tensor(0.001))


def model(x, y):
  w = pyro.sample('w', pdist.Normal(0., 1.))
  b = pyro.sample('b', pdist.Normal(0., 1.))
  # define model
  y_hat = w * x + b
  # variance of distribution centered around y
  # sigma = pyro.sample('sigma', pdist.Uniform(0., 10.))
  sigma = pyro.sample('sigma', pdist.Normal(0., 1.))
  with pyro.iarange('data', len(x)):
    pyro.sample('obs', pdist.Normal(y_hat, sigma), obs=y)


def guide(x, y):
  # parameters of (w : weight)
  w_loc = pyro.param('w_loc', torch.tensor(0.))
  w_scale = pyro.param('w_scale', torch.tensor(1.))
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


def inference(train_x, train_y, num_epochs=2000):
  svi = SVI(model, guide, optim.Adam({'lr' : 0.005}),
      loss=Trace_ELBO(),
      num_samples=1000
      )

  for i in range(num_epochs):
    elbo = svi.step(train_x, train_y)
    if i % 200 == 0:
      print('Elbo loss : {}'.format(elbo))

  svi_posterior = svi.run(train_x, train_y)
  sites = [ 'w', 'b', 'sigma' ]
  for site, values in summary(svi_posterior, sites).items():
    print("Site: {}".format(site))
    print(values, "\n")


def summary(traces, sites):
    marginal = EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()
    site_stats = {}
    for i in range(marginal.shape[1]):
        site_name = sites[i]
        marginal_site = pd.DataFrame(marginal[:, i]).transpose()
        describe = partial(pd.Series.describe,
            percentiles=[.05, 0.25, 0.5, 0.75, 0.95]
            )
        site_stats[site_name] = marginal_site.apply(describe, axis=1) \
            [["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats


if __name__ == '__main__':
  # get data
  train_x, train_y = noisy()

  # for x, y in zip(train_x, train_y):
  #   print('|  {:.2f} | {:.3f} |'.format(x.item(), y.item()))

  inference(train_x, train_y)
