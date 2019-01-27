import minipyro as pyro

from optim import *
from elbo import *
from svi import *

import torch
import pyro.distributions as pdist


def model(data):
  """ Model : z_loc -> x

  Args
    data : list of scalars

  """
  z_loc = pyro.sample('z_loc', pdist.Normal(0., 1.))
  with pyro.plate('data', size=len(data), dim=-1):
    # normally distributed observations
    pyro.sample('obs', pdist.Normal(z_loc, 1.), obs=data)


def guide(data):
  """ Guide is a variational distribution """

  # define parameters
  #  loc and scale for latent variable `z_loc`
  guide_loc = pyro.param('guide_loc', torch.tensor(0.))
  guide_scale = pyro.param('guide_scale', torch.tensor(0.)).exp()

  # we would like to learn the distribution `loc`
  pyro.sample('z_loc', pdist.Normal(guide_loc, guide_scale))


def generate_data():
  return torch.randn(100) + 3.


if __name__ == '__main__':
  # generate data
  data = generate_data()

  # clear parameter store
  pyro.get_param_store().clear()

  # learning rate
  lr = 0.02
  # training steps
  num_steps = 1000

  # SVI for inference
  svi = SVI(model, guide, optim=Adam({'lr' : lr}), loss=elbo)

  for step in range(num_steps):
    loss = svi.step(data)
    if step%100 == 0:
      print('[{}] loss : {}'.format(step, loss))

  for name, value in pyro.get_param_store().items():
    print("{} = {}".format(name, value.detach().cpu().numpy()))
