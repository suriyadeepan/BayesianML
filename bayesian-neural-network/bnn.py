import pyro
import pyro.distributions as pdist
import pyro.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

from pyro.infer import Trace_ELBO, SVI

from random import shuffle
import argparse


def random_sample(t, k):
  x, y = t
  indices = torch.randperm(len(x))
  return x[indices][:k], y[indices][:k]


def get_param(name):
  return pyro.get_param_store()[name]


def mnist():
  trans = transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]
      )
  # if not exist, download mnist dataset
  train = list(datasets.MNIST(root='.', train=True, transform=trans, download=True))
  test = list(datasets.MNIST(root='.', train=False, transform=trans, download=True))

  # convenience
  t = torch.tensor

  train = ( torch.cat([ d[0].view(1, 28 * 28) for d in train ], dim=0), t([ d[1] for d in train ]) )
  test = ( torch.cat([ d[0].view(1, 28 * 28) for d in test ], dim=0), t([ d[1] for d in test ]) )

  return train, test


class FFNet(nn.Module):
  """ Why go through the trouble of defining weight distributions
      and sampling from them? Let's just create an nn.Module and
      lift it. This way, when we sample a nn.Module, all the weights
      are sampled. We do not need to write sample statements.
      Sweet!
  """

  def __init__(self, dim_in, dim_hid, dim_out):
    super(FFNet, self).__init__()
    self.linear1 = nn.Linear(dim_in, dim_hid)
    self.linear2 = nn.Linear(dim_hid, dim_out)

  def forward(self, x):
    out = F.relu(self.linear1(x))
    return F.relu(self.linear2(out))


def model(x, y):
  # set prior on weights of `linear_1` and `linear_2`
  w1 = pdist.Normal(
      loc=torch.zeros_like(net.linear1.weight),
      scale=torch.ones_like(net.linear1.weight)
      )
  b1 = pdist.Normal(
      loc=torch.zeros_like(net.linear1.bias),
      scale=torch.ones_like(net.linear1.bias)
      )
  w2 = pdist.Normal(
      loc=torch.zeros_like(net.linear2.weight),
      scale=torch.ones_like(net.linear2.weight)
      )
  b2 = pdist.Normal(
      loc=torch.zeros_like(net.linear2.bias),
      scale=torch.ones_like(net.linear2.bias)
      )

  # a dictionary of priors
  priors = {
      'linear1.weight' : w1,
      'linear1.bias' : b1,
      'linear2.weight' : w2,
      'linear2.bias' : b2
      }

  # lift neural net module
  lifted_net = pyro.random_module("module", net, priors)

  # sample a net
  nn_model = lifted_net()

  # run the sampled model
  # y_hat = torch.log_softmax(nn_model(x))
  y_hat = nn_model(x)

  with pyro.plate('data'):
    pyro.sample('obs', pdist.Categorical(logits=y_hat), obs=y)


def guide(x, y):

  def build_param(p, name):
    mu = pyro.param('mu_{}'.format(name), torch.randn_like(p))
    sigma = F.softplus(
        pyro.param('sigma_{}'.format(name), torch.randn_like(p))
        )
    prior = pdist.Normal(mu, sigma)
    return prior, mu, sigma

  priors = {
      'linear1.weight' : build_param(net.linear1.weight, 'linear1_w')[0],
      'linear1.bias' : build_param(net.linear1.bias, 'linear1_b')[0],
      'linear2.weight' : build_param(net.linear2.weight, 'linear2_w')[0],
      'linear2.bias' : build_param(net.linear2.bias, 'linear2_b')[0]
      }

  # lift neural net module
  lifted_net = pyro.random_module("module", net, priors)

  return lifted_net()


def inference(train_x, train_y, batch_size, eval_fn=None, num_epochs=10000):

  svi = SVI(model, guide, optim.Adam({'lr' : 0.005}),
      loss=Trace_ELBO(),
      num_samples=len(train_x)
      )

  for i in range(num_epochs):

    if batch_size > 0:  # random sample `batch_size` data points
      batch_x, batch_y = random_sample((train_x, train_y), batch_size)
    else:
      batch_x, batch_y = train_x, train_y  # feed the whole training set

    # run a step of SVI
    elbo = svi.step(batch_x, batch_y)

    if i % 100 == 0:
      print('[{}/{}] Elbo loss : {}'.format(i, num_epochs, elbo))
      if eval_fn:
        print('Evaluation Accuracy : ', eval_fn())

  # print('pyro\'s Param Store')
  # for k, v in pyro.get_param_store().items():
  # print(k, v)


def evaluate(test_x, test_y):
  # get parameters
  # priors = { name : get_param(name) for name in
  #     [ 'mu_linear1.weight' , 'mu_linear1.bias',
  #       'mu_linear2.weight', 'mu_linear2.bias' ]
  #     }
  # build model for prediction
  # nn_model = pyro.random_module("module", net, priors)()

  nn_model = net
  net.linear1.weight = nn.Parameter(get_param('mu_linear1_w'))
  net.linear1.bias = nn.Parameter(get_param('mu_linear1_b'))
  net.linear2.weight = nn.Parameter(get_param('mu_linear2_w'))
  net.linear2.bias = nn.Parameter(get_param('mu_linear2_b'))

  def predict(x):
    return torch.argmax(torch.softmax(nn_model(x), dim=-1))

  success = 0
  for xi, yi in zip(test_x, test_y):
    prediction = predict(xi.view(1, -1)).item()
    success += int(int(yi) == prediction)

  return 100. * success / len(test_x)


if __name__ == '__main__':

  # MNIST configuration
  get_data, dim_in, dim_hid, dim_out, batch_size = mnist, 784, 256, 10, 256

  # create net
  net = FFNet(dim_in, dim_hid, dim_out)

  # get data
  (train_x, train_y), (test_x, test_y) = get_data()

  # infer params
  inference(train_x, train_y, batch_size=batch_size,
      eval_fn=lambda : evaluate(test_x, test_y)
      )
