import pyro
import pyro.distributions as pdist
import pyro.optim as optim

import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

from pyro.infer import Trace_ELBO, SVI

from random import shuffle
import argparse
from tqdm import tqdm


# cmd line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='iris', help='Dataset : iris/mnist')
args, unknown = parser.parse_known_args()


def random_sample(t, k):
  x, y = t
  indices = torch.randperm(len(x))
  return x[indices][:k], y[indices][:k]


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


def mnist():
  trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
  # if not exist, download mnist dataset
  train = list(datasets.MNIST(root='.', train=True, transform=trans, download=True))
  test = list(datasets.MNIST(root='.', train=False, transform=trans, download=True))

  # convenience
  t = torch.tensor

  train = ( torch.cat([ d[0].view(1, 28*28) for d in train ], dim=0), t([ d[1] for d in train ]) )
  test = ( torch.cat([ d[0].view(1, 28*28) for d in test ], dim=0), t([ d[1] for d in test ]) )

  return train, test


def model(x, y, dim_in, dim_out):
  w = pyro.sample('w', pdist.Normal(torch.zeros(dim_in, dim_out), torch.ones(dim_in, dim_out)))
  b = pyro.sample('b', pdist.Normal(torch.zeros(1, dim_out), torch.ones(1, dim_out)))

  # define model [1, 3] x [4, 3] + [1, 3] = [1, 3]
  y_hat = torch.matmul(x, w) + b  # use `logits` directly in `Categorical()`

  # observe data
  # with pyro.plate('data', len(x), subsample_size=min(len(x), 100)) as idx:
  with pyro.plate('data'):
    # notice the Bernoulli distribution
    # pyro.sample('obs', pdist.Categorical(logits=y_hat),
    #  obs=y.index_select(0, idx))
    pyro.sample('obs', pdist.Categorical(logits=y_hat), obs=y)


def guide(x, y, dim_in, dim_out):
  # parameters of (w : weight)
  w_loc = pyro.param('w_loc', torch.zeros(dim_in, dim_out))
  w_scale = F.softplus(pyro.param('w_scale', torch.ones(dim_in, dim_out)))

  # parameters of (b : bias)
  b_loc = pyro.param('b_loc', torch.zeros(1, dim_out))
  b_scale = F.softplus(pyro.param('b_scale', torch.ones(1, dim_out)))

  # sample (w, b)
  w = pyro.sample('w', pdist.Normal(w_loc, w_scale))
  b = pyro.sample('b', pdist.Normal(b_loc, b_scale))


def inference(train_x, train_y, dim_in, dim_out, batch_size, num_epochs=20000):
  """ NOTE : there must be a better way to feed dim_in/dim_out
      perhaps we could infer them from train_x, train_y?
  """
  svi = SVI(model, guide, optim.Adam({'lr' : 0.005}),
      loss=Trace_ELBO(),
      num_samples=len(train_x)
      )

  for i in tqdm(range(num_epochs)):

    if batch_size > 0:  # random sample `batch_size` data points
      batch_x, batch_y = random_sample((train_x, train_y), batch_size)
    else:
      batch_x, batch_y = train_x, train_y  # feed the whole training set

    # run a step of SVI
    elbo = svi.step(batch_x, batch_y, dim_in, dim_out)

    if i % 100 == 0:
      print('Elbo loss : {}'.format(elbo))

  print('pyro\'s Param Store')
  for k, v in pyro.get_param_store().items():
    print(k, v)


def get_param(name):
  return pyro.get_param_store()[name]


if __name__ == '__main__':

  if args.dataset == 'iris':
    get_data, dim_in, dim_out, batch_size = iris, 4, 3, 0
  else:
    get_data, dim_in, dim_out, batch_size = mnist, 784, 10, 256

  # get data
  (train_x, train_y), (test_x, test_y) = get_data()
  # infer params
  inference(train_x, train_y, dim_in, dim_out, batch_size=batch_size)

  # parameters
  w, b = [ get_param(name) for name in ['w_loc', 'b_loc'] ]

  predict = lambda x : torch.argmax(torch.softmax(torch.matmul(x, w) + b, dim=-1))

  success = 0
  for xi, yi in zip(test_x, test_y):
    prediction = predict(xi.view(1, -1)).item()
    print('y vs y_hat : {}/{}'.format(int(yi), prediction))
    success += int(int(yi) == prediction)

  print(':: Accuracy >> ', 100. * success / len(test_x))
