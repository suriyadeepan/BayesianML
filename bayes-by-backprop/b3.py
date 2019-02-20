import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.optim import Adam
from torchvision import datasets
from torchvision import transforms

import math
from tqdm import tqdm

# mixing coefficient for Scaled Gaussian Mixture
PI = 0.5
SIGMA1 = 0.2
SIGMA2 = 0.001


class Gaussian:
  """ Reparameterized Gaussian


  """
  def __init__(self, mu, rho):
    self.mu = torch.tensor(mu) if isinstance(mu, type(6.9)) else mu
    self.rho = torch.tensor(rho) if isinstance(rho, type(9.6)) else rho
    self.normal = dist.Normal(0, 1)

  @property
  def sigma(self):
    # log1p <- ln(1 + input)
    #  why we need a function for this, is beyond me.
    return torch.log1p(torch.exp(self.rho))  # ln(1 + input)

  def sample(self):
    epsilon = self.normal.sample(self.mu.size())
    return self.mu + self.sigma * epsilon

  def log_prob(self, input):
    return (-math.log(math.sqrt(2 * math.pi))
        - torch.log(self.sigma)
        - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class ScaledGaussianMixture:
  """ Scaled Mixture of Gaussians

  We use an engineered mixture of gaussians for the prior distribution

  mu1 = mu2 = 0
  rho1 > rho2
  rho2 << 1

  """
  def __init__(self, pi, sigma1, sigma2):
    self.pi = pi
    self.sigma1 = sigma1
    self.sigma2 = sigma2
    self.gaussian1 = dist.Normal(0, sigma1)
    self.gaussian2 = dist.Normal(0, sigma2)

  def log_prob(self, input):
    prob1 = torch.exp(self.gaussian1.log_prob(input))
    prob2 = torch.exp(self.gaussian1.log_prob(input))
    return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()


class BayesianLinear(nn.Module):

  def __init__(self, dim_in, dim_out):
    super(BayesianLinear, self).__init__()
    self.dim_in = dim_in
    self.dim_out = dim_out

    # weight distribution
    self.w_mu = nn.Parameter((-0.2 - 0.2) * torch.rand(dim_out, dim_in) + 0.2)
    self.w_rho = nn.Parameter((-5. + 4.) * torch.rand(dim_out, dim_in) - 4.)
    self.w = Gaussian(self.w_mu, self.w_rho)

    # bias distribution
    self.b_mu = nn.Parameter((-0.2 - 0.2) * torch.rand(dim_out) + 0.2)
    self.b_rho = nn.Parameter((-5. + 4.) * torch.rand(dim_out) - 4.)
    self.b = Gaussian(self.b_mu, self.b_rho)

    # prior distribution
    self.w_prior = ScaledGaussianMixture(PI, SIGMA1, SIGMA2)
    self.b_prior = ScaledGaussianMixture(PI, SIGMA1, SIGMA2)
    self.log_prior = 0
    self.log_variational_posterior = 0  # q

  def forward(self, input, sample=False, calc_log_prob=False):

    if self.training or sample:  # while training or sampling
      w = self.w.sample()
      b = self.b.sample()
    else:
      w = self.w.mu
      b = self.b.mu

    if self.training or calc_log_prob:
      # calculate logprob of prior for sampled weights
      self.log_prior = self.w_prior.log_prob(w) + self.b_prior.log_prob(b)
      # calculate logprob of posterior (w, b) distributions
      self.log_variational_posterior = self.w.log_prob(w) + self.b.log_prob(b)
    else:
      self.log_prior, self.log_variational_posterior = 0, 0  # coz we ain't training

    return F.linear(input, w, b)


class BayesianNeuralNet(nn.Module):
  """ Bayesian Neural Network

  A network built of `BayesianLinear` layers

  """

  def __init__(self, dim_in, dim_hid, dim_out):
    super(BayesianNeuralNet, self).__init__()
    self.linear1 = BayesianLinear(dim_in, dim_hid)
    self.linear2 = BayesianLinear(dim_hid, dim_hid)
    self.linear3 = BayesianLinear(dim_hid, dim_out)
    # expose dims
    self.dim_out = dim_out

  def forward(self, x, sample=False):
    x = F.relu(self.linear1(x, sample))
    x = F.relu(self.linear2(x, sample))
    x = F.log_softmax(self.linear3(x, sample), dim=1)
    return x

  def log_prior(self):
    return self.linear1.log_prior\
        + self.linear2.log_prior\
        + self.linear3.log_prior

  def log_variational_posterior(self):
    return self.linear1.log_variational_posterior\
        + self.linear2.log_variational_posterior\
        + self.linear3.log_variational_posterior

  def sample_elbo(self, input, target, num_samples=2, batch_size=64):
    outputs = torch.zeros(num_samples, batch_size, self.dim_out)
    log_priors = torch.zeros(num_samples)
    log_variational_posteriors = torch.zeros(num_samples)
    for i in range(num_samples):
      outputs[i] = self(input, sample=True)  # run forward
      log_priors[i] = self.log_prior()
      log_variational_posteriors[i] = self.log_prior()

    # average log-priors and log-posteriors
    log_prior = log_priors.mean()
    log_variational_posterior = log_variational_posteriors.mean()

    # calculate NLL loss
    nll = F.nll_loss(outputs.mean(0), target, size_average=False)
    # calculate KL divergence
    kl = (log_variational_posterior - log_prior) / 10

    # total loss
    return kl + nll


def train_epoch(net, trainset, optim, batch_size):
  net.train()  # train-mode
  train_x, train_y = trainset
  # iterations = len(train_x) // batch_size
  iterations = 300

  epoch_loss = 0
  for idx in tqdm(range(iterations)):
    optim.zero_grad()
    batch_x = train_x[idx * batch_size : (idx + 1) * batch_size ]
    batch_y = train_y[idx * batch_size : (idx + 1) * batch_size ]
    loss = net.sample_elbo(batch_x, batch_y, batch_size=batch_size)
    loss.backward()
    optim.step()
    epoch_loss += loss.item()

  return epoch_loss / iterations


def train(net, dataset, batch_size, num_epochs=100):
  trainset, testset = dataset
  optim = Adam(net.parameters())

  for epoch in range(num_epochs):
    epoch_loss = train_epoch(net, trainset, optim, batch_size=batch_size)
    print('[{}] train : {:10.4f}; eval : {:10.4f}%'.format(
      epoch, epoch_loss,
      evaluate_ensemble(net, testset, batch_size=batch_size)
      ))


def evaluate(net, testset, batch_size):
  net.eval()  # train-mode
  test_x, test_y = testset
  # iterations = len(test_x) // batch_size
  iterations = 30

  epoch_loss = 0
  for idx in range(iterations):
    batch_x = test_x[idx * batch_size : (idx + 1) * batch_size ]
    batch_y = test_y[idx * batch_size : (idx + 1) * batch_size ]
    loss = net.sample_elbo(batch_x, batch_y, batch_size=batch_size)
    epoch_loss += loss.item()

  return epoch_loss / iterations


def evaluate_ensemble(net, testset, batch_size=64):
  net.eval()
  test_x, test_y = testset
  iterations = 80
  corrects = 0
  for idx in range(iterations):
      batch_x = test_x[idx * batch_size : (idx + 1) * batch_size ]
      batch_y = test_y[idx * batch_size : (idx + 1) * batch_size ]
      outputs = []
      for i in range(3):
          outputs.append(net(batch_x, sample=True))
      outputs.append(net(batch_x, sample=False))
      av_output = torch.stack(outputs, dim=0).mean(0)
      preds = av_output.argmax(dim=1)
      corrects += (preds == batch_y).float().sum()

  return 100. * corrects / (iterations * batch_size)


def mnist():
  trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
  # if not exist, download mnist dataset
  train = list(datasets.MNIST(root='.', train=True, transform=trans, download=True))
  test = list(datasets.MNIST(root='.', train=False, transform=trans, download=True))

  # convenience
  t = torch.tensor

  train = ( torch.cat([ d[0].view(1, 28 * 28) for d in train ], dim=0),
      t([ d[1] for d in train ]) )
  test = ( torch.cat([ d[0].view(1, 28 * 28) for d in test ], dim=0),
      t([ d[1] for d in test ]) )

  return train, test


if __name__ == '__main__':

  # MNIST config
  get_data, dim_in, dim_out, batch_size = mnist, 784, 10, 256
  # get data
  trainset, testset = get_data()
  # instantiate model
  bnn = BayesianNeuralNet(28 * 28, 400, 10)
  # training
  train(bnn, (trainset, testset), batch_size=64)
