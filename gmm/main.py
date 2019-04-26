from gmm import GMM

import torch
import utils


def sample(mu, var, N=500):
  return torch.stack(
      [ torch.normal(mu, var.sqrt()) for i in range(N) ],
      dim=0
  )


def generate_3_clusters():
  # generate 3 clusters
  c1 = sample(torch.Tensor([2.5, 2.5]), torch.Tensor([1.2, .8]), 500)
  c2 = sample(torch.Tensor([7.5, 7.5]), torch.Tensor([.75, .5]), 500)
  c3 = sample(torch.Tensor([8, 1.5]), torch.Tensor([.6, .8]), 1000)

  return torch.cat([c1, c2, c3])


if __name__ == '__main__':
  # generate data
  data = generate_3_clusters()
  # 3 components
  K = 3
  # create model
  gm = GMM(data, K=3)
  # training iterations
  iterations = 50
  # early stopping threshold
  thresh = 1e-6

  loss_p = 100000.
  for i in range(iterations):
    # run a step
    loss_c = gm.step()
    print(f'[{i}] Loss : {loss_c}')
    # difference
    if torch.abs(loss_c - loss_p).item() < thresh:
      print('Early Stopping')
      break
    # keep track of previous
    loss_p = loss_c

    # get likelihood
    utils.plot_density(*utils.get_density(
      gm.mu, gm.var.log(), gm.pi, gm.get_likelihood, N=100,
      X_range=(-2, 12), Y_range=(-2, 12)),
      i=i)

  # get likelihood
  utils.plot_density(*utils.get_density(
    gm.mu, gm.var.log(), gm.pi, gm.get_likelihood, N=100,
    X_range=(-2, 12), Y_range=(-2, 12)),
    i=i, show=True)
