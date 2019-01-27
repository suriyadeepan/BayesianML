"""

A thin wrapper around Adam Optimizer

Manage dynamically generated

"""
import torch


class Adam(object):

  def __init__(self, args):
    self.args = args
    # each parameter will get its own optimizer
    # we manage a dictionary of optimizers indexed by parameter
    self.optims = {}

  def __call__(self, params):
    for param in params:
      if param in self.optims:
        optim = self.optims[param]
      else:
        optim = torch.optim.Adam([param], **self.args)
        self.optims[param] = optim

      # run a gradient step of optimizer
      optim.step()
