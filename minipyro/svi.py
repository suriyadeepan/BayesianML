""" Stochastic Variational Inference

"""

import minipyro as pyro


class SVI(object):

  def __init__(self, model, guide, optim, loss):
    self.model = model
    self.guide = guide
    self.optim = optim
    self.loss  = loss

  def step(self, *args, **kwargs):
    """
    This is the workhorse of inference procedure

    [1] Run model
    [2] Run guide
    [3] Construct loss function
    [4] Update gradients

    Wrap call to model and guide in a `trace`, to record all the
    parameters that are encountered.

    """
    with pyro.trace() as param_capture:
      # record parameters only (of type "sample")
      with pyro.block(hide_fn=lambda msg: msg['type'] == 'sample'):
        loss = self.loss(self.model, self.guide, *args, **kwargs)

    # calculate gradients
    loss.backward()

    # fetch parameters from trace
    params = [ site['value'] for site in param_capture.values() ]

    # optimization step
    self.optim(params)

    # clear gradients
    for p in params:
      p.grad = p.new_zeros(p.shape)

    return loss.item()
