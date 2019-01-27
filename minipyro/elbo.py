""" Evidence Lower Bound

"""

import minipyro as pyro


def elbo(model, guide, *args, **kwargs):
  # run trace on guide
  # args : same signature as SVI.step()
  # trace sites
  guide_trace = pyro.trace(guide).get_trace(*args, **kwargs)
  # run model with replay
  # reuse the sites from guide
  model_trace = pyro.trace(pyro.replay(model, guide_trace)).get_trace(*args, **kwargs)

  # construct elbo loss
  elbo_loss = 0.

  # log p(z) term from model_trace
  # iterate through samples sites in model
  for site in model_trace.values():
    if site['type'] == 'sample':
      # TODO : figure out how `log_prob` works
      log_p_z = site['fn'].log_prob(site['value']).sum()
      elbo_loss = elbo_loss + log_p_z

  # -log q(z) term from guide_trace
  # iterate through samples sites in model
  for site in guide_trace.values():
    if site['type'] == 'sample':
      log_q_z = -site['fn'].log_prob(site['value']).sum()
      elbo_loss = elbo_loss + log_q_z

  return -elbo_loss
