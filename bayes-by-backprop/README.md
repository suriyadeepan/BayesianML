# Bayes by Backprop

- We approximate `p` with `q` parameterized by `theta`
- Sample from `q` when we see data
- We come up with a tractable objective function

```python
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
```

- Prior is a scaled mixture of Gaussians
