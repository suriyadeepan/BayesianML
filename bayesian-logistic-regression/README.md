# Bayesian Logistic Regression

So, what has channged? Obviously the model has to include a non-linearity. I'm applying sigmoid.

```python
y_hat = torch.sigmoid((w * x).sum(dim=1) + b)
```

What else? `y` is binary variable 0 or 1. So, we sample from `Bernoulli` with probability `y_hat` while observing data.

```python
# observe data
with pyro.iarange('data', len(x)):
  # notice the Bernoulli distribution
  pyro.sample('obs', pdist.Bernoulli(y_hat), obs=y)
```

That's pretty much it. There isn't any change to the guide, except for `softplus` function applied to `w_scale` and `b_scale`. I do not know why.

**TODO** : figure out the purpose of `softplus`!

But the trained model produces `100%` accuracy on a held-out test set. How about that!

```
x : tensor([5.6000, 3.0000, 4.5000, 1.5000]), y vs y_hat : 1.0/1
x : tensor([4.4000, 2.9000, 1.4000, 0.2000]), y vs y_hat : 0.0/0
x : tensor([5.8000, 2.7000, 4.1000, 1.0000]), y vs y_hat : 1.0/1
x : tensor([4.6000, 3.6000, 1.0000, 0.2000]), y vs y_hat : 0.0/0
x : tensor([4.9000, 3.1000, 1.5000, 0.1000]), y vs y_hat : 0.0/0
x : tensor([5.1000, 3.8000, 1.9000, 0.4000]), y vs y_hat : 0.0/0
x : tensor([5.5000, 4.2000, 1.4000, 0.2000]), y vs y_hat : 0.0/0
x : tensor([5.5000, 2.4000, 3.8000, 1.1000]), y vs y_hat : 1.0/1
x : tensor([5.5000, 2.4000, 3.7000, 1.0000]), y vs y_hat : 1.0/1
x : tensor([4.9000, 3.1000, 1.5000, 0.1000]), y vs y_hat : 0.0/0
x : tensor([6.0000, 3.4000, 4.5000, 1.6000]), y vs y_hat : 1.0/1
x : tensor([5.5000, 2.6000, 4.4000, 1.2000]), y vs y_hat : 1.0/1
x : tensor([6.0000, 2.9000, 4.5000, 1.5000]), y vs y_hat : 1.0/1
x : tensor([5.0000, 3.4000, 1.6000, 0.4000]), y vs y_hat : 0.0/0
x : tensor([7.0000, 3.2000, 4.7000, 1.4000]), y vs y_hat : 1.0/1
x : tensor([6.6000, 3.0000, 4.4000, 1.4000]), y vs y_hat : 1.0/1
x : tensor([5.7000, 4.4000, 1.5000, 0.4000]), y vs y_hat : 0.0/0
x : tensor([5.9000, 3.0000, 4.2000, 1.5000]), y vs y_hat : 1.0/1
```
