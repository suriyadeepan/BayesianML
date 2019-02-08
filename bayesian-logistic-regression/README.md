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

# Multi-class Logistic Regression

What has changed? We use `Categorical` instead of `Bernoulli`.
The model has to be changed to output a 3-d vector (3 classes).

```python
w = pyro.sample('w', pdist.Normal(torch.zeros(4, 3), torch.ones(4, 3)))
b = pyro.sample('b', pdist.Normal(torch.zeros(1, 3), torch.ones(3)))

# define model [1, 3] x [4, 3] + [1, 3] = [1, 3]
y_hat = torch.matmul(x, w) + b  # use `logits` directly in `Categorical()`
```

The dimensions of mean and variance of of `w` and `b` must be adjusted accordingly, in `guide()`.

```python
# parameters of (w : weight)
w_loc = pyro.param('w_loc', torch.zeros(4, 3))
w_scale = F.softplus(pyro.param('w_scale', torch.ones(4, 3)))

# parameters of (b : bias)
b_loc = pyro.param('b_loc', torch.zeros(1, 3))
b_scale = F.softplus(pyro.param('b_scale', torch.ones(1, 3)))
```

And Voila! We have achieved `100%` accuracy on IRIS dataset.

```
x : tensor([4.7000, 3.2000, 1.3000, 0.2000]), y vs y_hat : 0/0
x : tensor([5.4000, 3.9000, 1.7000, 0.4000]), y vs y_hat : 0/0
x : tensor([6.4000, 3.1000, 5.5000, 1.8000]), y vs y_hat : 2/2
x : tensor([4.9000, 3.1000, 1.5000, 0.1000]), y vs y_hat : 0/0
x : tensor([6.0000, 3.0000, 4.8000, 1.8000]), y vs y_hat : 2/2
x : tensor([5.0000, 3.0000, 1.6000, 0.2000]), y vs y_hat : 0/0
x : tensor([5.8000, 2.7000, 4.1000, 1.0000]), y vs y_hat : 1/1
x : tensor([4.6000, 3.6000, 1.0000, 0.2000]), y vs y_hat : 0/0
x : tensor([6.8000, 3.0000, 5.5000, 2.1000]), y vs y_hat : 2/2
x : tensor([6.3000, 3.3000, 4.7000, 1.6000]), y vs y_hat : 1/1
x : tensor([6.2000, 2.2000, 4.5000, 1.5000]), y vs y_hat : 1/1
x : tensor([6.2000, 2.8000, 4.8000, 1.8000]), y vs y_hat : 2/2
x : tensor([6.3000, 2.5000, 5.0000, 1.9000]), y vs y_hat : 2/2
x : tensor([6.7000, 3.1000, 5.6000, 2.4000]), y vs y_hat : 2/2
x : tensor([5.7000, 2.8000, 4.5000, 1.3000]), y vs y_hat : 1/1
x : tensor([5.8000, 2.7000, 3.9000, 1.2000]), y vs y_hat : 1/1
x : tensor([5.3000, 3.7000, 1.5000, 0.2000]), y vs y_hat : 0/0
x : tensor([5.1000, 3.5000, 1.4000, 0.2000]), y vs y_hat : 0/0
x : tensor([6.3000, 2.3000, 4.4000, 1.3000]), y vs y_hat : 1/1
x : tensor([4.7000, 3.2000, 1.6000, 0.2000]), y vs y_hat : 0/0
x : tensor([7.7000, 2.6000, 6.9000, 2.3000]), y vs y_hat : 2/2
x : tensor([4.4000, 3.2000, 1.3000, 0.2000]), y vs y_hat : 0/0
x : tensor([4.9000, 2.4000, 3.3000, 1.0000]), y vs y_hat : 1/1
x : tensor([6.4000, 2.7000, 5.3000, 1.9000]), y vs y_hat : 2/2
```
