# Bayesian Linear Regression

`noisy()` creates synthetic data based on `2. * x + .3 + noise`; where the noise is gaussian (`Normal(0., 0.001)`)

|   x   |   y   |
| ------| ------|
|  0.00 | 0.301 |
|  0.10 | 0.501 |
|  0.20 | 0.701 |
|  0.30 | 0.901 |
|  0.40 | 1.101 |
|  0.50 | 1.301 |
|  0.60 | 1.501 |
|  0.70 | 1.701 |
|  0.80 | 1.901 |
|  0.90 | 2.101 |
|  1.00 | 2.301 |

We define `model` and `guide`. Both these functiont take `x, y` as arguments. In `model`, we sample weight `w` and bias `b` from Normal distributions `Normal(0., 1.)`. We construct a simple linear regression model `y_hat = w * x + b`. We define a normal distribution for target variable `y` centered around `w * x + b`. 

```python
w = pyro.sample('w', pdist.Normal(0., 1.))
b = pyro.sample('b', pdist.Normal(0., 1.))

y_hat = w * x + b
```

We then, define a variance `sigma` for the distribution over `y`.

```python
sigma = pyro.sample('sigma', pdist.Normal(0., 1.))
```

In next step, we iterate over the observed data `(x,y)`. We sample `y` from a normal distribution we built, centered around `w * x + b` with variance `sigma`, (observing) given the ground truth for target variable `y`.

```python
with pyro.iarange('data', len(x)):
  pyro.sample('obs', pdist.Normal(y_hat, sigma), obs=y)
```

Based on what we've seen so far, what do we need to infer? What are the parameters of the model? That's easy. We just need `w, b, sigma`. Note that we need to infer distributions over them.

Let's build a guide function keeping that in mind.

We assume the variables `w, b, sigma` are all normally distributed. We define the mean and scale of these variables as pyro parameters `pyro.param`.

```python
# parameters of (w : weight)
w_loc = pyro.param('w_loc', torch.tensor(0.))
w_scale = pyro.param('w_scale', torch.tensor(1.))
# parameters of (b : bias)
b_loc = pyro.param('b_loc', torch.tensor(0.))
b_scale = pyro.param('b_scale', torch.tensor(1.))
# parameters of (sigma)
sigma_loc = pyro.param('sigma_loc', torch.tensor(1.))
sigma_scale = pyro.param('sigma_scale', torch.tensor(0.05))
```

Then we sample `w, b, sigma` from Normal distributions, as we did in the model, except now, we define the normal distributions using the pyro parameters we defined above. Now the task of inference procedure, is to estimate the values of these parameters `w_loc, w_scale, b_loc, b_scale, sigma_loc, sigma_scale`, such that the loss is minimized. What loss? The ELBO loss which corresponds to how well our model fits on the data. Too general? Well, yeah, I know. We'll get to ELBO loss another time. [Bear with me](http://explosm.net/comics/3902/).

```python
# sample (w, b, sigma)
w = pyro.sample('w', pdist.Normal(w_loc, w_scale))
b = pyro.sample('b', pdist.Normal(b_loc, b_scale))
sigma = pyro.sample('sigma', pdist.Normal(sigma_loc, sigma_scale))
```

Then we combine `w, b, x` together and define our model once again.

```python
y_hat = w * x + b
```

Why do we do this last step? Is there any significance to it?

Apparently not.  I just removed that line and everything works fine. So why do we have that line? So that, the guide looks like the model? What is going on here? Let's get back to this puzzle later.

Moving on. Now we need to infer the `pyro.param`'s. We create an instance of `SVI`. 
SVI does the learning part quietly and provides us a posterior over `w`, `b` and `sigma` (variance).

We get the statistics of the posteriors and summarize them.

The posteriors are nicely centered around the values we chose in `noisy()` function.

|     | estimate | truth |
|-----|----------|-------|
| `w` | 1.999    | 2.0   |
| `b` | 0.297    | 0.3   |


# Multivariate Linear Regression

|  x0   |   x1  |  x2  |    y   |
|-------|-------|------|--------|
| -1.40 | -4.10 | 1.20 | -5.299 |
| -3.50 | -4.90 | -0.40 | -13.799 |
| -1.20 | -1.40 | 4.30 | 9.601 |
| 2.40 | 0.10 | 0.00 | 3.301 |
| -0.30 | 4.50 | -1.70 | 4.301 |
| -1.80 | -4.40 | -0.80 | -12.299 |
| 4.00 | 4.00 | 0.40 | 13.901 |
| -4.50 | 0.00 | -0.20 | -4.399 |
| -1.90 | -4.30 | 2.10 | -3.499 |
| -2.60 | 4.80 | -0.50 | 6.201 |
| 1.60 | 0.70 | -1.80 | -1.699 |
| -2.10 | 0.20 | -4.20 | -13.599 |
| -3.80 | -1.90 | 1.80 | -1.499 |
| -0.10 | 4.10 | 0.70 | 10.901 |
| 4.90 | 2.20 | 4.70 | 24.101 |
| 4.50 | 1.60 | 1.00 | 11.401 |
| -3.20 | 3.00 | -3.10 | -5.799 |

Editing the code a bit, for Multi-variate Linear Regression, we have,

```python
def model(x, y):
  w = pyro.sample('w', pdist.Normal(torch.zeros(3), torch.ones(3)))
```

##

```python
def guide(x, y):
  w_loc = pyro.param('w_loc', torch.zeros(3))
  w_scale = pyro.param('w_scale', torch.ones(3))

  ...

  w = pyro.sample('w', pdist.Normal(w_loc, w_scale))
```
 
|     | estimate | truth |
|-----|----------|-------|
| `w` | [ 0.996, 1.994, 3.006 ] | [ 1.0, 2.0, 3.0 ] |
| `b` | 0.699    | 0.7   |

One problem though. ELBO returns NaN values often. What's going on, in there?
