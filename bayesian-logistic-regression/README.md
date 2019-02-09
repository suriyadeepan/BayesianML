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

## MNIST

Classification of MNIST digits works. This required sub-sampling. I set the `batch_size` to `256` and passed `256` datapoints from the training set, at each step of `svi.step()` instead of the whole training set as we did for IRIS.

I've added an evaluation function which runs the model on test set and return the accuracy.

```python
def evaluate(test_x, test_y):
  # get parameters
  w, b = [ get_param(name) for name in ['w_loc', 'b_loc'] ]
  # build model for prediction

  def predict(x):
    return torch.argmax(torch.softmax(torch.matmul(x, w) + b, dim=-1))

  success = 0
  for xi, yi in zip(test_x, test_y):
    prediction = predict(xi.view(1, -1)).item()
    success += int(int(yi) == prediction)

  return 100. * success / len(test_x)
```

##

I'm getting an accuracy of `87.8%`, which is a win in my book.

```
[0/20000] Elbo loss : 6395.336477279663
Evaluation Accuracy :  51.69
[100/20000] Elbo loss : 9695.971691131592
Evaluation Accuracy :  61.56
[200/20000] Elbo loss : 3536.014835357666
Evaluation Accuracy :  70.17
[300/20000] Elbo loss : 1980.0322170257568
Evaluation Accuracy :  77.9
[400/20000] Elbo loss : 3451.1762771606445
Evaluation Accuracy :  81.38
[500/20000] Elbo loss : 2417.754991531372
Evaluation Accuracy :  82.79
[600/20000] Elbo loss : 3140.6585998535156
Evaluation Accuracy :  84.66
[700/20000] Elbo loss : 2804.7498960494995
Evaluation Accuracy :  84.69
[800/20000] Elbo loss : 2545.027126312256
Evaluation Accuracy :  84.78
[900/20000] Elbo loss : 2475.8251628875732
Evaluation Accuracy :  86.45
[1000/20000] Elbo loss : 1587.4794616699219
Evaluation Accuracy :  86.01
[1100/20000] Elbo loss : 2229.5697078704834
Evaluation Accuracy :  84.63
[1200/20000] Elbo loss : 1878.2400512695312
Evaluation Accuracy :  86.79
[1300/20000] Elbo loss : 1774.492763519287
Evaluation Accuracy :  85.3
[1400/20000] Elbo loss : 1764.1221342086792
Evaluation Accuracy :  86.95
[1500/20000] Elbo loss : 1764.941035270691
Evaluation Accuracy :  87.41
[1600/20000] Elbo loss : 2172.166663169861
Evaluation Accuracy :  86.54
[1700/20000] Elbo loss : 1643.249366760254
Evaluation Accuracy :  86.49
[1800/20000] Elbo loss : 2266.14005279541
Evaluation Accuracy :  85.27
[1900/20000] Elbo loss : 1817.266318321228
Evaluation Accuracy :  88.07
[2000/20000] Elbo loss : 2014.1496315002441
Evaluation Accuracy :  86.25
[2100/20000] Elbo loss : 1636.5766687393188
Evaluation Accuracy :  87.47
[2200/20000] Elbo loss : 2472.379104614258
Evaluation Accuracy :  87.69
[2300/20000] Elbo loss : 1383.4860372543335
Evaluation Accuracy :  85.02
[2400/20000] Elbo loss : 1569.5183238983154
Evaluation Accuracy :  86.64
[2500/20000] Elbo loss : 1459.0109167099
Evaluation Accuracy :  85.88
[2600/20000] Elbo loss : 2602.050601005554
Evaluation Accuracy :  87.14
[2700/20000] Elbo loss : 1544.3672218322754
Evaluation Accuracy :  87.32
[2800/20000] Elbo loss : 2307.0844678878784
Evaluation Accuracy :  87.8
```
