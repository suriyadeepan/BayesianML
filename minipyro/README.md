# mini-pyro

The objective is to build a minimal pyro from scratch. 
Not exactly. We still depend on `pyro.distributions`. 
Why though? What is so special about `pyro.distributions` ?

**TODO** : Figure out why we MUST use `pyro.distributions`.
Oh.. To make the code less than 500 lines?

Define model.

```python
def model(data):
  # sample latent variable
  z_loc = pyro.sample('z_loc', pdist.Normal(0., 1.))
  # sample observations under the ContextManager `plate`
```

**What does the plate do? Why plate?**

`plate(name, size, dim)` returns `PlateMessenger` instance

`PlateMessenger(fn=None, size, dim)`

`PlateMessenger` instance is added to the global STACK in `__enter__` method of `Messenger`. It processes messages created at sample sites. What does it do? It just reshapes it. Nothing else.

**What happens inside** `pyro.sample()` **?**

Nothing much. We pass a distribution (function) to `pyro.sample()`, which is returned without further processing if the global STACK is empty. Else we create a message corresponding to `sample()`. Put put everything we know about the sample site, into the message, which is just a dictionary. 

```python
initial_msg = {
      'type' : 'sample', # sample site
      'name' : name,     # name of sample from sample() call
      'fn'   : fn,       # prob distribution from sample() call
      'value': obs,      # set the value to observations if obs is given in sample()
      'args' : ()        # can't think of case where args takes any value
      }
```

Now what? Now, we apply the messengers/effect handlers in the global STACK to the `initial_msg`, using `apply_stack()` and return the message value. Kids, we've reached the end of `pyro.sample()`.

Let's look at what happens inside `apply_stack()`.
What do you expect to happen, man?

We apply the effects in the stack one by one, from the most recent to the least recent.

```python
for pointer, handler in enumerate(reversed(STACK)):
    handler.process_msg(msg)
    # When "stop" field is set, any messengers above it, 
    # on the stack are stopped from being applied
    if msg.get('stop'):
      break

    # :: from `block(Messenger)` class
    # Sites hidden by block will only have the handlers below 
    # block on the PYRO_STACK applied
```

`block` allows inference or other effectful computations to be nested inside models.
It prevents certain sites from being changed by other handlers.

```python
class block(Messenger):

  def __init__(self, fn=None, hide_fn=lambda msg: True):
    self.hide_fn = hide_fn
    super(block, self).__init__(fn)

  def process_msg(self, msg):
    if self.hide_fn(msg):
      msg['stop'] = True
```

`block` implementation is shockingly simple. All it does is, set `stop` attribute of message to `True`, which stops other handlers above it on the STACK to act on the message.

Where were we? Oh yeah, `apply_stack()`. What else is going on in there?
We iterate through the handlers in the stack and apply them on the message.
After applying all the handlers, we check if `msg['value']` is `None`. 
The value will either be set by the `sample()` function if `obs` is given to the `sample()` call.
Or the handlers might have set the value. **Can they do that?**
If we somehow end up with a `None` value in message, we execute the function associated with the message. Remember the probability distribution we passed to the `sample()` call? Yeah, we execute that and set the value to `msg['value']`.

We aren't done yet.

The handlers have a second method that "post-processes" the message, `postprocess_msg()`. Notice that the we apply `postprocess_msg()` in the reverse order of handlers in the STACK. We start where we left off. We gotta honor the `block`.

```python
for handler in STACK[-pointer - 1:]:
    handler.postprocess_msg(msg)
```

We have applied the handlers in the STACK. Return the message. Our message just went through so many changes. It left as a boy and came back as a man.

**Note to self** : Insert meme here

We return the `value` of message and that's the end of `sample()` function.

Next we define a `guide`.
The guide represents a variational distribution.
It has the same signature as the `model`.

```python
def guide(data):
  # define parameters
  #  loc and scale for latent variable `z_loc`
  guide_loc = pyro.param('guide_loc', torch.tensor(0.))
  guide_scale = pyro.param('guide_scale', torch.tensor(0.)).exp()

  # we would like to learn the distribution `loc`
  pyro.sample('z_loc', pdist.Normal(guide_loc, guide_scale))
```

We define parameters we'd like to learn, using `pyro.param(name, init_value=None)`.

```python
def param(name, init_value=None):

  def fn(init_value):
    value = PARAMS.setdefault(name, init_value)
    value.requires_grad_()
    return value
```

`fn` adds parameter `(name, value)` to global PARAMS list and returns `init_value`.
We build a message of type `param` and apply handlers in the stack on it.
And then return the value associated with `msg`.

```python
initial_msg = {
    'type' : 'param',
    'name' : name,
    'value': None,          # why set value to None? because we wait till all effects are applied
    'fn'   : fn,            # fn is a wrapper around the init_value
    'args' : (init_value, ) # fn takes init_value as input
    }
```

We sample from the same site `z_loc` corresponding to latent sample site in the model.

```python
pyro.sample('z_loc', pdist.Normal(guide_loc, guide_scale))
```

We have defined the model and guide. We now, generate synthetic data for the model to fit on. That brings us to our next question.

**How does the learning work?**

We have Stochastic Variational Inference for that. Let's see how that works.

```python
svi = SVI(model, guide, optim=Adam({'lr' : lr}), loss=elbo)
for step in range(num_steps):
  loss = svi.step(data)
```

The learning happens here, `svi.step(data)`. Lets jump in.

```python
def step(self, data):

  with pyro.trace() as param_capture:
    with pyro.block(hide_fn=lambda msg: msg['type'] == 'sample'):
      loss = self.loss(self.model, self.guide, *args, **kwargs)
```

We have loss function is nested inside two nested messengers. 
We know the purpose of `block`; to protect the site values from change.

**What's trace?**

We can intuit that `trace` records stuff while "effectful" functions (`sample(), param()`) are called. Keeps track of `msg` attributes and returns them.

```python
def postprocess_msg(self, msg):
  assert msg['name'] not in self.trace, "site with the same name exists!"
  self.trace[msg['name']] = msg.copy()
```

`postprocess_msg` huh? So `trace` records messages after the other handlers are done with `process_msg`. That makes sense.

Simple enough. Lets move on to the `loss(model, guide, data)` function.
`elbo(model, guide, data)` implements Evidence Lower Bound. 

**What the hell is an ELBO?**

Refer the [theory](#) section.

Lets continue with `elbo(...)`. 

```python
guide_trace = pyro.trace(guide).get_trace(*args, **kwargs)
```

Why run trace again? 
Because we need to `guide` trace and `model` trace, as separate dictionaries.

`pyro.trace(...)` creates a trace instance and `guide` function gets attached to instance, `self.fn`. 
`trace.get_trace(data)` uses `self(data)` as a fancy way to run `self.fn` -> `guide` function in the context of the instace (self/trace).
We don't care about the output of `guide`.
But we do care about the traced values.
`trace.get_trace(data)` returns traced values (messages built by `sample` and `param` functions inside `guide`).

Alright. Heavy stuff guys.. Heavy subject matter.

What next?

Now we trace the model. The caveat is, we use `replay` to reuse values at sample sites visited by the guide, present in `guide_trace`.

```python
model_trace = pyro.trace(pyro.replay(model, guide_trace)).get_trace(*args, **kwargs)
```

We get what `pyro.replay(...)` does. But how does it do it? Say no more.

```python
  def process_msg(self, msg):
    if msg['name'] in self.guide_trace:
      msg['value'] = self.guide_trace[msg['name']]['value']
```

That was easy. All it does is, look for sites visited by the guide and reuse the values evaluated during the guide trace. We get this trace and set it as `model_trace`.
We calculate ELBO loss by accumulating `log_p` (from model trace)and `log_q` (from guide trace).
We make use of `log_prob()` function of probability distributions used in `sample()` functions, given a `value`.

**How do we connect this to ELBO theory?**

That, my friend, is where I'm confused af.

What else? Gradients are calculated based on `loss` value.
The optimizer updates the parameters based on the calculated gradients.
Clear the gradients and return the `loss` value.

You run `svi.step()` 1000 times and we are done!
