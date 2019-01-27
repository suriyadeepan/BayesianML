import torch

from collections import OrderedDict

STACK = []
PARAMS = {}


"""
Global States

[1] Effect Handler Stack
[2] Parameter Store

"""


class Messenger(object):

  def __init__(self, fn=None):
    self.fn = fn

  def __enter__(self):
    STACK.append(self)

  def __exit__(self, *args, **kwargs):
    assert STACK[-1] is self
    STACK.pop()

  def process_msg(self, msg):
    pass

  def postprocess_msg(self, msg):
    pass

  def __call__(self, *args, **kwargs):
    with self:
      return self.fn(*args, **kwargs)


class trace(Messenger):

  def __enter__(self):
    super(trace, self).__enter__()
    self.trace = OrderedDict()
    return self.trace

  def postprocess_msg(self, msg):
    assert msg['name'] not in self.trace, "site with the same name exists!"
    self.trace[msg['name']] = msg.copy()

  def get_trace(self, *args, **kwargs):
    self(*args, **kwargs)  # NOTE : What does this do?
    return self.trace


class replay(Messenger):

  def __init__(self, fn, guide_trace):
    self.guide_trace = guide_trace
    super(replay, self).__init__(fn)

  def process_msg(self, msg):
    if msg['name'] in self.guide_trace:
      msg['value'] = self.guide_trace[msg['name']]['value']


class block(Messenger):

  def __init__(self, fn=None, hide_fn=lambda msg: True):
    self.hide_fn = hide_fn
    super(block, self).__init__(fn)

  def process_msg(self, msg):
    if self.hide_fn(msg):
      msg['stop'] = True


class PlateMessenger(Messenger):

  def __init__(self, fn, size, dim):
    assert dim < 0    # select dimension from last
    self.size = size  # number of items
    self.dim = dim    # dimension along which we are iterating
    super(PlateMessenger, self).__init__(fn)

  def process_msg(self, msg):
    if msg['type'] == 'sample':
      # I asumme `batch_shape` looks like this [ 4, 2, 5, 3 ]
      batch_shape = msg['fn'].batch_shape
      if len(batch_shape) < -self.dim or batch_shape[self.dim] != self.size:
        # [1] if -dim > 4, something is wrong
        # [2] does #items match the items in dimension 'dim' ?
        #
        # include dummy dimensions to compensate for the discrepancy
        batch_shape = [1] * (-self.dim - len(batch_shape)) + list(batch_shape)
        # NOTE : we are altering the batch_shape people
        batch_shape[self.dim] = self.size
        # expand "fn" based on batch_shape
        msg['fn'] = msg['fn'].expand(torch.Size(batch_shape))

  def __iter__(self):
    return range(self.size)


def apply_stack(msg):
  # iterate through global stack
  for pointer, handler in enumerate(reversed(STACK)):
    handler.process_msg(msg)
    # When "stop" field is set, any messengers above it, on the stack are
    #  stopped from being applied
    if msg.get('stop'):
      break

  if msg['value'] is None:
    # run function and set value
    msg['value'] = msg['fn'](*msg['args'])

  # post-process after application of 'fn'
  #  use pointer to jump to last location
  for handler in STACK[-pointer - 1:]:
    handler.postprocess_msg(msg)

  return msg


def sample(name, fn, obs=None):
  """
  pyro's sample is an effectful version of torch.distributions.sample(...)
    [.] Construct initial message
    [.] Call apply_stack

  """
  if not STACK:  # there are no active messengers
    # draw a sample and return it
    return fn()

  # there are active messengers
  # Initialize message
  initial_msg = {
      'type' : 'sample',
      'name' : name,
      'fn'   : fn,
      'args' : (),
      'value': obs
      }

  # send message to the messengers in the stack
  msg = apply_stack(initial_msg)
  return msg['value']


def param(name, init_value=None):

  def fn(init_value):
    value = PARAMS.setdefault(name, init_value)
    value.requires_grad_()
    return value

  if not STACK:  # no active messengers
    return fn(init_value)

  # there are active messengers
  # Initialize message
  initial_msg = {
      'type' : 'param',
      'name' : name,
      'value': None,
      'fn'   : fn,
      'args' : (init_value, )
      }

  msg = apply_stack(initial_msg)

  return msg['value']


# plate helper
def plate(name, size, dim):
  return PlateMessenger(fn=None, size=size, dim=dim)


def get_param_store():
  return PARAMS
