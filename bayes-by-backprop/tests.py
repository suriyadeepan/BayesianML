def test_Gaussian():
  from b3 import Gaussian
  g = Gaussian(0., 0.01)
  sample = g.sample()
  assert sample < 2. and sample > -2.  # enforce range
  assert g.log_prob(0.01) > -1.  # very probable
  assert g.log_prob(9) < -1      # improbable
