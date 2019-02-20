def test_Gaussian():
  from b3 import Gaussian
  g = Gaussian(0., 0.01)
  sample = g.sample()
  assert sample < 2. and sample > -2.  # enforce range
  assert g.log_prob(0.01) > -1.  # very probable
  assert g.log_prob(9) < -1      # improbable


def test_evaluate_ensemble():
  from b3 import evaluate_ensemble
  from b3 import mnist
  from b3 import BayesianNeuralNet
  trainset, testset = mnist()
  bnn = BayesianNeuralNet(28 * 28, 400, 10)
  av_output = evaluate_ensemble(bnn, testset)
  # assert av_output.size() == torch.Size(1, 10)
