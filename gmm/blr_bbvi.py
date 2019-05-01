import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from scipy.stats import norm

N = 100
P = 4
rs = np.random.RandomState(0)
X = rs.randn(N,P)
z_real = rs.randn(P)
y = rs.binomial(1,sigmoid(np.dot(X,z_real)))


def elbo_grad(z_sample, mu, sigma):
  score_mu = (z_sample - mu)/(sigma)
  score_logsigma = (-1/(2*sigma) + np.power((z_sample - mu),2)/(2*np.power(sigma,2))) * sigma
  log_p = np.sum(y * np.log(sigmoid(np.dot(X,z_sample))) + (1-y) * np.log(1-sigmoid(np.dot(X,z_sample))))\
      + np.sum(norm.logpdf(z_sample, np.zeros(P), np.ones(P)))
  log_q = np.sum(norm.logpdf(z_sample, mu, np.sqrt(sigma)))
  return np.concatenate([score_mu,score_logsigma])*(log_p - log_q)


rs = np.random.RandomState(0)
S = 10
n_iter = 10000
mu = rs.randn(P)
G = np.zeros((2*P,2*P))
eta = 1.0
log_sigma = rs.randn(P)
mus = np.zeros((n_iter,P))
delta_lambda = np.zeros(n_iter)

print( "Beginning to optimize")
for t in range(n_iter):
  mus[t] = mu
  if t % 500 == 0:
    print( "Iteration: ", t)
    print( "Mu: ", mu)
    print( "Sigma: ", np.exp(log_sigma))
  sigma = np.exp(log_sigma)
  samples = np.array([rs.normal(mu, np.sqrt(sigma)) for s in range(S)])
  grad_estimate = np.mean(np.array([elbo_grad(z_sample, mu, sigma) for z_sample in samples]),axis=0)
  G = G + np.outer(grad_estimate,grad_estimate)
  mu_new = mu + (eta * 1/np.sqrt(np.diag(G)))[:P] * grad_estimate[:P]
  log_sigma_new = log_sigma + (eta * 1/np.sqrt(np.diag(G)))[P:] * grad_estimate[P:]
  delta_lambda[t] = np.linalg.norm(mu_new-mu)
  if np.linalg.norm(mu_new-mu) < 0.01:
      break
  mu = mu_new
  log_sigma = log_sigma_new
print( "Optimization complete")
print( "Final mu: ", mu)
print( "Final sigma: ", np.exp(log_sigma))
print( "Real values: ", z_real)

x_plot = np.linspace(-2, 2, 100)
plt.plot(x_plot,norm.pdf(x_plot, mu[0], np.sqrt(sigma[0])),"royalblue",linewidth=1,label="Posterior")
plt.axvline(x=z_real[0],c="royalblue",linestyle='dashed',linewidth=1,label="True Value")
plt.plot(x_plot,norm.pdf(x_plot, mu[1], np.sqrt(sigma[1])),c="orangered",linewidth=1)
plt.axvline(x=z_real[1],c="orangered",linestyle='dashed',linewidth=1)
plt.plot(x_plot,norm.pdf(x_plot, mu[2], np.sqrt(sigma[2])),c="forestgreen",linewidth=1)
plt.axvline(x=z_real[2],c="forestgreen",linestyle='dashed',linewidth=1)
plt.plot(x_plot,norm.pdf(x_plot, mu[3], np.sqrt(sigma[3])),c="red",linewidth=1)
plt.axvline(x=z_real[3],c="red",linestyle='dashed',linewidth=1)
plt.xlabel("Value")
plt.ylabel("Density")
leg = plt.legend(loc=1)
leg.legendHandles[0].set_color('black')
leg.legendHandles[1].set_color('black')
plt.savefig('densities.png', format='png',bbox_inches='tight',dpi = 300)

fig = plt.figure(figsize=(14,4), facecolor='white')
ax_1 = fig.add_subplot(121, frameon=True)
ax_2 = fig.add_subplot(122, frameon=True)
ax_1.plot(range(t+1),mus[:(t+1)])
ax_1.set_xlabel("Iteration")
ax_1.set_ylabel("Variational Mean")
ax_2.plot(range(t+1),delta_lambda[:(t+1)])
ax_2.set_xlabel("Iteration")
ax_2.set_ylabel("Change in Variational Mean")
plt.savefig('trace_plots.png', format='png',bbox_inches='tight',dpi=300)
