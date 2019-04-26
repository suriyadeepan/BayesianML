import torch
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')


def logsumexp(x, dim, keepdim=False):
    max, _ = torch.max(x, dim=dim, keepdim=keepdim)
    out = max + (x - max).exp().sum(dim=dim, keepdim=keepdim).log()
    return out


def get_density(mu, logvar, pi, likelihood_f, N=50, X_range=(0, 5), Y_range=(0, 5)):
    """ Get the mesh to compute the density on. """
    X = np.linspace(*X_range, N)
    Y = np.linspace(*Y_range, N)
    X, Y = np.meshgrid(X, Y)
    # get the design matrix
    points = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    points = torch.from_numpy(points).float()
    # compute the densities under each mixture
    P = likelihood_f(points, mu, logvar, log=False)
    # get likelihood as an argument
    # sum the densities to get mixture density
    Z = torch.sum(P, dim=0).data.numpy().reshape([N, N])
    return X, Y, Z


def plot_density(X, Y, Z, i=0, show=False):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1,
        antialiased=True, cmap=cm.inferno)
    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.inferno)
    # adjust the limits, ticks and view angle
    ax.set_zlim(-0.15, 0.2)
    ax.set_zticks(np.linspace(0, 0.2, 5))
    ax.view_init(27, -21)
    if not show:
      plt.savefig(f'plots/fig_{i}.png', dpi=400, bbox_inches='tight')
    else:
      plt.show()
