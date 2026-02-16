import numpy as np
import matplotlib.pyplot as plt

def mvn_pdf(grid, mean, cov):
    """
    Multivariate normal PDF evaluated on a grid.
    grid: (..., 2) array of points
    mean: (2,)
    cov : (2,2)
    returns: (...) array
    """
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)

    diff = grid - mean
    expo = -0.5 * np.einsum("...i,ij,...j->...", diff, inv, diff)
    norm = 1.0 / np.sqrt((2.0 * np.pi) ** 2 * det)
    return norm * np.exp(expo)

def mixture_pdf(grid, means, covs, weights=None):
    means = np.asarray(means, dtype=float)
    k = means.shape[0]

    if weights is None:
        weights = np.ones(k, dtype=float) / k
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

    out = np.zeros(grid.shape[:-1], dtype=float)
    for w, m, c in zip(weights, means, covs):
        out += w * mvn_pdf(grid, m, c)
    return out

def quantile_levels(Z, qs=(0.80, 0.88, 0.93, 0.96, 0.985, 0.995), eps=1e-15):
    """
    Choose contour levels from quantiles of the positive values of Z
    for nicely spaced lines.
    """
    Zpos = Z[Z > eps]
    if Zpos.size == 0:
        raise ValueError("Z has no positive values above eps; check your PDF/grid.")
    return np.quantile(Zpos, qs)

def main():
    # ----- Grid -----
    lim = 10.0
    n = 500
    x = np.linspace(-lim, lim, n)
    y = np.linspace(-lim, lim, n)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx, yy], axis=-1)

    # ----- Unit Gaussian prior (blue), centered at origin -----
    prior_mean = np.array([0.0, 0.0])
    prior_cov  = np.eye(2)  # unit Gaussian
    prior = mvn_pdf(grid, prior_mean, prior_cov)

    # ----- Mixture of 3 Gaussians (red), overlapping allowed -----
    # Put the mixture as a whole away from the origin so it is separated from the prior.
    means = np.array([
        [5.8,  2.5],
        [6.8,  3.1],
        [6.2,  1.4],
    ])

    # Covariances chosen to allow overlap (not super tight).
    covs = np.array([
        [[1.2,  0.4],
         [0.4,  0.9]],

        [[0.9, -0.2],
         [-0.2, 1.1]],

        [[1.0,  0.3],
         [0.3,  1.0]],
    ])

    weights = np.array([0.33, 0.34, 0.33])
    mix = mixture_pdf(grid, means, covs, weights)

    # ----- Plot: same axes, blue prior + red mixture -----
    plt.figure(figsize=(7.5, 6.5))

    # Choose contour levels from quantiles so both sets look nice
    prior_levels = quantile_levels(prior)
    mix_levels   = quantile_levels(mix)

    # Blue prior contours
    plt.contour(xx, yy, prior, levels=prior_levels, colors="blue", linewidths=1.6)

    # Red mixture contours
    plt.contour(xx, yy, mix, levels=mix_levels, colors="red", linewidths=1.6)

    # Optional: mark mixture component means
    plt.scatter(means[:, 0], means[:, 1], c="red", s=35, marker="x", linewidths=2)

    # Optional: mark prior mean
    plt.scatter([prior_mean[0]], [prior_mean[1]], c="blue", s=35, marker="o")

    plt.title("VAE intuition: unit Gaussian prior (blue) vs. multimodal aggregated posterior (red)")
    plt.xlabel("$z_1$")
    plt.ylabel("$z_2$")
    plt.gca().set_aspect("equal", "box")
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()