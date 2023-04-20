"""
The purpose of this file is to contain classes for various types of Archimedian and Elliptical Copulae
"""

import numpy as np
import sympy
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

plt.style.use("seaborn-v0_8")
plt.rcParams.update({"font.family": "Times New Roman"})
plt.rcParams.update(
    {"axes.grid": True, "axes.linewidth": 0.5, "axes.edgecolor": "black"}
)


def ClaytonCopula():
    def __init__(self):
        pass

    def cdf(self, vector):
        return self.phi_inverse(np.sum([self.phi(x) for x in vector]))

    def phi(self, t, alpha):
        """Generator Function"""
        return 1 / alpha * (t ** (-alpha) - 1)

    def phi_inverse(self, t, alpha):
        """Inverse of Generator Function"""
        return (alpha * t + 1) ** (-1 / alpha)

    def fit(self, data):
        """Fit a mxn dataset based on neg log likelihood"""

    def pdf(self, vector):
        """provides the marginal distribution"""

        # TODO: only possible with sympy for two variables for now
        u, v, alpha = sympy.symbols("u v alpha")

    def marginal_cdf(self, mask, vector):
        """Provides the marginal CDF or more of the random variables conditional on the others."""

    def plot_2d(self, mask):
        """Plot of copula in 2D conditional on other random variables"""

    def plot_3d(self, mask):
        """3D plot of multivariate copula CDF"""
