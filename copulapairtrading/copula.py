"""
The purpose of this file is to contain classes for various types of Archimedian and Elliptical Copulae
"""

from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy
from scipy.optimize import minimize_scalar
from statsmodels.distributions.empirical_distribution import ECDF
from pandas import DataFrame
from typing import Union

plt.style.use("seaborn-v0_8")
plt.rcParams.update({"font.family": "Times New Roman"})
plt.rcParams.update(
    {"axes.grid": True, "axes.linewidth": 0.5, "axes.edgecolor": "black"}
)


class ClaytonCopula:
    def __init__(self, ndims: int = None, alpha: float = None, isOrdinal: bool = False):
        """
        This class implements the Clayton Copula

        Parameters
        ----------
        ndims : number of dimensions in our dataset to model
        alpha : parameter for our copula
        isOrdinal : is our data already converted into ranked data/cdf
        """
        self.ndims = ndims
        self.alpha = alpha
        self.isOrdinal = isOrdinal

    def convert_to_ecdf(self, vector: DataFrame):
        """
        This function converts our numeric RVs into their respective empirical cdfs

        Parameters
        ----------
        vector : dataframe

        Returns
        -------
        dataframe of ranked vectors
        """

        if not self.ndims:
            self.ndims = vector.shape[1]

        ecdf = np.empty(shape=(vector.shape[0], vector.shape[1]))

        for idx in range(self.ndims):
            ecdf[:, idx] = ECDF(vector.iloc[:, idx])(vector.iloc[:, idx])

        self.isOrdinal = True

        ecdf = np.where(ecdf == 1, 0.9999, ecdf)
        ecdf = np.where(ecdf == 0, 0.0001, ecdf)

        ecdf = pd.DataFrame(ecdf, index=vector.index)
        ecdf.columns = vector.columns

        return ecdf

    def cdf(self, vector: DataFrame, alpha: float):
        """
        This function gets the cdf value correspoding to a set of row vectors and alpha

        Parameters
        ----------
        vector : dataset
        alpha : copula parameter

        Returns
        -------
        CDF value of copula  in (0, 1)
        """
        return self.phi_inverse(
            np.sum([self.phi(t=x, alpha=alpha) for x in vector]), alpha=alpha
        )

    def phi(self, t: Union[DataFrame, float], alpha: float):
        """
        Generator Function

        Parameters
        ----------
        t : input parameter
        alpha : copula paramter

        Returns
        -------

        """
        return 1 / alpha * (t ** (-alpha) - 1)

    def phi_inverse(self, t: Union[DataFrame, float], alpha: float):
        """Inverse of Generator Function

        Parameters
        ----------
        t : input parameter
        alpha : copula paramter

        Returns
        -------

        """
        return (alpha * t + 1) ** (-1 / alpha)

    def pdf(self, vector: DataFrame, alpha: float):
        """
        This function provides the pdf at a given row vector and alpha level

        Parameters
        ----------
        vector : dataset
        alpha : copula parameter

        Returns
        -------
        pdf value
        """

        if not self.isOrdinal:
            vector = self.convert_to_ecdf(vector)

        vector_symbols = [
            (sympy.symbols(f"vector_{idx}"), vector[i]) for idx, i in enumerate(vector)
        ]

        def fprime(vector_symbols):
            return sympy.diff(self.cdf(vector_symbols, alpha), *vector_symbols)

        pdf_func = fprime([x[0] for x in vector_symbols])

        return sympy.lambdify([x[0] for x in vector_symbols], pdf_func)(
            *[x[1] for x in vector_symbols]
        )

    def marginal_cdf(self, mask, vector, alpha):
        """Provides the marginal CDF or more of the random variables conditional on the others."""

        if not self.isOrdinal:
            vector = self.convert_to_ecdf(vector)

        vector_symbols = [
            (sympy.symbols(f"vector_{idx}"), vector[i]) for idx, i in enumerate(vector)
        ]

        # @cache
        def fprime(vector_symbols):
            return sympy.diff(
                self.cdf(vector_symbols, alpha), *list(compress(vector_symbols, mask))
            )

        marginal_cdf_func = fprime([x[0] for x in vector_symbols])

        return sympy.lambdify([x[0] for x in vector_symbols], marginal_cdf_func)(
            *[x[1] for x in vector_symbols]
        )

    # @cache
    def fit(self, vector):
        """Fit a mxn dataset based on neg log likelihood to get optimal value of alpha"""

        if not self.isOrdinal:
            vector = self.convert_to_ecdf(vector)

        def neg_log_likelihood(alpha, vector):
            return -np.sum(np.log(self.pdf(vector, alpha)))

        result = minimize_scalar(
            neg_log_likelihood,
            args=(vector),
            bounds=(-1, 2**32),
            method="bounded",
            options={"disp": True},
        )

        self.alpha = result["x"]

        return result["x"]

    def plot_2d(self, mask):
        """Plot of copula in 2D conditional on other random variables"""

    def plot_3d(self, mask):
        """3D plot of multivariate copula CDF"""
