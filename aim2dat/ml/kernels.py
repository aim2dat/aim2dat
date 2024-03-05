"""Kernels to be used with scikit-learn models."""

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.ml.metrics import ffprint_cosine


def krr_ffprint_cosine(X, Y):
    """
    Cosine kernel for the F-Fingerprint Transformer based on the cosine distance metric
    (:doi:`10.1063/1.3079326`).

    Parameters
    ----------
    X : numpy.array
        F-Fingerprint.
    Y : numpy.array
        F-Fingerprint.

    Returns
    -------
    : float
        Cosine kernel.
    """
    return 1.0 - 2.0 * ffprint_cosine(X, Y)


def krr_ffprint_laplace(X, Y, gamma=None):
    """
    Laplacian kernel for the F-Fingerprint Transformer based on the cosine distance metric
    (:doi:`10.1063/1.3079326`).

    Parameters
    ----------
    X : numpy.array
        F-Fingerprint.
    Y : numpy.array
        F-Fingerprint.

    Returns
    -------
    : float
        Laplacian kernel.
    """
    if gamma is None:
        gamma = 1.0 / X[1]
    return np.exp(-1.0 * gamma * ffprint_cosine(X, Y))
