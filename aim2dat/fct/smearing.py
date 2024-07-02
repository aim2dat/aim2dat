"""Module to implement different smearing methods."""

# Third party library imports
import numpy as np
from scipy.ndimage import filters


def _gaussian(x: np.array, sigma: float) -> np.array:
    """Gaussian function.

    Parameters
    ----------
    x : np.array
        x-values.
    sigma : float
        Scale parameter corresponding to the standard deviation.

    Returns
    -------
    np.array
        Gaussian function values.

    """
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x**2) / (2 * sigma**2))


def _lorentzian(x: np.array, sigma: float) -> np.array:
    """Lorentzian function.

    Parameters
    ----------
    x : np.array
        x-values.
    sigma : float
        Scale parameter corresponding to the full-width at half-maximum (FWHM).

    Returns
    -------
    np.array
        Lorentzian function values.

    """
    return 1 / np.pi * 0.5 * sigma / (x**2 + (0.5 * sigma) ** 2)


AVAILABLE_SMEARING_METHODS = {"gaussian": _gaussian, "lorentzian": _lorentzian}


def apply_smearing(
    y: np.array,
    sampling_width: float = 1.0,
    sigma: float = 0.5,
    radius: int = None,
    method: str = "gaussian",
) -> np.array:
    """Apply smearing to a dataset. Different smearing methods can be specified.

    Parameters
    ----------
    y : np.array
        y-values of dataset.
    sampling_width : float
        Sampling width of the x-axis, i.e. distance between adjacent x-values. Defaults to 1.0.
    sigma : float
        Scale parameter. Defaults to 0.5.
    radius : int
        Radius of the kernel.
    method : str
        String to specify smearing method, see AVAILABLE_SMEARING_METHODS.
        Defaults to 'gaussian'.

    Returns
    -------
    np.array
        Smeared y-values.

    """
    smearing_method = AVAILABLE_SMEARING_METHODS[method]
    if radius is None:
        radius = int(4 * sigma + 0.5)

    weights = smearing_method(np.arange(-radius, radius + sampling_width, sampling_width), sigma)
    weights = weights / weights.sum()

    y_smeared = np.convolve(y, weights, mode="same")

    return y_smeared


# Class implementation of the above functions


class _BaseSmearing:
    _allowed_methods = ["gaussian", "lorentzian"]

    def __init__(self, method="gaussian"):
        self.method = method

    @property
    def method(self):
        """Smearing method."""
        return self._method

    @method.setter
    def method(self, value):
        if value not in self._allowed_methods:
            raise ValueError(f"Smearing method {value} is not supported.")
        self._method = value

    @staticmethod
    def _gaussian(x, sigma):
        """Gaussian function."""
        return _gaussian(x, sigma)

    @staticmethod
    def _lorentzian(x, sigma):
        """Lorentzian function."""
        return _lorentzian(x, sigma)

    def apply_smearing(self, x, y):
        """Apply smearing."""
        smearing_method = getattr(self, self.method)

        weights = smearing_method(x, self.sigma)
        weights = weights / weights.sum()

        y_total = np.sum(y)

        y_smeared = filters.convolve1d(y, weights)

        y_smeared = y_total / y_smeared.sum() * y_smeared

        return y_smeared
