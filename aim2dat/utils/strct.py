"""Utility functions used for atomic structures."""

# Standard library imports
from typing import List

# Third party library imports
import numpy as np


def _get_cell_from_lattice_p(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> List[List[float]]:
    """
    Get cell matrix from lattice parameters.

    Parameters
    ----------
    a : float
        Length of the first vector.
    b : float
        Length of the second vector.
    c : float
        Length of the third vector.
    alpha : float
        Angle between b and c.
    beta : float
        Angle between a and c.
    gamma : float
        Angle between a and b.

    Returns
    -------
    list
        Nested list of the three cell vectors.
    """
    eps = 1.0e-10
    sin = []
    cos = []
    for angle in [alpha, beta, gamma]:
        if abs(angle - 90.0) < eps:
            cos.append(0.0)
            sin.append(1.0)
        elif abs(angle + 90.0) < eps:
            cos.append(0.0)
            sin.append(-1.0)
        else:
            cos.append(np.cos(angle * np.pi / 180.0))
            sin.append(np.sin(angle * np.pi / 180.0))

    c1 = float(c) * cos[1]
    c2 = float(c) * (cos[0] - cos[1] * cos[2]) / sin[2]
    c3 = float(np.sqrt(float(c) ** 2.0 - c1**2.0 - c2**2.0))

    v1 = float(a) * np.array([1.0, 0.0, 0.0])
    v2 = float(b) * np.array([cos[2], sin[2], 0.0])
    return [v1.tolist(), v2.tolist(), [c1, c2, c3]]
