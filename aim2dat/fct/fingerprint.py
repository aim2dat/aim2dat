"""Module to compute a fingerprint for spectra."""

from typing import Tuple

# Standard library imports
import numpy as np
import matplotlib.pyplot as plt


class FunctionDiscretizationFingerprint:
    """Fingerprint for functions based on the DOS-Fingerprint presented in
    :doi:`10.1038/s41597-022-01754-z`.
    """

    def __init__(self, grid, **kwargs):
        """Initialize object."""
        self._fingerprints = {}
        self.grid = grid

        self.precision = 6
        for attr, val in kwargs.items():
            self.__setattr__(attr, val)

    def _add_fingerprint(self, fingerprint: np.array, label: str):
        """Add fingerprint to internal fingerprint dictionary.

        Parameters
        ----------
        fingerprint : np.array
            The calculated discretized fingerprint.
        label : str
            Label for the internal memory.
        """
        if label in self._fingerprints:
            raise ValueError(f"Key: {label} already exists.")

        self._fingerprints[label] = fingerprint

    def _return_fingerprint(self, label: str) -> np.array:
        """Return fingerprint that belongs to the label."

        Parameters
        ----------
        label : str
            Internal label of the desired fingerprint.

        Returns
        -------
        type
            np.array : The discretized fingerprint.

        """
        if label not in self._fingerprints:
            raise ValueError(f"Key: {label} does not exist.")

        return self._fingerprints[label]

    def _integrate(self, x_values: np.array, y_values: np.array) -> Tuple[np.array, np.array]:
        """Numerically integrates the function.

        Parameters
        ----------
        x_values : np.array
            x-values of the function.
        y_values : np.array
            y-values of the function.

        Returns
        -------
        Tuple[np.array, np.array]
            The x-values of the grid and the integrated values.

        """
        x_ = np.array([g[0] for g in self.grid])

        y_integrated = []
        for i, x in enumerate(x_[:-1]):
            x_interp = np.linspace(x, x_[i + 1], 5)
            y_interp = np.interp(x_interp, x_values, y_values)
            y_integrated.append(np.trapz(y_interp, x_interp))
        return x_, y_integrated

    def calculate_fingerprint(
        self, x_values: np.array, y_values: np.array, label: str = None
    ) -> np.array:
        """Calculate the fingerprint.

        Parameters
        ----------
        x_values : np.array
            x-values of the function.
        y_values : np.array
            y-values of the function. In case it's a 2D-array, each row will
            be interpreted as a dataset and the fingerprint is calculated
            by concatenating the individual fingerprints.
        label : str
            Label for the internal memory. Defaults to None.

        Returns
        -------
        np.array
            The discretized fingerprint.

        """
        if len(y_values.shape) == 1:
            y_values = np.vstack([y_values])

        fingerprint = np.array([])
        bins = np.column_stack([g[1] for g in self.grid[:-1]])
        for y_vals in y_values:
            _, integrated_y = self._integrate(x_values, y_vals)
            fingerprint_individual = np.where(bins <= integrated_y, 1.0, 0.0).flatten()
            fingerprint = np.concatenate([fingerprint, fingerprint_individual])

        if label:
            self._add_fingerprint(fingerprint, label)
        return fingerprint

    def compare_fingerprints(self, label_1: str, label_2: str) -> float:
        """Compare two fingerprints that are stored in the internal memory.

        Parameters
        ----------
        label_1 : str
            Label of the first fingerprint.
        label_2 : str
            Label of the second fingerprint.

        Returns
        -------
        float
            Similarity measure.

        """
        fingerprint1 = self._return_fingerprint(label_1)
        fingerprint2 = self._return_fingerprint(label_2)

        if fingerprint1.shape != fingerprint2.shape:
            raise ValueError("The fingerprints need to have the same shape.")

        similarity = np.around(
            np.dot(fingerprint1, fingerprint2)
            / (
                np.linalg.norm(fingerprint1) ** 2
                + np.linalg.norm(fingerprint2) ** 2
                - np.dot(fingerprint1, fingerprint2)
            ),
            self.precision,
        )

        return similarity

    def plot_fingerprint(self, x_values: np.array, y_values: np.array) -> plt.Figure:
        """Plot the discretized function and the corresponding grid.

        Parameters
        ----------
        x_values : np.array
            x-values of the function.
        y_values : np.array
            y-values of the function.

        Returns
        -------
        plt.Figure
            Plot of the discretized function.

        """
        x_values, integrated_y = self._integrate(x_values, y_values)

        grid = self.grid

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(
            x_values[:-1],
            integrated_y,
            color="blue",
            width=np.diff(x_values),
            align="edge",
        )
        for i, g in enumerate(grid[:-1]):
            for h in g[1]:
                ax.plot((g[0], grid[i + 1][0]), (h, h), color="red", linewidth=0.5)
            ax.axvline(g[0], color="red", linewidth=0.5)

        ax.set_ylim(0, 1.2 * max(integrated_y))
        plt.close()

        return fig
