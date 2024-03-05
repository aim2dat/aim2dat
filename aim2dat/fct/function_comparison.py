"""Module to analyze spectra."""

# Standard library imports
from scipy import spatial
from scipy import stats
import numpy as np

# Internal library imports
from aim2dat.fct.fingerprint import FunctionDiscretizationFingerprint
from aim2dat.fct.discretization import DiscretizedGrid


class FunctionAnalysis:
    """Class to analyze and compare functions."""

    def __init__(self, **kwargs):
        """Initialize object."""
        super().__init__(**kwargs)

        self._data = {}

    @property
    def allowed_distance_methods(self) -> list:
        """Return allowed distance methods."""
        return ["euclidian", "cosine", "total", "absolute"]

    def import_data(self, data_label: str, x_values: np.array, y_values: np.array):
        """Import data into the internal memory.

        Parameters
        ----------
        data_label : str
            Label for the internal memory.
        x_values : np.array
            x-values of the function.
        y_values : np.array
            y-values of the function.

        """
        if data_label in self._data:
            raise ValueError(f"Key {data_label} already exists.")

        self._data[data_label] = {"x_values": x_values, "y_values": y_values}

    def _return_data(self, label: str) -> dict:
        """Return dataset of the internal memory.

        Parameters
        ----------
        label : str :
            Label of dataset.

        Returns
        -------
        type
            dict : Dictionary containing x- and y-values of the function.

        """
        if label not in self._data:
            raise ValueError(f"Key {label} does not exist.")

        return self._data[label]

    def _calulate_distance(self, method: str) -> "callable":
        """Return distance method based on different metrics.

        Parameters
        ----------
        method : str
            Distance method to use.

        Returns
        -------
        type
            callable : Callable distance method.

        """
        return getattr(self, f"_calculate_{method}_distance")

    def _calculate_euclidian_distance(self, data1: np.array, data2: np.array) -> float:
        """Calculate the euclidian norm of the difference vector of two vectors.

        Parameters
        ----------
        data1 : np.array
            First dataset.
        data2 : np.array
            Second dataset.

        Returns
        -------
        float
            Euclidian norm.

        """
        return np.linalg.norm(data1 - data2)

    def _calculate_cosine_distance(self, data1: np.array, data2: np.array) -> float:
        """Calculate the cosine distance between two vectors.

        Parameters
        ----------
        data1 : np.array
            First dataset.
        data2 : np.array
            Second dataset.

        Returns
        -------
        float
            Cosine distance.

        """
        return spatial.distance.cosine(data1, data2)

    def _calculate_total_distance(self, data1: np.array, data2: np.array) -> float:
        """Calculate the summed total difference between two vectors.

        Parameters
        ----------
        data1 : np.array
            First dataset.
        data2 : np.array
            Second dataset.

        Returns
        -------
        float
            Total difference.

        """
        return np.sum(data1 - data2)

    def _calculate_absolute_distance(self, data1: np.array, data2: np.array) -> float:
        """Calculate the summed absolute difference between two vectors.

        Parameters
        ----------
        data1 : np.array
            First dataset.
        data2 : np.array
            Second dataset.

        Returns
        -------
        float
            Absolute difference.

        """
        return np.sum(np.abs(data1 - data2))

    def calculate_correlation(self, label1: str, label2: str) -> float:
        """Calculate the pearson-correlation between the values of two functions.

        Parameters
        ----------
        label_1 : str
            Label of the first dataset.
        label_2 : str
            Label of the second dataset.

        Returns
        -------
        float
            The correlation.

        """
        data1 = self._return_data(label1).get("y_values")
        data2 = self._return_data(label2).get("y_values")

        return stats.pearsonr(data1, data2)[0]

    # ToDo add cross correlation at some point

    def calculate_distance(self, label1: str, label2: str, method: str = "euclidian") -> float:
        """Calculate the distance between the values of two functions.

        Parameters
        ----------
        label1 : str
            Label of the first dataset.
        label2 : str
            Label of the second dataset.
        method : str
            The metric to calculate the distance. Defaults to "euclidian".

        Returns
        -------
        float
            Abs. error

        """
        data1 = self._return_data(label1).get("y_values")
        data2 = self._return_data(label2).get("y_values")

        distance_method = self._calulate_distance(method)

        return distance_method(data1, data2)

    def _calculate_area(self, label: str) -> float:
        """Calculate the enclosed area of a function.

        Parameters
        ----------
        label : str
            Label of the dataset.

        Returns
        -------
        float
            The enclosed area.

        """
        data = self._return_data(label)
        x_data = data.get("x_values")
        y_data = data.get("y_values")

        area = np.trapz(y_data, x_data)

        return area

    def compare_areas(self, label1: str, label2: str) -> float:
        """Compare the enclosed areas of two functions.

        Parameters
        ----------
        label1 : str
            Label of the first dataset.
        label2 : str
            Label of the second dataset.

        Returns
        -------
        float
            The ratio of the areas. (A1 / A2)

        """
        area1 = self._calculate_area(label1)
        area2 = self._calculate_area(label2)

        return area1 / area2

    def calculate_discrete_fingerprint(self, label: str, grid: DiscretizedGrid) -> np.array:
        """Calculate a discretized fingerprint of a function (:doi:`10.1038/s41597-022-01754-z`).

        Parameters
        ----------
        label : str
            Label of the dataset.
        grid : DiscretizedGrid
            The grid to discretize the function.

        Returns
        -------
        np.array
            Discretized fingerprint.

        """
        data = self._return_data(label)
        x_data = data.get("x_values")
        y_data = data.get("y_values")

        spectra_fp = FunctionDiscretizationFingerprint(grid=grid)
        fp = spectra_fp.calculate_fingerprint(x_data, y_data)

        return fp

    def compare_functions_by_discrete_fingerprint(
        self, label1: str, label2: str, grid: DiscretizedGrid
    ) -> float:
        """
        Compare two functions based on a discretized fingerprint
        (:doi:`10.1038/s41597-022-01754-z`).

        Parameters
        ----------
        label1 : str
            Label of the first dataset.
        label2 : str
            Label of the second dataset.
        grid : DiscretizedGrid
            The grid to discretize the function.

        Returns
        -------
        float
            Similarity.

        """
        data1 = self._return_data(label1)
        x_data1 = data1.get("x_values")
        y_data1 = data1.get("y_values")

        data2 = self._return_data(label2)
        x_data2 = data2.get("x_values")
        y_data2 = data2.get("y_values")

        spectra_fp = FunctionDiscretizationFingerprint(grid=grid)

        _ = spectra_fp.calculate_fingerprint(x_data1, y_data1, label1)
        _ = spectra_fp.calculate_fingerprint(x_data2, y_data2, label2)

        return spectra_fp.compare_fingerprints(label1, label2)
