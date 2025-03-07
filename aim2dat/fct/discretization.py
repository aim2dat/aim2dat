"""Module to create grids to discretize a function."""

from __future__ import annotations
from typing import Union, Callable
import copy

import numpy as np
import matplotlib.pyplot as plt


def limit_array(
    input_array: np.array, min_value: Union[float, int], max_value: Union[float, int]
) -> np.array:
    """Limit an array to a given minimum and maximum value.

    In case the range is not covered, the corresponding values are added at the first
    or last index.

    Parameters
    ----------
    input_list : np.array
        array that should be limited
    min_value : int
        minimum value
    max_value : int
        max_value

    Returns
    -------
    np.array
        limited array

    """
    if input_array[0] == min_value and input_array[-1] == max_value:
        return input_array
    else:
        limited_array = np.copy(
            input_array[(input_array > min_value) & (input_array < max_value)]
        ).astype(float)
        limited_array = np.insert(limited_array, 0, min_value)
        limited_array = np.append(limited_array, max_value)
        return limited_array


class DiscretizedAxis:
    """Class to create an axis to discretize a 1d function i.e. a 2d plot
    in a grid. Different methods for the discretization are available.
    """

    _available_discretization_methods = ["exponential", "gaussian", "uniform"]

    def __init__(self, axis_type, **kwargs):
        """Initialize object."""
        self._axis = None
        self.axis_type = axis_type
        self._discretization_method = None
        self.precision = 6

        self.min = None
        self.max = None

        self.mu = None
        self.sigma = None
        self.min_step = None
        self.max_num_steps = 5

        for attr, value in kwargs.items():
            self.__setattr__(attr, value)

    def __repr__(self):
        """Represent the object."""
        repr_message = (
            f"DiscretizedAxis\n\taxis_type: {self.axis_type}\n\t"
            f"max: {self.max}\n\tmin: {self.min}\n\tmin_step: {self.min_step}"
            f"\n\tmax_num_steps: {self.max_num_steps}\n\tprecision: {self.precision}\n\t"
            f"discretization_method: {self.discretization_method.__name__}\n\n"
        )
        return repr_message + object.__repr__(self)

    def __add__(self, other: DiscretizedAxis) -> Union[DiscretizedAxis, DiscretizedGrid]:
        """Addition for `DiscretizedAxis` objects.
        If both objects have the same `axis_type` attribute, the addition two objects
        will be merged. The range of the `other` object that is not present in the current
        object will be added. If the current object already covers the range, nothing will
        be changed and the first object will be returned.

        In case the two objects have a different `axis_type` attribute, the object of
        `axis_type` `y` will be used as the y-axis and a `DiscretizedGrid` will be returned.

        Parameters
        ----------
        other : DiscretizedAxis
            Axis that should be added to the first argument.
        other : DiscretizedAxis :


        Returns
        -------
        DiscretizedAxis, DiscretizedGrid
            depending on the `axis_type` attributes, an axis or grid is returned.

        """
        if self.is_empty or other.is_empty:
            raise ValueError("One of the axis_objects is empty.")

        if self.axis_type == other.axis_type:
            upper_axis_to_add = np.array([])
            lower_axis_to_add = np.array([])

            if self.max < other.max:
                upper_axis_to_add = other.axis[other.axis >= self.max]
                upper_axis_to_add = upper_axis_to_add - (upper_axis_to_add[0] - self.max)
                upper_axis_to_add = upper_axis_to_add[1:]

            if other.min < self.min:
                lower_axis_to_add = other.axis[other.axis <= self.min]
                lower_axis_to_add = lower_axis_to_add + (self.min - lower_axis_to_add[-1])
                lower_axis_to_add = lower_axis_to_add[:-1]

            if any([len(upper_axis_to_add) != 0, len(lower_axis_to_add) != 0]):
                new_axis = DiscretizedAxis(axis_type=self.axis_type)
                new_axis.axis = np.concatenate(
                    [lower_axis_to_add, self.axis, upper_axis_to_add], axis=None
                )

                return new_axis

            return self
        else:
            axes = {axis.axis_type: axis for axis in [self, other]}
            return DiscretizedGrid(discretized_x=axes["x"], discretized_y=axes["y"])

    def __mul__(self, other: DiscretizedAxis) -> DiscretizedGrid:
        """Multiplication of two `DiscretizedAxis` objects results in a `DiscretizedGrid` object.
        In contrtrrast to the addition of two `DiscretizedAxis` objects with different
        `axis_type` attributes, the multiplication weights the y intervals based on
        the x intervals.

        Parameters
        ----------
        other : DiscretizedAxis
            Axis that should be combined with the first argument to form a grid.

        Returns
        -------
            DiscretizedGrid
                A discretized grid.

        """
        if self.axis_type == other.axis_type or (self.is_empty or other.is_empty):
            raise ValueError(
                "`axis_type` needs to be different for multiplication of two objects and both"
                "objects have to be non empty."
            )

        # ToDo revisit the weights
        # At the moment the weights are calculated based on the relative step size
        # compared to the smallest step size in the current axis.
        axes = {axis.axis_type: axis for axis in [self, other]}
        step_sizes = np.abs((np.diff(axes["x"].axis) / axes["x"].min_step)).flatten()
        weights = step_sizes / step_sizes.min()
        weights = np.concatenate([weights, weights[-1]], axis=None)
        return DiscretizedGrid(discretized_x=axes["x"], discretized_y=axes["y"], y_weights=weights)

    @property
    def is_empty(self) -> bool:
        """Check whether the axis is empty.

        Returns
        -------
        bool
            Whether the axis is empty.

        """
        return self.axis is None or len(self.axis) == 0

    @property
    def axis(self) -> np.array:
        """Axis array. Contains the discrete values.

        Returns
        -------
        np.array
            The discretized range.

        """
        return self._axis

    @axis.setter
    def axis(self, value: np.array) -> None:
        if value.squeeze().ndim != 1:
            raise ValueError("axis array has to be one dimensional.")
        self._axis = value.reshape(self.shape).round(self.precision)
        if self.max is None:
            self.max = max(self.axis.flatten())
        if self.min is None:
            self.min = min(self.axis.flatten())
        if self.min_step is None:
            self.min_step = np.diff(self.axis.flatten()).min()

    def discretize_axis(self, **kwargs) -> "DiscretizedAxis":
        """Perform the discretization of the specified range.

        Returns
        -------
        DiscretizedAxis

        """
        if self.max is None or self.min is None:
            raise ValueError("Before discretizing an axis, `min` and `max` have to be specified.")
        discretized_axis = self.discretization_method(**kwargs)
        discretized_axis = discretized_axis.round(self.precision)
        self.axis = limit_array(discretized_axis, self.min, self.max)

        return self

    @property
    def axis_type(self) -> str:
        """Specify whether this axis should be used as `x` or `y` axis in a grid.

        Returns
        -------
        str
            indicating whether this axis should be used as `x` or `y`
            axis in case of a merge into a grid.

        Raises
        ------
        ValueError
            The `axis_type` attribute needs to be specified

        """
        return self._axis_type

    @axis_type.setter
    def axis_type(self, value: str):
        if value not in ["x", "y"]:
            raise ValueError("`axis_type` accepts only `x` or `y` as value.")
        self.shape = (1, -1) if value == "x" else (-1, 1)
        self._axis_type = value

    @property
    def shape(self) -> tuple:
        """Tuple specifying the dimensions of the axis (like numpy).

        Returns
        -------
        tuple
            The shape of the axis.

        """
        return self._shape

    @shape.setter
    def shape(self, value: tuple):
        if len(value) != 2:
            raise ValueError("`shape` needs to consist of two numbers.")
        self._shape = value

    def transpose(self) -> "DiscretizedAxis":
        """Change the `axis_type`: `x --> y` or `y --> x`.

        Returns
        -------
        DiscretizedAxis
            Transposed axis.

        """
        new_axis = copy.deepcopy(self)
        new_axis.axis_type = "x" if self.axis_type == "y" else "y"
        new_axis.shape = (self.shape[1], self.shape[0])
        new_axis.axis = self.axis.reshape(new_axis.shape)
        return new_axis

    @property
    def T(self) -> "DiscretizedAxis":
        """Change the `axis_type`: `x --> y` or `y --> x`.

        Returns
        -------
        DiscretizedAxis
            Transposed axis.

        """
        return self.transpose()

    @property
    def discretization_method(self) -> Callable:
        """Discretize the specified range.
        Can be chosen via a string, accepting the methods specified in
        `_available_discretization_methods` or by passing a callable
        function.

        Returns
        -------
        Callable
            Method to discretize the axis.

        """
        if self._discretization_method is None:
            raise ValueError("`discretization_method` has to be specifief before discretization.")

        return self._discretization_method

    @discretization_method.setter
    def discretization_method(self, value: Union[str, Callable]) -> None:
        if isinstance(value, str):
            if value not in self._available_discretization_methods:
                raise ValueError(f"Discretization method `{value}` is not implemented.")

            self._discretization_method = getattr(self, "_" + value + "_discretization")
        elif callable(value):
            self._discretization_method = value
        else:
            raise ValueError(
                f"Discretization method `{value}` has to be a string or a callable function."
            )

    def _exponential_discretization(self, mu: Union[int, float]) -> np.array:
        """Discretize the range based on exponentially increasing/decreasing step sizes.

        Parameters
        ----------
        mu : int, float
            Mean value for the exponential function

        Returns
        -------
        np.array
            Discretized axis values.

        """

        def step(x, N):
            return min((1 + (np.exp(x - mu)) // 1), N)

        grid = [self.min]
        while grid[-1] < self.max:
            grid.append(
                round(
                    grid[-1] + step(grid[-1], self.max_num_steps) * self.min_step, self.precision
                )
            )

        return np.array(grid)

    # Try to vectorize using numpy
    def _gaussian_discretization(
        self, mu: Union[int, float], sigma: Union[int, float], **kwargs
    ) -> np.array:
        """Discretize the range based on the gaussian distribution.

        Parameters
        ----------
        mu : int, float
            Mean of the gaussian (denser grid in this region).
        sigma : int, float
            Standard deviation of the gaussian.

        Returns
        -------
        np.array
            Discretized axis values.

        """

        def step(x, N):
            return (1 + (1 - np.exp(-((x - mu) ** 2) / sigma**2)) * N) // 1

        grid = [mu]

        while grid[-1] < self.max:
            grid.append(
                round(
                    grid[-1] + step(grid[-1], self.max_num_steps) * self.min_step, self.precision
                )
            )

        while self.min < grid[0]:
            grid.insert(
                0,
                round(grid[0] - step(grid[0], self.max_num_steps) * self.min_step, self.precision),
            )

        return np.array(grid)

    def _uniform_discretization(self) -> np.array:
        """Discretize the range into uniformly distributed intervals.

        Returns
        -------
        np.array
            Discretized axis values.

        """
        grid = np.arange(self.min, self.max + self.min_step, self.min_step)

        return grid


class DiscretizedGrid:
    """Class to create a grid to discretize a 1d function i.e. a 2d plot.

    Use the `plot_grid` method to visualize the created grid.
    """

    def __init__(self, **kwargs):
        """Initialize object."""
        self._grid = None

        self.discretized_x = None
        self.discretized_y = None
        self.y_weights = None

        for attr, value in kwargs.items():
            self.__setattr__(attr, value)

    # Probably the merge of Grids is not necessary since every configuration
    # can be created based on the initial Axis objects.

    # # ToDo add method to add two grids and combine multiple grid types
    # # + should merge the grids in x direction
    # def __add__(self, other):
    #     return NotImplemented

    # # / operator should merge grids in y direction
    # def __truediv__(self, other):
    #     return NotImplemented

    def __iter__(self):
        """Iterate over the internal grid."""
        for col in self.grid:
            yield col

    def __getitem__(self, index):
        """Access grid by index. The index refers to the x-axis."""
        return self.grid[index]

    @property
    def is_empty(self) -> bool:
        """Check whether the axis is empty.

        Returns
        -------
        bool
            Whether the axis is empty.

        """
        return self.grid is None or len(self.grid) == 0 or any([len(g[1]) == 0 for g in self.grid])

    @property
    def grid(self) -> list:
        """Return the internal grid as a list of lists.

        Returns
        -------
        type
            list: List of lists representing the x-values and discretized y-values.

        """
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value

    def create_grid(self) -> "DiscretizedGrid":
        """Create the internal grid which is based on a list of lists.
        Each list contains the energy-value (x) as the first argument and the
        DOS-values (y) as a list in the second argument.

        Returns
        -------
        DiscretizedGrid
            Discretized grid.

        """
        x = self.discretized_x.axis
        # Sort y-values in descending order
        y = -np.sort(-self.discretized_y.axis, axis=0)

        weights = self.y_weights if self.y_weights is not None else np.ones(x.size)

        y_matrix = (y.flatten() * weights.reshape(-1, 1)).round(8)

        grid = [[x_i, y_i.tolist()] for x_i, y_i in zip(x.flatten(), y_matrix)]

        self.grid = grid

        return self

    def plot_grid(self):
        """Plot the grid."""
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, g in enumerate(self[:-1]):
            for h in g[1]:
                ax.plot((g[0], self[i + 1][0]), (h, h), color="grey", linewidth=0.5)
            ax.axvline(g[0], color="grey", linewidth=0.5)

        ax.set_ylim(self.discretized_y.min, self.discretized_y.max)
        ax.set_xlim(self.discretized_x.min, self.discretized_x.max)
        plt.close()

        return fig
