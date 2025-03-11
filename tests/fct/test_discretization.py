"""Tests for the discretization module."""

# Standard library imports
import pytest
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

# Internal imports
from aim2dat.fct.discretization import (
    limit_array,
    DiscretizedAxis,
    DiscretizedGrid,
)

cwd = os.path.dirname(__file__) + "/"


@pytest.fixture
def data():
    """Create example data for axis."""
    return np.arange(-5, 6)


@pytest.fixture
def axis_gaussian():
    """Create example gaussian axis."""
    axis = DiscretizedAxis(axis_type="x", max=0.49, min=0, min_step=0.01)
    axis.discretization_method = "gaussian"
    axis.discretize_axis(mu=0.2, sigma=2)
    return axis


@pytest.fixture
def axis_exponential():
    """Create example exponential axis."""
    axis = DiscretizedAxis(axis_type="x", max=1, min=0, min_step=0.01, max_num_steps=10)
    axis.discretization_method = "exponential"
    axis.discretize_axis(mu=0.1)
    return axis


@pytest.fixture
def axis_uniform():
    """Create example uniform axis."""
    axis = DiscretizedAxis(axis_type="x", max=1, min=0, min_step=0.1)
    axis.discretization_method = "uniform"
    axis.discretize_axis()
    return axis


@pytest.fixture
def axis_uniform2():
    """Create example uniform axis."""
    axis = DiscretizedAxis(axis_type="x", max=1.4, min=-0.55, min_step=0.1)
    axis.discretization_method = "uniform"
    axis.discretize_axis()
    return axis


@pytest.fixture
def axis_uniform3():
    """Create example uniform axis."""
    axis = DiscretizedAxis(axis_type="x", max=0.9, min=0.2, min_step=0.1)
    axis.discretization_method = "uniform"
    axis.discretize_axis()
    return axis


@pytest.fixture
def axis_custom():
    """Create custom axis."""
    axis = DiscretizedAxis("x")
    axis.axis = np.array([0.2, 0.5, 0.6, 0.9, 1.3])
    return axis


@pytest.fixture
def grid_uniform(axis_uniform, axis_uniform2):
    """Create example uniform grid."""
    grid = axis_uniform + axis_uniform2.T
    grid.create_grid()
    return grid


def test_limit_array(data):
    """Test function to limit an array."""
    np.testing.assert_array_equal(limit_array(data, -6, 1), np.arange(-6, 2))
    np.testing.assert_array_equal(limit_array(data, -4, 7), np.append(np.arange(-4, 6), 7))
    np.testing.assert_array_equal(
        limit_array(data, -2.5, 3.4), np.concatenate([[-2.5], np.arange(-2, 4), [3.4]])
    )


def test_print(axis_uniform):
    """Test representation."""
    repr_message = (
        f"DiscretizedAxis\n\taxis_type: {axis_uniform.axis_type}\n\t"
        f"max: {axis_uniform.max}\n\tmin: {axis_uniform.min}\n\tmin_step: {axis_uniform.min_step}"
        f"\n\tmax_num_steps: {axis_uniform.max_num_steps}"
        f"\n\tprecision: {axis_uniform.precision}\n\t"
        f"discretization_method: {axis_uniform.discretization_method.__name__}\n\n"
    ) + object.__repr__(axis_uniform)

    assert repr_message == axis_uniform.__str__()


def test_is_empty(axis_uniform):
    """Test is_empty property."""
    assert not axis_uniform.is_empty
    assert DiscretizedAxis("x").is_empty


def test_axis_type_transpose(axis_uniform):
    """Test transpose."""
    assert axis_uniform.shape[0] == 1
    axis_transposed = axis_uniform.T

    assert axis_transposed.axis_type == "y"
    assert (
        axis_transposed.shape[0] == axis_uniform.shape[1]
        and axis_transposed.shape[1] == axis_uniform.shape[0]
    )
    with pytest.raises(ValueError):
        DiscretizedAxis("z")
    with pytest.raises(ValueError):
        DiscretizedAxis("x", shape=(1,))
    with pytest.raises(ValueError):
        axis = DiscretizedAxis("x")
        axis.axis = np.arange(8).reshape(2, 4)


def test_uniform_axis(axis_uniform):
    """Test uniform axis discretization."""
    np.testing.assert_array_equal(
        axis_uniform.axis, np.arange(0, 1.1, 0.1).round(axis_uniform.precision).reshape(1, -1)
    )


def test_gaussian_axis(axis_gaussian):
    """Test gaussian axis."""
    with open(cwd + "reference_data/gaussian_axis.yaml") as file:
        ref = yaml.load(file, Loader=yaml.Loader)
    np.testing.assert_array_equal(axis_gaussian.axis, np.array(ref).reshape(1, -1))


def test_exponential_axis(axis_exponential):
    """Test exponential axis."""
    with open(cwd + "reference_data/exponential_axis.yaml") as file:
        ref = yaml.load(file, Loader=yaml.Loader)
    np.testing.assert_array_equal(axis_exponential.axis, np.array(ref).reshape(1, -1))


def test_discretization_method():
    """Test handling of unsupported methods."""
    with pytest.raises(ValueError):
        DiscretizedAxis("x", discretization_method="test")
    with pytest.raises(ValueError):
        DiscretizedAxis("x", discretization_method=2)
    with pytest.raises(ValueError):
        axis = DiscretizedAxis("x")
        axis.discretize_axis()
    with pytest.raises(ValueError):
        axis = DiscretizedAxis("x", min=0, max=1)
        axis.discretize_axis()

    def _discr(**kwargs):
        return np.ones(3)

    axis = DiscretizedAxis("x", min=0, max=4)
    axis.discretization_method = _discr
    axis.discretize_axis()

    np.testing.assert_array_equal(
        axis.axis, np.concatenate([0, np.ones(3), 4], axis=None).reshape(1, -1)
    )


def test_add(axis_uniform, axis_uniform2, axis_uniform3):
    """Test addition of two axis instances."""
    axis_uniform_init = np.copy(axis_uniform)
    new_axis = axis_uniform + axis_uniform2
    new_axis3 = axis_uniform + axis_uniform3

    assert isinstance(new_axis, DiscretizedAxis)
    np.testing.assert_array_equal(
        new_axis.axis,
        np.concatenate([np.arange(-0.5, 1.4, 0.1), [1.35]])
        .round(new_axis.precision)
        .reshape(1, -1),
    )
    assert new_axis3 is axis_uniform
    assert new_axis3 == axis_uniform_init

    with pytest.raises(ValueError):
        axis_uniform + DiscretizedAxis("x")

    grid = axis_uniform + axis_uniform2.T
    assert isinstance(grid, DiscretizedGrid)
    assert grid.is_empty
    grid.create_grid()
    assert not grid.is_empty

    np.testing.assert_array_equal([g[0] for g in grid], axis_uniform.axis.flatten())
    assert all([all(g[1] == -np.sort(-axis_uniform2.axis.flatten())) for g in grid])


def test_mul(axis_uniform, axis_custom):
    """Test multiplication of two axis instances"""
    with pytest.raises(ValueError):
        axis_uniform * axis_custom
    with pytest.raises(ValueError):
        axis_uniform * DiscretizedAxis("y")

    grid = axis_uniform.T * axis_custom
    grid.create_grid()

    np.testing.assert_array_equal([g[0] for g in grid], axis_custom.axis.flatten())
    assert all(
        [
            all(g[1] == -np.sort(-axis_uniform.axis.flatten() * x).round(8))
            for g, x in zip(grid, [3, 1, 3, 4, 4])
        ]
    )


def test_plot_grid(grid_uniform):
    """Test plotting of grid."""
    fig = grid_uniform.plot_grid()
    ax = fig.axes

    assert isinstance(fig, plt.Figure)
    assert len(ax) == 1
