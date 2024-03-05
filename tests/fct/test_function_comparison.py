"""Tests for the function comparison module.S"""

# Standard library imports
import pytest
import numpy as np
import yaml
import os

# Internal imports
from aim2dat.fct.function_comparison import FunctionAnalysis
from aim2dat.fct import DiscretizedAxis

cwd = os.path.dirname(__file__) + "/"


@pytest.fixture
def fct(x_data, y_data, y_data2):
    """Create FunctionAnalysis object instance."""
    func_analysis = FunctionAnalysis()
    func_analysis.import_data("1", x_data, y_data)
    func_analysis.import_data("2", x_data, y_data2)

    with open(cwd + "reference_data/energy_dos.yaml") as file:
        energy, dos = yaml.load(file, Loader=yaml.Loader)

    func_analysis.import_data("fp1", np.array(energy), np.array(dos))
    func_analysis.import_data("fp2", np.array(energy), np.array(dos) + 1.2)

    return func_analysis


@pytest.fixture
def grid():
    """Create example fingerprint instance."""
    axis = DiscretizedAxis(axis_type="x", max=0.5, min=0.0, min_step=0.05)
    axis.discretization_method = "uniform"
    axis.discretize_axis()

    grid = axis + axis.T
    grid.create_grid()

    return grid


@pytest.fixture
def x_data():
    """Create example x data."""
    return np.arange(6)


@pytest.fixture
def y_data():
    """Create example y data."""
    return np.array([1, 2, 3, 4, 5, 6])


@pytest.fixture
def y_data2():
    """Create example y data."""
    return np.array([6, 5, 4, 3, 2, 1])


def test_import_data(fct, x_data, y_data):
    """Test import of data."""
    fct.import_data("test_data", x_data, y_data)

    returned_data = fct._return_data("test_data")
    assert all(returned_data["x_values"] == x_data)
    assert all(returned_data["y_values"] == y_data)

    with pytest.raises(ValueError):
        fct._return_data("test_data1")
    with pytest.raises(ValueError):
        fct.import_data("test_data", x_data, y_data)


def test_calculate_correlation(fct, y_data, y_data2):
    """Test calculation of pearson correlation."""
    check_correlation = ((y_data - y_data.mean()) * (y_data2 - y_data2.mean())).sum() / (
        len(y_data) * y_data.std() * y_data2.std()
    )

    assert round(check_correlation, 8) == round(fct.calculate_correlation("1", "2"), 8)


def test_calculate_distance(fct):
    """Test calculation of distances."""
    distance_euclidian = fct.calculate_distance("1", "2", "euclidian")
    assert round(distance_euclidian, 8) == round(np.sqrt(70), 8)

    distance_cosine = fct.calculate_distance("1", "2", "cosine")
    assert round(distance_cosine, 8) == round(1 - 56 / 91, 8)

    distance_total = fct.calculate_distance("1", "2", "total")
    assert round(distance_total, 8) == round(0, 8)

    distance_absolute = fct.calculate_distance("1", "2", "absolute")
    assert round(distance_absolute, 8) == round(18, 8)


def test_calculate_area(fct, x_data, y_data):
    """Test calculation of area."""
    dx = np.diff(x_data)
    fi_fi1 = y_data[:-1] + y_data[1:]
    area = (fi_fi1 / 2 * dx).sum()

    assert area == fct._calculate_area("1")


def test_compare_areas(fct):
    """Test comparison of areas."""
    assert 1 == fct.compare_areas("1", "2")


def test_calculate_discrete_fingerprint(fct, grid):
    """Test calculation of fingerprint."""
    fp = fct.calculate_discrete_fingerprint("fp1", grid)
    with open(cwd + "reference_data/fingerprint1.yaml") as file:
        ref = yaml.load(file, Loader=yaml.Loader)

    np.testing.assert_array_equal(ref, fp)


def test_compare_functions_by_discrete_fingerprint(fct, grid):
    """Test comparison of fingerprints."""
    fp_comp = fct.compare_functions_by_discrete_fingerprint("fp1", "fp2", grid)

    assert fp_comp == 0.5
