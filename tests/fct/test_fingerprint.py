"""Tests for the fingerprint module."""

# Standard library imports
import pytest
import matplotlib.pyplot as plt
import yaml
import numpy as np
import os

# Internal imports
from aim2dat.fct.fingerprint import FunctionDiscretizationFingerprint
from aim2dat.fct import DiscretizedAxis

cwd = os.path.dirname(__file__) + "/"


@pytest.fixture
def fingerprint():
    """Create example fingerprint instance."""
    axis = DiscretizedAxis(axis_type="x", max=0.5, min=0.0, min_step=0.05)
    axis.discretization_method = "uniform"
    axis.discretize_axis()

    grid = axis + axis.T
    grid.create_grid()
    func_disc_fp = FunctionDiscretizationFingerprint(grid=grid, precision=6)
    func_disc_fp._add_fingerprint(np.array([1, 2, 3]), "test_fp")
    func_disc_fp._add_fingerprint(np.array([1, 2]), "test_fp_small")

    return func_disc_fp


@pytest.fixture
def data():
    """Create example energy and dos data."""
    with open(cwd + "reference_data/energy_dos.yaml") as file:
        energy, dos = yaml.load(file, Loader=yaml.Loader)

    return np.array(energy), np.array(dos)


def test_add_fingerprint(fingerprint):
    """Test internal handling."""
    with pytest.raises(ValueError):
        fingerprint._add_fingerprint(np.array([1]), "test_fp")


def test_return_fingerprint(fingerprint):
    """Test internal handling."""
    with pytest.raises(ValueError):
        fingerprint._return_fingerprint("test_fp2")

    np.testing.assert_array_equal(fingerprint._return_fingerprint("test_fp"), np.array([1, 2, 3]))


def test_integrate(fingerprint, data):
    """Test integration."""
    energy, dos = data
    integrated_x, integrated_y = fingerprint._integrate(energy, dos)

    with open(cwd + "reference_data/energy_dos_integrated.yaml") as file:
        ref = yaml.load(file, Loader=yaml.Loader)

    np.testing.assert_array_equal(ref[0], integrated_x)
    np.testing.assert_array_equal(ref[1], integrated_y)


def test_calculate_fingerprint(fingerprint, data):
    """Test fingerprint calculation."""
    energy, dos = data
    fp = fingerprint.calculate_fingerprint(energy, dos)

    with open(cwd + "reference_data/fingerprint1.yaml") as file:
        ref = yaml.load(file, Loader=yaml.Loader)

    np.testing.assert_array_equal(ref, fp)


def test_compare_fingerprints(fingerprint, data):
    """Test comparison of fingerprints."""
    energy, dos = data
    _ = fingerprint.calculate_fingerprint(energy, dos, "fp")
    _ = fingerprint.calculate_fingerprint(energy, dos + 1.2, "fp1")
    compared_fp = fingerprint.compare_fingerprints("fp", "fp1")

    with pytest.raises(ValueError):
        fingerprint.compare_fingerprints("test_fp_small", "test_fp")

    assert compared_fp == 0.5


def test_plot_fingerprint(fingerprint, data):
    """Test plotting of fingerprint."""
    energy, dos = data
    fig = fingerprint.plot_fingerprint(energy, dos)
    ax = fig.axes

    assert isinstance(fig, plt.Figure)
    assert len(ax) == 1
