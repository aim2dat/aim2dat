"""Tests for the smearing module."""

# Standard library imports
import numpy as np
import os
import pytest

# Internal imports
from aim2dat.io import read_yaml_file
from aim2dat.fct.smearing import apply_smearing

cwd = os.path.dirname(__file__) + "/"


def test_smearing():
    """Test smearing methods."""
    data, gaussian, lorentzian = read_yaml_file(cwd + "reference_data/smearing.yaml")

    smeared_gaussian = apply_smearing(np.array(data), sigma=2.5, method="gaussian")
    smeared_lorentzian = apply_smearing(np.array(data), sigma=2.5, method="lorentzian")

    np.testing.assert_array_equal(np.round(gaussian, 12), np.round(smeared_gaussian, 12))
    np.testing.assert_array_equal(np.round(lorentzian, 12), np.round(smeared_lorentzian, 12))

    with pytest.raises(KeyError):
        apply_smearing(np.array(data), sigma=2.5, method="test")
