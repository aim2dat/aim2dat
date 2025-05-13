"""Test io functions for xmgrace files."""

# Standard library imports
import os

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.io import read_xmgrace_band_structure


XMGRACE_PATH = os.path.dirname(__file__) + "/xmgrace/"


def test_read_xmgrace_band_structure():
    """Test read_xmgrace_band_structure function."""
    kpoints = {
        "W": (0.25, 0.75, 0.5),
        "L": (0.5, 0.5, 0.5),
        "Gamma": (0.0, 0.0, 0.0),
        "X": (0.0, 0.5, 0.5),
        "K": (0.375, 0.75, 0.375),
    }
    data = read_xmgrace_band_structure(XMGRACE_PATH + "test_band_structure.agr", kpoints=kpoints)
    assert len(data) == 2
    for i, data0 in enumerate(data):
        ref_data = np.load(XMGRACE_PATH + f"test_band_structure_{i}.npz")
        assert all(len(data0[key]) == len(ref_data[key]) for key in ref_data.keys())

        for i, (idx, label) in enumerate(ref_data["path_labels"]):
            assert data0["path_labels"][i][0] == int(idx)
            assert data0["path_labels"][i][1] == label
        np.testing.assert_allclose(data0["kpoints"], ref_data["kpoints"], atol=1.0e-5)
        np.testing.assert_allclose(data0["bands"], ref_data["bands"], atol=1.0e-5)
