"""Test spectrum plots."""

# Standard library imports
import os

# Third party library imports
import pytest
import numpy as np

# Internal library imports
from aim2dat.plots import SpectrumPlot
from aim2dat.io import read_yaml_file


MAIN_PATH = os.path.dirname(__file__) + "/spectrum/"


def _transform_numpy_to_list(datasets):
    for dataset in datasets:
        for data in dataset:
            if isinstance(data["x_values"], np.ndarray):
                data["x_values"] = data["x_values"].astype(float).tolist()
            else:
                data["x_values"] = [float(x) for x in data["x_values"]]
            if isinstance(data["y_values"], np.ndarray):
                data["y_values"] = data["y_values"].astype(float).tolist()
            else:
                data["y_values"] = [float(y) for y in data["y_values"]]

    return datasets


@pytest.fixture
def spectrum_data():
    """Create example spectrum data."""
    x = np.linspace(0, 10, 1000)

    y = (
        3 * np.exp(-((x - 1) ** 2) / 0.1**2)
        + 1.5 * np.exp(-((x - 5) ** 2) / 2**2)
        + 2 * np.exp(-((x - 7) ** 2) / 0.5**2)
        + 1.5 * np.exp(-((x - 3) ** 2) / 5**2)
        + 0.2 * np.sin(5 * np.pi * x)
    )
    return x, y


def test_import_spectrum(spectrum_data):
    """Test import of spectrum."""
    spectrum_plot = SpectrumPlot()
    spectrum_plot.import_spectrum("test", *spectrum_data, "eV")

    with pytest.raises(ValueError):
        spectrum_plot.import_spectrum("test_fail", *spectrum_data, "TEST_UNIT")

    np.testing.assert_array_equal(spectrum_plot._data["test"]["x_values"], spectrum_data[0])
    np.testing.assert_array_equal(spectrum_plot._data["test"]["y_values"], spectrum_data[1])
    assert spectrum_plot._data["test"]["unit_x"] == "eV"


def test_prepare_to_plot(spectrum_data, nested_dict_comparison):
    """Test _prepare_to_plot function for different use cases."""
    spectrum_plot = SpectrumPlot(y_label="Intensity [arb. units]", _auto_set_y_label=False)
    spectrum_plot.import_spectrum("test", *spectrum_data, "eV")
    spectrum_plot.import_spectrum("test_unit", *spectrum_data, "J")

    ref_init = read_yaml_file(MAIN_PATH + "spectrum_init.yaml")
    data_sets_init, *unused = spectrum_plot._prepare_to_plot(["test"], [0])
    nested_dict_comparison(
        {"data_sets": _transform_numpy_to_list(data_sets_init)}, ref_init["ref"]
    )

    ref_unit_conv = read_yaml_file(MAIN_PATH + "spectrum_unit_conv.yaml")
    data_sets_unit_conv, *unused = spectrum_plot._prepare_to_plot(["test_unit"], [0])
    nested_dict_comparison(
        {"data_sets": _transform_numpy_to_list(data_sets_unit_conv)}, ref_unit_conv["ref"]
    )

    ref_peaks = read_yaml_file(MAIN_PATH + "spectrum_peaks.yaml")
    spectrum_plot.detect_peaks = True
    data_sets_peaks, *unused = spectrum_plot._prepare_to_plot(["test"], [0])
    spectrum_plot.detect_peaks = False
    nested_dict_comparison(
        {"data_sets": _transform_numpy_to_list(data_sets_peaks), "peaks": spectrum_plot.peaks},
        ref_peaks["ref"],
    )

    ref_smooth = read_yaml_file(MAIN_PATH + "spectrum_smooth.yaml")
    spectrum_plot.smooth_spectra = True
    spectrum_plot.smearing_method = "gaussian"
    spectrum_plot.smearing_sigma = 10
    spectrum_plot.smearing_delta = None
    data_sets_smooth, *unused = spectrum_plot._prepare_to_plot(["test"], [0])
    nested_dict_comparison(
        {"data_sets": _transform_numpy_to_list(data_sets_smooth)}, ref_smooth["ref"]
    )

    ref_smooth_orig = read_yaml_file(MAIN_PATH + "spectrum_smooth_orig.yaml")
    spectrum_plot.plot_original_spectra = True
    data_sets_smooth_orig, *unused = spectrum_plot._prepare_to_plot(["test"], [0])
    nested_dict_comparison(
        {"data_sets": _transform_numpy_to_list(data_sets_smooth_orig)}, ref_smooth_orig["ref"]
    )
