"""Test radial distribution plots."""

# Standard library imports
import os


# Third party library imports
import pytest

# Internal library imports
from aim2dat.plots import PartialRDFPlot
from aim2dat.io import read_yaml_file


MAIN_PATH = os.path.dirname(__file__) + "/partial_rdf/"


def test_errors():
    """Test errors."""
    prdf_plot = PartialRDFPlot()
    with pytest.raises(ValueError) as error:
        prdf_plot.x_unit = "test"
    assert str(error.value) == "'test' is not a supported unit."


def test_rdf_class(nested_dict_comparison):
    """Test dataset import and _prepare_to_plot function."""
    # Import dataset
    inputs = read_yaml_file(MAIN_PATH + "partial_rdf_import_input.yaml")
    import_ref = read_yaml_file(MAIN_PATH + "partial_rdf_import_ref.yaml")
    plot_ref = read_yaml_file(MAIN_PATH + "partial_rdf_plot_ref.yaml")
    prdf_plot = PartialRDFPlot()
    prdf_plot.element_pairs = ["As-Ga", "Ga-Ga", "TEST"]
    prdf_plot.import_rdf(**inputs)
    for i in range(len(prdf_plot._data["test"])):
        prdf_plot._data["test"][i]["x_values"] = prdf_plot._data["test"][i]["x_values"].tolist()
    nested_dict_comparison(prdf_plot._data, import_ref)

    # Plot dataset
    plot_data_sets, _, _, _, _, _ = prdf_plot._prepare_to_plot(["test"], [0])
    assert len(plot_data_sets) == len(plot_ref)
    for plot_ds1, plot_ds2 in zip(plot_data_sets, plot_ref):
        assert len(plot_ds1) == len(plot_ds2)
        for ds1, ds2 in zip(plot_ds1, plot_ds2):
            nested_dict_comparison(ds1, ds2)
