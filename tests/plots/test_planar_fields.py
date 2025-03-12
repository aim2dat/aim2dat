"""Test planar fields plots."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.io import read_yaml_file, read_critic2_plane
from aim2dat.plots.planar_fields import PlanarFieldPlot


MAIN_PATH = os.path.dirname(__file__) + "/planar_fields_files/"


@pytest.mark.parametrize("system", [("ZIF-8_rhodef")])
def test_import_field(nested_dict_comparison, system):
    """Test field import."""
    plane = read_critic2_plane(MAIN_PATH + system + ".plane")
    pf_plot = PlanarFieldPlot()
    pf_plot.import_field("test", **plane)

    plane = pf_plot._data["test"]
    for key in ["x_values", "y_values", "z_values"]:
        plane[key] = plane[key].tolist()

    ref_plane = read_yaml_file(MAIN_PATH + system + "_import_ref.yaml")
    nested_dict_comparison(plane, ref_plane)


@pytest.mark.parametrize("system,test_case", [("ZIF-8_rhodef", "ang")])
def test_prepare_plot(nested_dict_comparison, system, test_case):
    """Test _prepare_to_plot function for different use cases."""
    plane = read_critic2_plane(MAIN_PATH + system + ".plane")
    ref_dict = read_yaml_file(MAIN_PATH + system + "_" + test_case + "_ref.yaml")
    pf_plot = PlanarFieldPlot()
    for label, value in ref_dict["attributes"].items():
        setattr(pf_plot, label, value)
    pf_plot.import_field("test", **plane)
    (
        data_sets,
        x_ticks,
        y_ticks,
        x_tick_labels,
        y_tick_labels,
        sec_axis,
    ) = pf_plot._prepare_to_plot(**ref_dict["plot_args"])
    for idx1 in range(len(data_sets)):
        for idx2 in range(len(data_sets[idx1])):
            for label in ["x_values", "y_values"]:
                if label in data_sets[idx1][idx2]:
                    data_sets[idx1][idx2][label] = [
                        float(val) for val in data_sets[idx1][idx2][label]
                    ]
            if "z_values" in data_sets[idx1][idx2]:
                data_sets[idx1][idx2]["z_values"] = [
                    [float(val0) for val0 in val] for val in data_sets[idx1][idx2]["z_values"]
                ]
    nested_dict_comparison(
        {
            "data_sets": data_sets,
            "x_ticks": x_ticks,
            "y_ticks": y_ticks,
            "x_tick_labels": x_tick_labels,
            "y_tick_labels": y_tick_labels,
            "sec_axis": sec_axis,
        },
        ref_dict["ref"],
    )
