"""Test spectrum plots."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.plots.partial_charges import PartialChargesPlot
from aim2dat.io import read_yaml_file


MAIN_PATH = os.path.dirname(__file__) + "/partial_charges/"


def test_pc_errors():
    """Test errors of PartialChargesPlot"""
    pc_plot = PartialChargesPlot()

    with pytest.raises(ValueError) as error:
        pc_plot.pc_plot_type = "Wurst"
    assert str(error.value) == "`pc_plot_type` must contain 'scatter' or 'bar'."

    with pytest.raises(TypeError) as error:
        pc_plot.pc_plot_type = 42
    assert str(error.value) == "`pc_plot_type` needs to be of type str."


@pytest.mark.parametrize(
    "system",
    [
        "ZIF_8_Cl_struc",
        "ZIF_8_Cl_comp",
        "ZIF_8_Cl_imi",
        "ZIF_8_Br_struc",
        "ZIF_8_Br_comp",
        "ZIF_8_Br_imi",
    ],
)
def test_import_partial_charges(nested_dict_comparison, system):
    """Test import of charges."""
    inputs = read_yaml_file(MAIN_PATH + "partial_charges_import_input.yaml")
    ref = read_yaml_file(MAIN_PATH + "partial_charges_import_ref.yaml")
    ref_wo_ckd = read_yaml_file(MAIN_PATH + "partial_charges_import_ref_wo_ckd.yaml")
    custom_kind_dict_error = read_yaml_file(MAIN_PATH + "index_error.yaml")
    pc_plot = PartialChargesPlot()
    pc_plot.import_partial_charges(**inputs[system])
    pc_data = pc_plot._data[system]
    with pytest.raises(ValueError):
        inputs[system].pop("custom_kind_dict")
        pc_plot.import_partial_charges(
            **inputs[system],
            custom_kind_dict=custom_kind_dict_error,
        )
    nested_dict_comparison(pc_data, ref[system])
    pc_plot_wo_ckd = PartialChargesPlot()
    pc_plot_wo_ckd.import_partial_charges(**inputs[system])
    pc_data_wo_ckd = pc_plot_wo_ckd._data[system]
    nested_dict_comparison(pc_data_wo_ckd, ref_wo_ckd[system])


@pytest.mark.parametrize(
    "system",
    [
        "MOF_5",
        "Ba_MOF_5_Cl",
    ],
)
def test_import_partial_charges_from_aiida_list(nested_dict_comparison, aiida_create_list, system):
    """Test import of charges from aiida list."""
    inputs = read_yaml_file(MAIN_PATH + "aiida_input.yaml")
    ref = read_yaml_file(MAIN_PATH + "aiida_ref.yaml")
    inputs[system]["pcdata"] = aiida_create_list(inputs[system]["pcdata"])
    pc_plot = PartialChargesPlot()
    pc_plot.import_from_aiida_list(**inputs[system])
    pc_data = pc_plot._data[system]
    nested_dict_comparison(pc_data, ref[system])


@pytest.fixture
def systems():
    """Systems for the tests."""
    systems = [
        "ZIF_8_Cl_struc",
        "ZIF_8_Cl_comp",
        "ZIF_8_Cl_imi",
        "ZIF_8_Br_struc",
        "ZIF_8_Br_comp",
        "ZIF_8_Br_imi",
    ]
    return systems


def test_prepare_to_plot(nested_dict_comparison, systems):
    """Test _prepare_to_plot function of BandStructureDOSPlot class."""
    data_sets = ["data_sets", "x_ticks", "y_ticks", "x_tick_labels", "y_tick_labels", "sec_axis"]
    inputs = read_yaml_file(MAIN_PATH + "partial_charges_import_input.yaml")
    ref = read_yaml_file(MAIN_PATH + "plot_import_ref.yaml")
    plot_data = PartialChargesPlot()
    for system in systems:
        plot_data.import_partial_charges(**inputs[system])
    raw = dict(
        zip(data_sets, plot_data._prepare_to_plot(data_labels=systems, subplot_assignment=[0]))
    )
    with_subplot_assignment = dict(
        zip(
            data_sets,
            plot_data._prepare_to_plot(data_labels=systems, subplot_assignment=[0, 0, 1, 2]),
        )
    )
    plot_data_wo_ckd = PartialChargesPlot()
    for system in systems:
        inputs[system].pop("custom_kind_dict")
        plot_data_wo_ckd.import_partial_charges(**inputs[system])
    wo_ckd = dict(
        zip(
            data_sets,
            plot_data_wo_ckd._prepare_to_plot(data_labels=systems, subplot_assignment=[0]),
        )
    )
    plot_data_wo_ckd.pc_plot_order = ["Zn", "Cl", "H"]
    with_pc_plot_assignment = dict(
        zip(
            data_sets,
            plot_data_wo_ckd._prepare_to_plot(data_labels=systems, subplot_assignment=[0]),
        )
    )
    with_pc_plot_and_subplot_assignment = dict(
        zip(
            data_sets,
            plot_data_wo_ckd._prepare_to_plot(
                data_labels=systems, subplot_assignment=[0, 0, 1, 2, 3, 4]
            ),
        )
    )
    plot_data_wo_ckd.pc_plot_type = "bar"
    bar_plot_type = dict(
        zip(
            data_sets,
            plot_data_wo_ckd._prepare_to_plot(data_labels=systems, subplot_assignment=[0]),
        )
    )
    nested_dict_comparison(raw, ref["raw"])
    nested_dict_comparison(with_subplot_assignment, ref["with_subplot_assignment"])
    nested_dict_comparison(wo_ckd, ref["wo_ckd"])
    nested_dict_comparison(with_pc_plot_assignment, ref["with_pc_plot_assignment"])
    nested_dict_comparison(
        with_pc_plot_and_subplot_assignment, ref["with_pc_plot_and_subplot_assignment"]
    )
    nested_dict_comparison(bar_plot_type, ref["bar_plot_type"])
