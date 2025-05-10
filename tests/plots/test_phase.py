"""Test phase diagram plots."""

# Standard library imports
import os

# Third party library imports
import pytest
import pandas as pd

# Internal library imports
from aim2dat.plots import PhasePlot
from aim2dat.strct import StructureCollection
from aim2dat.ext_interfaces.pandas import _turn_dict_into_pandas_df
from aim2dat.io import read_yaml_file
import aim2dat.utils.chem_formula as utils_cf

PHASES_PATH = os.path.dirname(__file__) + "/phases_files/"


def test_properties():
    """Test properties of PhasePlot."""
    phase_plot = PhasePlot()
    with pytest.raises(TypeError) as error:
        phase_plot.elements = "_"
    assert str(error.value) == "`elements` needs to be of type list or tuple."
    phase_plot.elements = ["Cs", 22]
    assert phase_plot.elements == ["Cs", "Ti"]

    phase_plot.plot_type = "scatter"
    assert phase_plot.plot_type == "scatter"
    with pytest.raises(ValueError) as error:
        phase_plot.plot_type = "_"
    assert (
        str(error.value)
        == "`plot_type` '_' is not suppported. Supported options are 'scatter', 'numbers'."
    )

    phase_plot.show_crystal_system = True
    assert phase_plot.show_crystal_system

    phase_plot.top_labels = [{"Cs": 2, "Ti": 1}, "Cs4Ti2", ["Cs", "Ti", "Ti", "Ti"]]
    assert phase_plot.top_labels == [{"Cs": 2, "Ti": 1}, {"Cs": 4, "Ti": 2}, {"Cs": 1, "Ti": 3}]
    phase_plot.top_labels = "Cs4Ti2"
    assert phase_plot.top_labels == [{"Cs": 4, "Ti": 2}]


def test_add_data_point():
    """Test add_data_point function of PhasePlot."""
    phase_plot = PhasePlot()
    with pytest.raises(TypeError) as error:
        phase_plot.add_data_point("test", 10)
    assert str(error.value) == "`formula` needs to be of type list/tuple/str/dict."
    with pytest.raises(TypeError) as error:
        phase_plot.add_data_point("test", {"Cs": 1.0}, attributes="")
    assert str(error.value) == "`attributes` needs to be of type dict."

    for chem_f in [{"Cs": 1.0, "Te": 1.0}, "CsTe", ["Cs", "Te"]]:
        phase_plot = PhasePlot()
        phase_plot.add_data_point("test", chem_f)
        assert phase_plot._data["test"][0] == {
            "chem_formula": {"Cs": 1.0, "Te": 1.0},
            "attributes": {},
            "space_group": None,
        }

    phase_plot = PhasePlot()
    phase_plot.add_data_point("test", "CsTe", formation_energy=0.1, stability=0.2)
    assert phase_plot._data["test"][0] == {
        "chem_formula": {"Cs": 1.0, "Te": 1.0},
        "attributes": {
            "formation_energy": {"value": 0.1, "unit": None},
            "stability": {"value": 0.2, "unit": None},
        },
        "space_group": None,
    }


@pytest.mark.parametrize(
    "system",
    ["mp_Cs-Sb"],
)
def test_pandas_df_import(nested_dict_comparison, system):
    """Test pandas data frame import."""
    pandas_dict = dict(read_yaml_file(PHASES_PATH + system + "_df.yaml"))
    ref_dict = read_yaml_file(PHASES_PATH + system + "_import_ref.yaml")
    df = _turn_dict_into_pandas_df(pandas_dict)
    phase_plot = PhasePlot()
    phase_plot.import_from_pandas_df("test", df)
    for entry in phase_plot._data["test"]:
        for attr_label, attr_details in entry["attributes"].items():
            if attr_details["value"] is pd.NA:
                attr_details["value"] = None
    nested_dict_comparison(phase_plot._data, ref_dict)


@pytest.mark.parametrize(
    "system",
    ["csp_Cs-Te"],
)
def test_strct_c_import(nested_dict_comparison, system):
    """Test StructureCollection import."""
    strct_c = StructureCollection()
    strct_c.import_from_hdf5_file(PHASES_PATH + system + ".h5")
    phase_plot = PhasePlot()
    phase_plot.import_from_structure_collection("test", strct_c)
    for entry, structure in zip(phase_plot._data["test"], strct_c):
        assert entry["chem_formula"] == utils_cf.transform_list_to_dict(structure["elements"])
        for attr_label, attr_details in entry["attributes"].items():
            if isinstance(attr_details, dict):
                attr_details = attr_details["value"]
            strct_val = structure["attributes"][attr_label]
            if isinstance(structure["attributes"][attr_label], dict):
                strct_val = strct_val["value"]
            assert attr_details == strct_val


@pytest.mark.parametrize(
    "system,test_case,input_type",
    [
        ("mp_Cs-Sb", "f_energy", "pandas"),
        ("mp_Cs-Sb", "stability", "pandas"),
        ("mp_Cs-Sb", "numbers", "pandas"),
        ("csp_Cs-Te", "f_energy", "strct_C"),
        ("mp_Cs-Sb", "direct_band_gap", "pandas"),
    ],
)
def test_prepare_plot(nested_dict_comparison, system, test_case, input_type):
    """Test _prepare_to_plot function for different use cases."""
    ref_dict = read_yaml_file(PHASES_PATH + system + "_" + test_case + "_ref.yaml")

    phase_plot = PhasePlot()
    for label, value in ref_dict["attributes"].items():
        setattr(phase_plot, label, value)
    if input_type == "pandas":
        pandas_dict = dict(read_yaml_file(PHASES_PATH + system + "_df.yaml"))
        df = _turn_dict_into_pandas_df(pandas_dict)
        phase_plot.import_from_pandas_df("test", df)
    else:
        strct_c = StructureCollection()
        strct_c.import_from_hdf5_file(PHASES_PATH + system + ".h5")
        phase_plot.import_from_structure_collection("test", strct_c)
    (
        data_sets,
        x_ticks,
        y_ticks,
        x_tick_labels,
        y_tick_labels,
        sec_axis,
    ) = phase_plot._prepare_to_plot(**ref_dict["plot_args"])
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
