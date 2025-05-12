"""Test SurfacePlot class."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.plots import SurfacePlot
from aim2dat.io import read_yaml_file
from aim2dat.ext_interfaces.pandas import _turn_dict_into_pandas_df


SURFACE_PATH = os.path.dirname(__file__) + "/surface/"


@pytest.mark.parametrize(
    "prop,value,ref_value,error_type,error_m",
    [
        ("plot_type", "chem_potential", "chem_potential", None, None),
        (
            "plot_type",
            "_",
            None,
            ValueError,
            "`plot_type` '_' is not supported. Supported values are "
            + "'chem_potential' or 'excess_atoms'.",
        ),
        (
            "plot_properties",
            0,
            None,
            TypeError,
            "`plot_properties` needs to be of type str or tuple/list of str objects.",
        ),
        (
            "plot_properties",
            ["surface_energy", "band_gap"],
            None,
            ValueError,
            "'surface_energy' can only be plotted as a single property.",
        ),
        (
            "plot_properties",
            ["ionization_potential", "band_gap"],
            ("ionization_potential", "band_gap"),
            None,
            None,
        ),
        ("plot_properties", "surface_energy", ("surface_energy",), None, None),
        ("area_unit", "ang", "ang", None, None),
        ("energy_unit", "Rydberg", "rydberg", None, None),
    ],
)
def test_properties_validation(prop, value, ref_value, error_type, error_m):
    """Test setting and validation of properties."""
    splot = SurfacePlot()
    if error_type is None:
        setattr(splot, prop, value)
        assert getattr(splot, prop) == ref_value, f"{prop} validation gives wrong value."
    else:
        with pytest.raises(error_type) as error:
            setattr(splot, prop, value)
        assert str(error.value) == error_m


def test_import_functions(create_plot_object, nested_dict_comparison):
    """Test importing data sets."""
    ref_dict = read_yaml_file(SURFACE_PATH + "Cs18Te9_import_ref.yaml")
    splot = create_plot_object(SurfacePlot, "matplotlib", ref_dict)
    nested_dict_comparison(splot.elemental_phases, ref_dict["ref"]["elemental_phases"])
    nested_dict_comparison(splot.bulk_phase, ref_dict["ref"]["bulk_phase"])
    nested_dict_comparison(splot._data, ref_dict["ref"]["data"])


def test_import_pandas_df(
    aiida_create_surfacedata,
    aiida_create_structuredata,
    aiida_create_bandsdata,
    nested_dict_comparison,
):
    """Test import pandas data frame."""
    ref_dict = read_yaml_file(SURFACE_PATH + "Cs8Te4_pandas_df_import_ref.yaml")
    pandas_dict = {
        "parent_node": [],
        "optimized_structure": [],
        "total_energy (Hartree)": ref_dict["pandas_df"]["total_energy (Hartree)"],
        "band_structure": [],
    }
    for surf_dict, opt_dict, bnd_dict in zip(
        ref_dict["pandas_df"]["parent_node"],
        ref_dict["pandas_df"]["optimized_structure"],
        ref_dict["pandas_df"]["band_structure"],
    ):
        surf_node = aiida_create_surfacedata(**surf_dict)
        surf_node.store()
        opt_node = aiida_create_structuredata(opt_dict)
        opt_node.store()
        bnd_node = aiida_create_bandsdata(**bnd_dict)
        bnd_node.store()
        pandas_dict["parent_node"].append(surf_node.pk)
        pandas_dict["optimized_structure"].append(opt_node.pk)
        pandas_dict["band_structure"].append(bnd_node.pk)

    pandas_df = _turn_dict_into_pandas_df(pandas_dict)
    splot = SurfacePlot()
    splot.import_from_pandas_df("test", pandas_df, extract_electronic_properties=True)
    nested_dict_comparison(splot._data, ref_dict["ref"])


@pytest.mark.parametrize(
    "system,test_case",
    [
        ("Cs18Te9", "chem_pot-surface_energy"),
        ("Cs18Te9", "chem_pot-ion_pot_band_gap"),
        ("Cs18Te9", "excess_atoms-ion_pot_band_gap"),
    ],
)
def test_prepare_plot(create_plot_object, nested_dict_comparison, system, test_case):
    """Test _prepare_to_plot function."""
    import_dict = read_yaml_file(SURFACE_PATH + system + "_import_ref.yaml")
    ref_dict = read_yaml_file(SURFACE_PATH + system + "_" + test_case + "_ref.yaml")
    import_dict["properties"] = ref_dict["properties"]
    splot = create_plot_object(SurfacePlot, "matplotlib", import_dict)
    (
        data_sets,
        x_ticks,
        y_ticks,
        x_tick_labels,
        y_tick_labels,
        sec_axis,
    ) = splot._prepare_to_plot(["100", "001"], [0, 1])
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
