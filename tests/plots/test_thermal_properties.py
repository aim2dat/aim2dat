"""Tests for the thermal_properties module of the plots sub-package."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.plots.thermal_properties import ThermalPropertiesPlot, QHAPlot
from aim2dat.io.phonopy import read_thermal_properties, read_qha_properties
from aim2dat.io import read_yaml_file


MAIN_PATH = os.path.dirname(__file__)
REF_PATH = os.path.dirname(__file__) + "/thermal_properties/"
PHONOPY_HA_PATH = MAIN_PATH + "/../io/phonopy_ha/"
PHONOPY_QHA_PATH = MAIN_PATH + "/../io/phonopy_qha/"


@pytest.mark.skip
def test_thermal_properties_plot(nested_dict_comparison, matplotlib_figure_comparison):
    """Test ThermalPropertiesPlot class."""
    import_ref = read_yaml_file(REF_PATH + "thermal_properties_import_ref.yaml")
    plot_ref = read_yaml_file(REF_PATH + "thermal_properties_matplotlib_ref.yaml")
    thermal_properties = read_thermal_properties(
        PHONOPY_HA_PATH + "phonopy_disp.yaml",
        force_sets_file_name=PHONOPY_HA_PATH + "FORCE_SETS",
        mesh=20,
        t_max=500.0,
        t_step=50.0,
    )

    tp_plot = ThermalPropertiesPlot()
    tp_plot.import_thermal_properties("test", **thermal_properties)
    nested_dict_comparison(tp_plot._data, import_ref)
    matplotlib_figure_comparison(tp_plot.plot("test"), plot_ref)


def test_qha_plot(nested_dict_comparison):
    """Test QHAPlot class."""
    import_ref = read_yaml_file(REF_PATH + "qha_import_ref.yaml")
    plot_ref = read_yaml_file(REF_PATH + "qha_ref.yaml")
    qha_properties = read_qha_properties(
        thermal_properties_file_names=[
            PHONOPY_QHA_PATH + f"thermal_properties.yaml-{idx0}" for idx0 in range(-5, 6)
        ],
        ev_file_name=PHONOPY_QHA_PATH + "/e-v.dat",
    )
    qha_plot = QHAPlot()
    qha_plot.import_qha_properties(
        "test",
        qha_properties["temperatures"],
        volume_temperature=qha_properties["volume_temperature"],
        bulk_modulus_temperature=qha_properties["bulk_modulus_temperature"],
        thermal_expansion=qha_properties["thermal_expansion"],
        helmholtz_volume=qha_properties["helmholtz_volume"],
        volumes=qha_properties["volumes"],
    )
    nested_dict_comparison(qha_plot._data, import_ref)
    (
        data_sets,
        x_ticks,
        y_ticks,
        x_tick_labels,
        y_tick_labels,
        sec_axis,
    ) = qha_plot._prepare_to_plot("test", None)
    nested_dict_comparison(
        {
            "data_sets": data_sets,
            "x_ticks": x_ticks,
            "y_ticks": y_ticks,
            "x_tick_labels": x_tick_labels,
            "y_tick_labels": y_tick_labels,
            "sec_axis": sec_axis,
        },
        plot_ref["ref"],
    )
