"""Tests for the band_structure_dos module."""

# Standard library imports
import os

# Third party library imports
import pytest
import numpy as np

# Internal library imports
from aim2dat.plots.band_structure_dos import BandStructurePlot, DOSPlot, BandStructureDOSPlot
from aim2dat.io import read_yaml_file

MAIN_PATH = os.path.dirname(__file__)
BAND_STRUCTURE_PATH = MAIN_PATH + "/band_structure_files/"
DOS_PATH = MAIN_PATH + "/dos_files/"


@pytest.mark.parametrize("test_case", [("Si")])
def test_band_analysis_functions(test_case):
    """Test the BandStructurePlot class."""
    # Read test file:
    inputs = dict(read_yaml_file(BAND_STRUCTURE_PATH + f"{test_case}_band_structure.yaml"))

    # Read reference file:
    ref = dict(
        read_yaml_file(BAND_STRUCTURE_PATH + f"{test_case}_band_analysis_ref.yaml", typ="unsafe")
    )

    # Create object:
    band_structure_plt = BandStructurePlot()
    band_structure_plt.import_band_structure(**inputs)

    # Check output:
    for fct_name, fct_details in ref.items():
        function = getattr(band_structure_plt, fct_name)
        output = function(**fct_details["inputs"])
        assert output == fct_details["result"]


@pytest.mark.parametrize("system", [("Si")])
def test_band_structure_plot_matplotlib_backend(matplotlib_figure_comparison, system):
    """Test matplotlib backend with the BandStructurePlot class."""
    # Read test file:
    inputs = dict(read_yaml_file(BAND_STRUCTURE_PATH + f"{system}_band_structure.yaml"))

    # Read reference file:
    ref = dict(read_yaml_file(BAND_STRUCTURE_PATH + f"{system}_matplotlib_ref.yaml", typ="unsafe"))

    # Create object:
    band_structure_plt = BandStructurePlot()
    band_structure_plt.import_band_structure(**inputs)

    # Compare figure:
    matplotlib_figure_comparison(band_structure_plt.plot(inputs["data_label"]), ref)


def test_dos_errors():
    """Test errors of DOSPlot."""
    dos_plot = DOSPlot()

    with pytest.raises(ValueError) as error:
        dos_plot.smearing_method = "test"
    assert (
        str(error.value)
        == "Smearing method 'test' is not supported. "
        + "Available methods are: 'gaussian', 'lorentzian'."
    )

    with pytest.raises(TypeError) as error:
        dos_plot.pdos_plot_type = 10
    assert str(error.value) == "`pdos_plot_type` needs to be of type str."
    with pytest.raises(ValueError) as error:
        dos_plot.pdos_plot_type = "_"
    assert str(error.value) == "`pdos_plot_type` must contain 'line' and/or 'fill'."

    dos_plot.import_total_dos("test", [0.0], [1.0])
    with pytest.raises(ValueError) as error:
        dos_plot.import_total_dos("test", [0.0], [1.0])
    assert str(error.value) == "Data label 'test' already contains tDOS data."
    with pytest.raises(ValueError) as error:
        dos_plot.import_total_dos("test_2", [0.0], [1.0, 2.0])
    assert str(error.value) == "Energy and DOS have different shapes: 1 != 2."

    dos_plot.import_projected_dos("test", [0.0], [{"kind": "At", "s": [1.0]}])
    with pytest.raises(ValueError) as error:
        dos_plot.import_projected_dos("test", [0.0], [{"kind": "At", "s": [1.0]}])
    assert str(error.value) == "Data label 'test' already contains pDOS data."
    with pytest.raises(ValueError) as error:
        dos_plot.import_projected_dos("test_2", [0.0], [{"kind": "At", "s": [1.0, 2.0]}])
    assert str(error.value) == "Energy and DOS have different shapes: 1 != 2."
    with pytest.raises(ValueError) as error:
        dos_plot.import_projected_dos("test_2", [0.0], [{"kind": "At", "x": [1.0, 2.0]}])
    assert str(error.value) == "Orbital projection on 'x' could not be processed."
    with pytest.raises(ValueError) as error:
        dos_plot.import_projected_dos("test_2", [0.0], [{"kind": "At", "sx": [1.0, 2.0]}])
    assert str(error.value) == "Orbital projection on 'sx' could not be processed."


@pytest.mark.parametrize(
    "system",
    [
        "Si_at_qe_smearing",
        "Si_at_qe_sum_kinds",
        "Si_uks",
        "CsTe_detect_eq_kinds",
        "CsTe_custom_kind_dict",
    ],
)
def test_dos_import(nested_dict_comparison, system):
    """Test pDOS and tDOS import."""
    inputs = dict(read_yaml_file(DOS_PATH + f"{system}.yaml"))
    ref = dict(read_yaml_file(DOS_PATH + f"{system}_import_ref.yaml"))
    dos_plot = DOSPlot(**inputs["attributes"])
    if "import_pdos_input" in inputs:
        dos_plot.import_projected_dos("test", **inputs["import_pdos_input"])
    if "import_tdos_input" in inputs:
        dos_plot.import_total_dos("test", **inputs["import_tdos_input"])
    dos_data = dos_plot._data["test"]
    for d_type in ["pdos", "tdos"]:
        if d_type in dos_data:
            dos_data[d_type]["dos"] = [
                [float(val) for val in row] for row in dos_data[d_type]["dos"]
            ]
            dos_data[d_type]["energy"] = [float(val) for val in dos_data[d_type]["energy"]]
    nested_dict_comparison(dos_data, ref)


def test_shift_dos(nested_dict_comparison):
    """Test shif_dos function of the DOSPlot class."""
    inputs = dict(read_yaml_file(DOS_PATH + "Si_at_qe.yaml"))
    ref = dict(read_yaml_file(DOS_PATH + "Si_at_qe_import_ref.yaml"))
    dos_plot = DOSPlot(**inputs["attributes"])
    dos_plot.import_projected_dos("test", **inputs["import_pdos_input"])
    dos_plot.import_total_dos("test", **inputs["import_tdos_input"])
    dos_plot.shift_dos("test", 10.0)
    ref["pdos"]["energy"] = [val + 10.0 for val in ref["pdos"]["energy"]]
    ref["tdos"]["energy"] = [val + 10.0 for val in ref["tdos"]["energy"]]
    dos_data = dos_plot._data["test"]
    for d_type in ["pdos", "tdos"]:
        dos_data[d_type]["dos"] = [[float(val) for val in row] for row in dos_data[d_type]["dos"]]
        dos_data[d_type]["energy"] = [float(val) for val in dos_data[d_type]["energy"]]
    nested_dict_comparison(dos_data, ref)


@pytest.mark.parametrize("system", ["Si_at"])
def test_pdos_import_aiida_xydata(nested_dict_comparison, aiida_create_xydata, system):
    """Test pDOS import via AiidA data nodes."""
    inputs = dict(read_yaml_file(DOS_PATH + f"{system}_aiida_pdos.yaml"))
    y_data = [[], [], []]
    for label, data_set in inputs["pdosdata"].items():
        x_data = data_set["x_data"]
        y_data[0] += [np.array(y_vals) for y_vals in data_set["y_data"][0]]
        y_data[1] += [label + "_" + orb_label for orb_label in data_set["y_data"][1]]
        y_data[2] += data_set["y_data"][2]
    x_data[0] = np.array(x_data[0])
    inputs["pdosdata"] = aiida_create_xydata(x_data, y_data)

    dos_plot = DOSPlot()
    dos_plot.smearing_delta = 0.005
    dos_plot.smearing_sigma = 5.0
    dos_plot.import_from_aiida_xydata("test", **inputs)
    dos_data = dos_plot._data["test"]
    for d_type in ["pdos", "tdos"]:
        if d_type in dos_data:
            dos_data[d_type]["dos"] = [
                [float(val) for val in row] for row in dos_data[d_type]["dos"]
            ]
            dos_data[d_type]["energy"] = [float(val) for val in dos_data[d_type]["energy"]]
    reference = dict(read_yaml_file(DOS_PATH + f"{system}_aiida_import_ref.yaml", typ="unsafe"))
    nested_dict_comparison(dos_data, reference)


@pytest.mark.parametrize("system,test_case", [("Si_uks", "sum_pdos"), ("Si_at_qe", "linefill")])
def test_dos_prepare_plot(nested_dict_comparison, system, test_case):
    """Test _prepare_to_plot function for different use cases."""
    inputs = dict(read_yaml_file(DOS_PATH + f"{system}.yaml"))
    ref_dict = dict(read_yaml_file(DOS_PATH + f"{system}_{test_case}_prepare_plot_ref.yaml"))
    dos_plot = DOSPlot(**inputs["attributes"], **ref_dict["attributes"])
    if "import_pdos_input" in inputs:
        dos_plot.import_projected_dos("test", **inputs["import_pdos_input"])
    if "import_tdos_input" in inputs:
        dos_plot.import_total_dos("test", **inputs["import_tdos_input"])
    (
        data_sets,
        x_ticks,
        y_ticks,
        x_tick_labels,
        y_tick_labels,
        sec_axis,
    ) = dos_plot._prepare_to_plot(**ref_dict["plot_args"])
    for idx1 in range(len(data_sets)):
        for idx2 in range(len(data_sets[idx1])):
            for label in ["x_values", "y_values"]:
                if label in data_sets[idx1][idx2]:
                    data_sets[idx1][idx2][label] = [
                        float(val) for val in data_sets[idx1][idx2][label]
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


def test_band_structure_dos_prepare_plot(nested_dict_comparison):
    """Test _prepare_to_plot function of BandStructureDOSPlot class."""
    inputs_dos = dict(read_yaml_file(DOS_PATH + "Si_at_qe.yaml"))
    inputs_bands = dict(read_yaml_file(BAND_STRUCTURE_PATH + "Si_band_structure.yaml"))
    ref_dict = dict(
        read_yaml_file(BAND_STRUCTURE_PATH + "Si_band_structure_dos_prepare_plot_ref.yaml")
    )

    bands_dos_plot = BandStructureDOSPlot(**ref_dict["attributes"], **inputs_dos["attributes"])
    bands_dos_plot.import_band_structure(**inputs_bands)
    bands_dos_plot.import_projected_dos("Si_band_structure", **inputs_dos["import_pdos_input"])
    bands_dos_plot.import_total_dos("Si_band_structure", **inputs_dos["import_tdos_input"])
    (
        data_sets,
        x_ticks,
        y_ticks,
        x_tick_labels,
        y_tick_labels,
        sec_axis,
    ) = bands_dos_plot._prepare_to_plot(**ref_dict["plot_args"])
    for idx1 in range(len(data_sets)):
        for idx2 in range(len(data_sets[idx1])):
            for label in ["x_values", "y_values"]:
                if label in data_sets[idx1][idx2]:
                    data_sets[idx1][idx2][label] = [
                        float(val) for val in data_sets[idx1][idx2][label]
                    ]
            if "x" in data_sets[idx1][idx2]:
                data_sets[idx1][idx2]["x"] = float(data_sets[idx1][idx2]["x"])
    for idx1 in range(len(x_ticks)):
        if x_ticks[idx1] is not None:
            for idx2 in range(len(x_ticks[idx1])):
                x_ticks[idx1][idx2] = float(x_ticks[idx1][idx2])
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
