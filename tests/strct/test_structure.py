"""Test Structure class."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import Structure
from aim2dat.io.yaml import load_yaml_file


STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"


def test_structure_print():
    """Test print statement of Structure class."""
    strct_dict = load_yaml_file(STRUCTURES_PATH + "Cs2Te_62_prim_kinds.yaml")
    structure = Structure(**strct_dict, label="test")
    assert structure.__str__() == (
        "----------------------------------------------------------------------\n"
        "-------------------------- Structure: test ---------------------------\n"
        "----------------------------------------------------------------------\n"
        "\n"
        " Formula: Cs8Te4\n"
        " PBC: [True True True]\n"
        "\n"
        "                                 Cell                                 \n"
        " Vectors: - [  9.5494   0.0000   0.0000]\n"
        "          - [  0.0000   5.8913   0.0000]\n"
        "          - [  0.0000   0.0000  11.6228]\n"
        " Lengths: [  9.5494   5.8913  11.6228]\n"
        " Angles: [ 90.0000  90.0000  90.0000]\n"
        " Volume: 653.8719\n"
        "\n"
        "                                Sites                                 \n"
        "  - Cs Cs1   [  3.3518   4.4185   0.8543] [  0.3510   0.7500   0.0735]\n"
        "  - Cs Cs1   [  0.2406   1.4728   2.0224] [  0.0252   0.2500   0.1740]\n"
        "  - Cs Cs1   [  5.0134   1.4728   3.7890] [  0.5250   0.2500   0.3260]\n"
        "  - Cs Cs1   [  8.1265   4.4185   4.9629] [  0.8510   0.7500   0.4270]\n"
        "  - Cs Cs2   [  1.4229   1.4728   6.6598] [  0.1490   0.2500   0.5730]\n"
        "  - Cs Cs2   [  4.5359   4.4185   7.8337] [  0.4750   0.7500   0.6740]\n"
        "  - Cs Cs2   [  9.3106   4.4185   9.6004] [  0.9750   0.7500   0.8260]\n"
        "  - Cs Cs2   [  6.1975   1.4728  10.7743] [  0.6490   0.2500   0.9270]\n"
        "  - Te Te    [  7.1907   4.4185   1.3017] [  0.7530   0.7500   0.1120]\n"
        "  - Te Te    [  2.4160   4.4185   4.5096] [  0.2530   0.7500   0.3880]\n"
        "  - Te Te    [  7.1334   1.4728   7.1131] [  0.7470   0.2500   0.6120]\n"
        "  - Te Te    [  2.3587   1.4728  10.3210] [  0.2470   0.2500   0.8880]\n"
        "----------------------------------------------------------------------"
    )

    strct_dict = load_yaml_file(STRUCTURES_PATH + "NH3.yaml")
    structure = Structure(**strct_dict)
    assert structure.__str__() == (
        "----------------------------------------------------------------------\n"
        "-------------------------- Structure: None ---------------------------\n"
        "----------------------------------------------------------------------\n"
        "\n"
        " Formula: NH3\n"
        " PBC: [False False False]\n"
        "\n"
        "                                Sites                                 \n"
        "  - N  None  [  0.0000   0.0000   0.0000]\n"
        "  - H  None  [  0.0000   0.0000   1.0080]\n"
        "  - H  None  [  0.9504   0.0000  -0.3360]\n"
        "  - H  None  [ -0.4752  -0.8230  -0.3360]\n"
        "----------------------------------------------------------------------"
    )


def test_structure_features():
    """Test features of Structure class."""
    strct_dict = load_yaml_file(STRUCTURES_PATH + "Cs2Te_62_prim_kinds.yaml")
    structure = Structure(**strct_dict)
    for idx, (el, pos) in enumerate(structure):
        assert el == strct_dict["elements"][idx]
        for idx0 in range(3):
            assert (
                abs(pos[idx0] - strct_dict["positions"][idx][idx0]) < 0.00001
            ), "Positions don't match."
    for idx, (el, kind, pos) in enumerate(structure.iter_sites(get_kind=True, get_cart_pos=True)):
        assert el == strct_dict["elements"][idx]
        assert kind == strct_dict["kinds"][idx]
        for idx0 in range(3):
            assert (
                abs(pos[idx0] - strct_dict["positions"][idx][idx0]) < 0.00001
            ), "Positions don't match."
    assert "positions" in structure
    assert structure["kinds"] == tuple(strct_dict["kinds"])

    with pytest.raises(ValueError) as error:
        structure.set_positions([[0.0, 0.0, 0.0]])
    assert str(error.value) == "`elements` and `positions` must have the same length."


def test_list_methods():
    """Test listing different method categories."""
    assert Structure.import_methods == [
        "from_file",
        "from_ase_atoms",
        "from_pymatgen_structure",
        "from_aiida_structuredata",
    ]
    assert Structure.export_methods == [
        "to_file",
        "to_ase_atoms",
        "to_pymatgen_structure",
        "to_aiida_structuredata",
    ]
    assert Structure.analysis_methods == [
        "determine_point_group",
        "determine_space_group",
        "calculate_distance",
        "calculate_angle",
        "calculate_dihedral_angle",
        "calculate_voronoi_tessellation",
        "calculate_coordination",
        "calculate_ffingerprint",
    ]
    assert Structure.manipulation_methods == [
        "delete_atoms",
        "scale_unit_cell",
        "substitute_elements",
    ]


def test_wrap_positions(structure_comparison):
    """Test wrapping positions onto unit cell."""
    strct_dict = load_yaml_file(STRUCTURES_PATH + "GaAs_216_conv.yaml")
    structure = Structure(**strct_dict)
    # TODO check get_positions wrap
    structure_comparison(structure, strct_dict)
    structure = Structure(**strct_dict, wrap=True)
    strct_dict["positions"][1][1] -= strct_dict["cell"][1][1]
    structure_comparison(structure, strct_dict)
