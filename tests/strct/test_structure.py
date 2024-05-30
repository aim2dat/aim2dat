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


def test_zeo_write_to_file(tmpdir):
    """Test write structure to zeo input files."""
    strct_dict = load_yaml_file(STRUCTURES_PATH + "Cs2Te_62_prim_kinds.yaml")
    structure = Structure(**strct_dict, label="test")
    file = tmpdir.join("Cs2Te_62_prim_kinds.cssr")
    structure.to_file(file.strpath)  # or use str(file)
    assert file.read() == (
        "9.549362 5.891268 11.622767\n"
        "90.0 90.0 90.0 SPGR = Pm\n"
        "12 0\n"
        "0 test\n"
        "  1 Cs 0.3510000000000001 0.75 0.0735 0 0 0 0 0 0 0 0 0\n"
        "  2 Cs 0.025200000000000004 0.25 0.174 0 0 0 0 0 0 0 0 0\n"
        "  3 Cs 0.5250000000000001 0.25 0.326 0 0 0 0 0 0 0 0 0\n"
        "  4 Cs 0.851 0.75 0.42700000000000005 0 0 0 0 0 0 0 0 0\n"
        "  5 Cs 0.149 0.25 0.573 0 0 0 0 0 0 0 0 0\n"
        "  6 Cs 0.475 0.75 0.674 0 0 0 0 0 0 0 0 0\n"
        "  7 Cs 0.9750000000000001 0.75 0.8260000000000001 0 0 0 0 0 0 0 0 0\n"
        "  8 Cs 0.649 0.25 0.9270000000000002 0 0 0 0 0 0 0 0 0\n"
        "  9 Te 0.753 0.75 0.112 0 0 0 0 0 0 0 0 0\n"
        "  10 Te 0.253 0.75 0.388 0 0 0 0 0 0 0 0 0\n"
        "  11 Te 0.747 0.25 0.612 0 0 0 0 0 0 0 0 0\n"
        "  12 Te 0.24700000000000005 0.25 0.8880000000000001 0 0 0 0 0 0 0 0 0"
    )
    file = tmpdir.join("Cs2Te_62_prim_kinds.v1")
    structure.to_file(file.strpath)  # or use str(file)
    assert file.read() == (
        "Unit cell vectors:\n"
        "va= 9.549362 0.0 0.0\n"
        "vb= 0.0 5.891268 0.0\n"
        "vc= 0.0 0.0 11.622767\n"
        "12\n"
        "Cs 3.3518260620000007 4.418451 0.8542733745\n"
        "Cs 0.24064392240000002 1.472817 2.022361458\n"
        "Cs 5.013415050000001 1.472817 3.789022042\n"
        "Cs 8.126507062 4.418451 4.962921509\n"
        "Cs 1.422854938 1.472817 6.659845491\n"
        "Cs 4.53594695 4.418451 7.8337449580000005\n"
        "Cs 9.31062795 4.418451 9.600405542\n"
        "Cs 6.197535938000001 1.472817 10.774305009\n"
        "Te 7.190669586 4.418451 1.301749904\n"
        "Te 2.415988586 4.418451 4.509633596\n"
        "Te 7.133373414 1.472817 7.113133404\n"
        "Te 2.3586924140000005 1.472817 10.321017096"
    )
    file = tmpdir.join("Cs2Te_62_prim_kinds.cuc")
    structure.to_file(file.strpath)  # or use str(file)
    assert file.read() == (
        "Processing: test\n"
        "Unit_cell: 9.549362 5.891268 11.622767 90.0 90.0 90.0\n"
        "Cs 0.3510000000000001 0.75 0.0735\n"
        "Cs 0.025200000000000004 0.25 0.174\n"
        "Cs 0.5250000000000001 0.25 0.326\n"
        "Cs 0.851 0.75 0.42700000000000005\n"
        "Cs 0.149 0.25 0.573\n"
        "Cs 0.475 0.75 0.674\n"
        "Cs 0.9750000000000001 0.75 0.8260000000000001\n"
        "Cs 0.649 0.25 0.9270000000000002\n"
        "Te 0.753 0.75 0.112\n"
        "Te 0.253 0.75 0.388\n"
        "Te 0.747 0.25 0.612\n"
        "Te 0.24700000000000005 0.25 0.8880000000000001"
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
        "to_dict",
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
