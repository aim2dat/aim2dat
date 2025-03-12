"""Test cube data."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.io import read_yaml_file
from aim2dat.aiida_data.gaussian_cube_data import GaussianCubeData

CUBE_PATH = os.path.dirname(__file__) + "/cube_files/"
REF_PATH = os.path.dirname(__file__) + "/ref_data/"


@pytest.mark.parametrize(
    "system",
    ["Si_crystal_wfn_03_1"],
)
def test_gaussian_cube_data(structure_comparison, nested_dict_comparison, system):
    """Test gaussian cube data type."""
    ref = read_yaml_file(REF_PATH + system + "_ref.yaml")
    with open(CUBE_PATH + system + ".cube", "r") as fobj:
        cube_data = GaussianCubeData.set_from_file(fobj)
        for idx, (line, ref_line) in enumerate(zip(cube_data.get_content().splitlines(), fobj)):
            assert (
                line == ref_line
            ), f"Retrieved file content does not match original cube file on line {idx}."

    # Trigger errors:
    with pytest.raises(ValueError) as error:
        cube_data.get_structure(unit="wrong")
    assert str(error.value) == "Unit 'wrong' is not supported."

    # Compare structures:
    structure_comparison(cube_data.get_structure(unit="angstrom"), ref.pop("structure"))

    # Compare cube data:
    data = {}
    for attr in [
        "title",
        "comment",
        "origin",
        "cell",
        "shape",
        "atomic_numbers",
        "atomic_charges",
        "atomic_positions",
        "dset_ids",
    ]:
        data[attr] = getattr(cube_data, attr)
    data["data"] = cube_data.get_cube_data().tolist()
    nested_dict_comparison(data, ref)
