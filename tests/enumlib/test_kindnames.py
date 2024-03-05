"""Test for the enumlib utils module."""

# Third party library imports
import pytest
import aiida.orm as aiida_orm

# Internal library imports
from aim2dat.aiida_workflows.enumlib.utils import get_kindnames

# aiida.load_profile("test_profile")


@pytest.mark.parametrize(
    "kind_names,to_enumerate",
    [
        (
            ["Al", "F", "Li", "Ni0", "O1"],
            [["Al", "Li"], ["Al", "Li"], ["Ni0"], ["F"], ["O1"]],
        ),
        (["Al", "F", "Li", "Ni0", "O1"], {"Ni1": ("Al", "Li"), "O": "F"}),
    ],
)
def test_get_kindnames(aiida_profile, kind_names, to_enumerate):
    """Test processing of kindnames."""
    structure = aiida_orm.StructureData()
    cell = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    structure.cell = cell
    structure.append_atom(name="Ni1", symbols=("Ni",), weights=(1.0,), position=[0.0, 0.0, 0.0])
    structure.append_atom(name="Ni1", symbols=("Ni",), weights=(1.0,), position=[1.0, 0.0, 0.0])
    structure.append_atom(name="Ni0", symbols=("Ni",), weights=(1.0,), position=[0.0, 1.0, 0.0])
    structure.append_atom(name="O", symbols=("O",), weights=(1.0,), position=[0.0, 0.0, 1.0])
    structure.append_atom(name="O1", symbols=("O",), weights=(1.0,), position=[1.0, 0.0, 1.0])
    assert kind_names == sorted(get_kindnames(structure, to_enumerate))
