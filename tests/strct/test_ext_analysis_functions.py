"""Test external structure analysis functions."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import Structure
from aim2dat.strct.ext_analysis import determine_molecular_fragments
from aim2dat.io.yaml import load_yaml_file

STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
FRAG_PATH = os.path.dirname(__file__) + "/fragment_analysis/"


@pytest.mark.parametrize(
    "system, file_suffix, backend",
    [
        ("Benzene", ".xyz", "ase"),
        ("ZIF-8", ".cif", "internal"),
    ],
)
def test_determine_molecular_fragments_function(
    structure_comparison, system, file_suffix, backend
):
    """Test determine_molecular_fragments function."""
    kwargs, ref = load_yaml_file(FRAG_PATH + system + ".yaml")
    strct = Structure.from_file(STRUCTURES_PATH + system + file_suffix, backend=backend)
    fragments = determine_molecular_fragments(strct, **kwargs)
    for frag, frag_ref in zip(fragments, ref):
        structure_comparison(frag, Structure(**frag_ref))
        assert frag.site_attributes["parent_indices"] == tuple(
            frag_ref["site_attributes"]["parent_indices"]
        )
