"""Test generate functions of the StructureCollection."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import SurfaceGeneration, Structure
from aim2dat.io import read_yaml_file


SURFACES_PATH = os.path.dirname(__file__) + "/surfaces/"
STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"


@pytest.mark.parametrize(
    "bulk_crystal, miller_indices",
    [
        ("TiO2_136", (1, 0, 0)),
        ("TiO2_136", (1, 0, 1)),
        ("TiO2_136", (0, 0, 1)),
        ("Cs2Te_62_prim", (1, 0, 0)),
    ],
)
def test_generate_surfaces(structure_comparison, bulk_crystal, miller_indices):
    """Test surface generation fucntion."""
    mil_ind_str = "".join(str(mil_idx) for mil_idx in miller_indices)
    ref_outputs = read_yaml_file(SURFACES_PATH + bulk_crystal + "_" + mil_ind_str + ".yaml")
    bulk_structure = read_yaml_file(STRUCTURES_PATH + bulk_crystal + ".yaml")
    bulk_structure = Structure(**bulk_structure)

    bulk_structure.label = bulk_crystal
    surf_gen = SurfaceGeneration(bulk_structure)
    surf_c = surf_gen.generate_surface_slabs(
        miller_indices=miller_indices, **ref_outputs["parameters"]
    )
    for label, ref_structure in ref_outputs["ref"].items():
        ref_structure["label"] = label
        structure_comparison(surf_c[label], ref_structure, tolerance=1e-3)
