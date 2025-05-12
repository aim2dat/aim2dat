"""Test brillouin_zone_2d module."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import Structure, SurfaceGeneration
from aim2dat.strct.surface_utils import _transform_slab_to_primitive
from aim2dat.strct.brillouin_zone_2d import _get_kpath
from aim2dat.io import read_yaml_file

STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
BR_ZONE_PATH = os.path.dirname(__file__) + "/brillouin_zone_2d/"


@pytest.mark.parametrize(
    "bulk_crystal, miller_indices, ter",
    [
        ("GaAs_216_prim", (1, 0, 0), 1),
        # ("GaAs_216_prim", (1, 1, 0), 1),
        # ("Cs2Te_62_prim", (1, 0, 0), 1),
        # ("Cs2Te_62_prim", (1, 1, 1), 1),
        # ("Cs2Te_194_prim", (1, 0, 1), 1),
        # ("Cs2Te_194_prim", (5, 1, 2), 1),
    ],
)
def test_kpath(structure_comparison, nested_dict_comparison, bulk_crystal, miller_indices, ter):
    """Test k-path generation in 2d Brillouin zone."""
    mil_ind_str = "".join(str(mil_idx) for mil_idx in miller_indices)
    ref_outputs = read_yaml_file(BR_ZONE_PATH + f"{bulk_crystal}_{mil_ind_str}_{ter}.yaml")
    bulk_structure = Structure(**read_yaml_file(STRUCTURES_PATH + bulk_crystal + ".yaml"))
    bulk_structure.attributes["label"] = bulk_crystal
    surf_gen = SurfaceGeneration(bulk_structure)

    surf_c = surf_gen.generate_surface_slabs(miller_indices=miller_indices)
    prim_slab, layer_group = _transform_slab_to_primitive(surf_c[ter - 1], 0.005, -1, 0)
    kpath = _get_kpath(
        prim_slab["cell"],
        2,
        layer_group,
        0.015,
        1.0e-5,
    )
    structure_comparison(
        prim_slab,
        ref_outputs["prim_slab"],
        tolerance=1e-3,
    )
    assert layer_group == ref_outputs["layer_group"], "Wrong layer group."
    nested_dict_comparison(kpath, ref_outputs["kpath"])
