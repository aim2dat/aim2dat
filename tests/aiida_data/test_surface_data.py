"""Test surface data."""

# Standard library imports
import os

# Third party library imports
import pytest
import aiida.orm as aiida_orm

# Internal library imports
from aim2dat.io import read_yaml_file
from aim2dat.strct import Structure, SurfaceGeneration
from aim2dat.aiida_workflows.utils import create_surface_slab
from aim2dat.ext_interfaces.aiida import (
    _extract_dict_from_aiida_structure_node,
)

STRUCTURES_PATH = os.path.dirname(__file__) + "/../strct/structures/"
REF_PATH = os.path.dirname(__file__) + "/ref_data/"


def test_surface_data(structure_comparison):
    """Test the surface data type."""
    miller_indices = (1, 0, 0)
    termination = 1
    tolerance = 0.005
    surface_area = 68.472835298556
    aperiodic_dir = 2

    surf_gen = SurfaceGeneration(
        structure=Structure(
            **read_yaml_file(STRUCTURES_PATH + "Cs2Te_62_prim.yaml"), label="Cs2Te_62_prim"
        )
    )
    ref_data = surf_gen.create_surface(miller_indices, termination, tolerance)
    surf_data = surf_gen.to_aiida_surfacedata(miller_indices, termination, tolerance)
    assert surf_data.miller_indices == miller_indices
    assert surf_data.termination == termination
    assert surf_data.aperiodic_dir == aperiodic_dir
    assert abs(surf_data.surface_area - surface_area) < 1e-5
    structure_comparison(surf_data.repeating_structure, ref_data["repeating_structure"])
    structure_comparison(surf_data.bottom_terminating_structure, ref_data["bottom_structure"])
    structure_comparison(surf_data.top_terminating_structure, ref_data["top_structure"])
    structure_comparison(surf_data.top_terminating_structure_nsym, ref_data["top_structure_nsym"])


@pytest.mark.aiida
@pytest.mark.parametrize(
    "miller_indices, termination",
    [
        ((1, 0, 0), 1),
        ((0, 1, 0), 1),
    ],
)
def test_create_surface_slab(
    structure_comparison, nested_dict_comparison, miller_indices, termination
):
    """Test the creation of surface slabs."""
    m_indices_str = "".join(str(idx0) for idx0 in miller_indices)
    ref_data = read_yaml_file(REF_PATH + f"surf_{m_indices_str}_{termination}_ref.yaml")
    tolerance = 0.005
    ref_data["slab"]["label"] = "test_label"

    surf_gen = SurfaceGeneration(
        structure=Structure(
            **read_yaml_file(STRUCTURES_PATH + "Cs2Te_62_prim.yaml"), label="Cs2Te_62_prim"
        )
    )
    surf_data = surf_gen.to_aiida_surfacedata(miller_indices, termination, tolerance)
    surf_output = create_surface_slab(
        surf_data,
        aiida_orm.Int(3),
        aiida_orm.Dict(
            dict={
                "return_primitive_slab": True,
                "return_path_p": True,
                "symmetrize": True,
                "label": "test_label",
            }
        ),
    )
    slab = _extract_dict_from_aiida_structure_node(surf_output["slab"])
    parameters = surf_output["parameters"].get_dict()
    structure_comparison(slab, ref_data["slab"])
    nested_dict_comparison(parameters, ref_data["parameters"])
