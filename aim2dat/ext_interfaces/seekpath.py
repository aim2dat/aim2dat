"""Interface to the seekpath library."""

# Third party library imports
from seekpath.getpaths import get_explicit_k_path

# Internal library imports
from aim2dat.ext_interfaces.spglib import (
    _transfrom_structure_to_cell,
    _transform_cell_to_structure,
)


def _get_explicit_k_path(
    structure,
    with_time_reversal=True,
    reference_distance=0.025,
    recipe="hpkot",
    threshold=1e-07,
    symprec=1e-05,
    angle_tolerance=-1.0,
):
    cell = _transfrom_structure_to_cell(structure)
    kpath = get_explicit_k_path(
        cell,
        with_time_reversal=with_time_reversal,
        reference_distance=reference_distance,
        recipe=recipe,
        threshold=threshold,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    prim_cell = (
        kpath.pop("primitive_lattice"),
        kpath.pop("primitive_positions"),
        kpath.pop("primitive_types"),
    )
    conv_cell = (kpath.pop("conv_lattice"), kpath.pop("conv_positions"), kpath.pop("conv_types"))
    return _transform_cell_to_structure(prim_cell), _transform_cell_to_structure(conv_cell), kpath
