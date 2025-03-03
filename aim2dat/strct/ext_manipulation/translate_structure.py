"""Module that implements routines to translate a structure."""

# Standard library imports
from typing import List, Union

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.strct.ext_manipulation.decorator import (
    external_manipulation_method,
)
from aim2dat.strct.ext_manipulation.utils import _check_distances
from aim2dat.strct import Structure


# TODO add tests for this function
@external_manipulation_method
def translate_structure(
    structure: Structure,
    vector: List[float],
    site_indices: Union[None, List[int]] = None,
    wrap: bool = False,
    dist_threshold: float = None,
    change_label: bool = False,
) -> Structure:
    """
    Work in progress
    """
    # TODO add doc-string

    # TODO include checks that vector has length of 3 and float type
    if site_indices is None:
        site_indices = list(range(len(structure)))

    positions = np.array(structure.positions)
    for idx in site_indices:
        positions[idx] += np.array(vector)
    new_structure = structure.to_dict()
    new_structure["positions"] = positions
    new_structure = Structure(**new_structure, wrap=wrap)
    _check_distances(new_structure, site_indices, dist_threshold, False)
    return new_structure, "_translated"  # TODO add length of vector??
