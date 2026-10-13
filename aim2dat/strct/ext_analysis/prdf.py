"""Calculate partial radial distribution function."""

# Standard library imports
from typing import Tuple

# Internal library imports
from aim2dat.strct.structure import Structure
from aim2dat.strct.ext_analysis.decorator import external_analysis_method
from aim2dat.strct.analysis.rdf import _calculate_prdf


@external_analysis_method(attr_mapping=None)
def calc_prdf(
    structure: Structure,
    r_max: float = 20.0,
    delta_bin: float = 0.005,
    distinguish_kinds: bool = False,
) -> Tuple[dict, dict]:
    """
    Calculate the partial radial distribution function.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure object.
    r_max : float (optional)
        Cut-off value for the maximum distance between two atoms in angstrom.
    delta_bin : float (optional)
        Bin size to discretize the function in angstrom.
    distinguish_kinds: bool (optional)
        Whether different kinds should be distinguished e.g. Ni0 and Ni1 would be considered as
        different elements if ``True``.

    Returns
    -------
    element_prdf : dict
        Dictionary containing all partial radial distribution functions of the structure
        summed over all atoms of the same element.
    atomic_prdf : dict
        Dictionary containing all individual partial radial distribution functions for each
        atomic site.
    """
    rdfs = _calculate_prdf(structure, r_max, delta_bin, distinguish_kinds, "prdf")

    for key, rdf in rdfs.items():
        if key.startswith("el_"):
            rdfs[key] = rdf.tolist()
        elif key.startswith("atomic_"):
            atomic_rdf = []
            for sit_idx, value in enumerate(rdf):
                atomic_rdf.append(value.tolist())
            rdfs[key] = atomic_rdf
    return rdfs
