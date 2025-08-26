"""Calculate f-fingerprint order parameters."""

# Standard library imports
from typing import List, Tuple

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.strct.structure import Structure
from aim2dat.strct.ext_analysis.decorator import external_analysis_method
from aim2dat.strct.analysis.rdf import _calculate_weights


@external_analysis_method(attr_mapping={"ffingerprint_order_p": ("el_order_p",)})
def calc_ffingerprint_order_p(
    structure: Structure,
    r_max: float = 15.0,
    delta_bin: float = 0.005,
    sigma: float = 0.05,
    distinguish_kinds: bool = False,
) -> Tuple[float, List[float]]:
    """
    Calculate order parameters for the total structure and for each individual site.

    The calculation is based on equation (5) in :doi:`10.1016/j.cpc.2010.06.007`.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure object.
    r_max : float (optional)
        Cut-off value for the maximum distance between two atoms in angstrom.
    delta_bin : float (optional)
        Bin size to descritize the function in angstrom.
    sigma : float (optional)
        Smearing parameter for the Gaussian function.
    distinguish_kinds: bool (optional)
        Whether different kinds should be distinguished e.g. Ni0 and Ni1 would be considered as
        different elements if ``True``.

    Returns
    -------
    total_order_p : float
        Order parameter of the structure.
    atomic_fingerprints : list
        List of order parameters for each atomic site.
    """

    def _calc_order_p(delta_bin, fprint, weights, cell_v, n_atoms):
        order_p = 0.0
        for el_pair in fprint["fingerprints"].keys():
            prefactor = weights[el_pair] * delta_bin * (cell_v / n_atoms) ** (1.0 / 3.0)
            order_p += prefactor * np.linalg.norm(np.array(fprint["fingerprints"][el_pair])) ** 2
        return order_p

    fprints = structure.calc_ffingerprint(
        r_max=r_max, delta_bin=delta_bin, sigma=sigma, distinguish_kinds=distinguish_kinds
    )

    element_dict = structure._element_dict
    if distinguish_kinds:
        element_dict = structure._kind_dict
    cell_v = structure["cell_volume"]
    weights = _calculate_weights(element_dict)
    n_atoms = sum(len(sites) for sites in element_dict.values())
    el_order_p = _calc_order_p(delta_bin, fprints[0], weights, cell_v, n_atoms)
    site_order_p = [
        _calc_order_p(delta_bin, fprint, weights, cell_v, n_atoms) for fprint in fprints[1]
    ]
    return {"order_p": el_order_p, "site_order_p": site_order_p}
