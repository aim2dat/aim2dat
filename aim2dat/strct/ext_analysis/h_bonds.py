"""Module implementing hydrogen analysis functions."""

# Internal library imports
from aim2dat.strct import Structure
from aim2dat.strct.ext_analysis.decorator import external_analysis_method
from aim2dat.elements import get_element_symbol


@external_analysis_method(attr_mapping=None)
def calc_hydrogen_bonds(
    structure: Structure,
    host_elements="O",
    bond_threshold=1.25,
    index_constraint=None,
    scheme: str = "baker_hubbard",
):
    """
    Search for hydrogen bonds.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure object.
    host_elements : list or str
        Elements considered as host atoms for hydrogen atoms and hydrogen bonds.
    bond_threshold : float
        Upper threshold to consider the hydrogen atom chemically bonded to the donor.
    index_constraint : list or None
        List of site indices to constrain the search.
    scheme : str
        The applied scheme. Supported options are `'baker_hubbard'`
        (:doi:`10.1016/0079-6107(84)90007-5`).

    Returns
    -------
    tuple
        Tuple of triples: site of the hydrogen acceptor, hydrogen site, and site of the hydrogen
        donor.
    """
    schemes = ["baker_hubbard"]
    if isinstance(host_elements, str):
        host_elements = [host_elements]
    host_elements = [get_element_symbol(el) for el in host_elements]
    if index_constraint is None:
        index_constraint = list(range(len(structure)))
    if scheme not in schemes:
        raise ValueError(f"`scheme` '{scheme}' is not supported. Valid options are: {schemes}.")

    host_indices = [
        idx
        for idx, el in enumerate(structure.elements)
        if el in host_elements and idx in index_constraint
    ]
    h_indices = [
        idx for idx, el in enumerate(structure.elements) if el == "H" and idx in index_constraint
    ]
    dists = structure.calc_distance(
        host_indices, h_indices, backfold_positions=True, use_supercell=False, return_pos=False
    )
    hbonds = []
    for host_idx in host_indices:
        for h_idx in h_indices:
            if dists[(host_idx, h_idx)] >= 2.5:
                continue

            for don_idx in host_indices:
                if dists[(don_idx, h_idx)] < bond_threshold and don_idx != host_idx:
                    if structure.calc_angle(h_idx, host_idx, don_idx) > 120.0:
                        hbonds.append((host_idx, h_idx, don_idx))
    return tuple(hbonds)
