"""Module that implements functions to change molecular or crystalline structures."""

# Internal library imports
from aim2dat.strct.structure import Structure
from aim2dat.strct.ext_manipulation.add_structure import add_structure_coord


def add_functional_group(
    structure: Structure,
    wrap: bool = False,
    host_index: int = 0,
    functional_group: str = "CH3",
    bond_length: float = 1.25,
    r_max: float = 15.0,
    cn_method: str = "minimum_distance",
    min_dist_delta: float = 0.1,
    n_nearest_neighbours: int = 5,
    econ_tolerance: float = 0.5,
    econ_conv_threshold: float = 0.001,
    voronoi_weight_type: float = "rel_solid_angle",
    voronoi_weight_threshold: float = 0.5,
    okeeffe_weight_threshold: float = 0.5,
    change_label: bool = False,
) -> Structure:
    """
    Add a functional group or an atom to a host site.

    Notes
    -----
        This function is depreciated and will be removed soon. Please use
        ``aim2dat.strct.ext_manipulation.add_structure_coord`` instead.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure to which the guest structure is added.
    wrap : bool (optional)
            Wrap atomic positions back into the unit cell.
    host_index : int
        Index of the host site.
    functional_group : str
        Functional group or element symbol of the added functional group or atom.
    bond_length : float
        Bond length between the host atom and the base atom of the functional group.
    r_max : float (optional)
        Cut-off value for the maximum distance between two atoms in angstrom.
    cn_method : str (optional)
        Method used to calculate the coordination environment.
    min_dist_delta : float (optional)
        Tolerance parameter that defines the relative distance from the nearest neighbour atom
        for the ``'minimum_distance'`` method.
    n_nearest_neighbours : int (optional)
        Number of neighbours that are considered coordinated for the ``'n_neighbours'``
        method.
    econ_tolerance : float (optional)
        Tolerance parameter for the econ method.
    econ_conv_threshold : float (optional)
        Convergence threshold for the econ method.
    voronoi_weight_type : str (optional)
        Weight type of the Voronoi facets. Supported options are ``'covalent_atomic_radius'``,
        ``'area'`` and ``'solid_angle'``. The prefix ``'rel_'`` specifies that the relative
        weights with respect to the maximum value of the polyhedron are calculated.
    voronoi_weight_threshold : float (optional)
        Weight threshold to consider a neighbouring atom coordinated.
    change_label : bool (optional)
        Add suffix to the label of the new structure highlighting the performed manipulation.

    Returns
    -------
    aim2dat.strct.Structure
        Structure with attached functional group.
    """
    from warnings import warn

    warn(
        "This function will be removed soon, please use "
        + "`strct.ext_manipulation.add_structure_coord` instead.",
        DeprecationWarning,
        2,
    )

    return add_structure_coord(
        structure=structure,
        wrap=wrap,
        host_indices=host_index,
        guest_indices=0,
        guest_structure=functional_group,
        bond_length=bond_length,
        r_max=r_max,
        cn_method=cn_method,
        min_dist_delta=min_dist_delta,
        n_nearest_neighbours=n_nearest_neighbours,
        econ_tolerance=econ_tolerance,
        econ_conv_threshold=econ_conv_threshold,
        voronoi_weight_type=voronoi_weight_type,
        voronoi_weight_threshold=voronoi_weight_threshold,
        okeeffe_weight_threshold=okeeffe_weight_threshold,
        change_label=change_label,
    )
