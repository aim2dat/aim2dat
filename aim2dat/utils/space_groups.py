"""Module containing functions to parse space groups and to get information on the lattice."""

# TODO change to libspg

# Third party library imports
import ase.spacegroup as ase_sg


def transform_to_nr(space_group):
    """
    Parse the space group into the corresponding number (if necessary).

    Parameters
    ----------
    space_group : int or str
        Space group of the crystal.

    Returns
    -------
    sg_num : int
        Number of the space group.
    """
    if isinstance(space_group, int):
        sg_num = space_group
    elif isinstance(space_group, str):
        # This might make prolems with nomenclature of other space groups as well..
        # if space_group == "Cmca":
        #     space_group = "Cmce"
        # elif space_group == "Cmma":
        #     space_group = "Cmme"
        if space_group.startswith("C") and space_group.endswith("a"):
            space_group = space_group[:-1] + "e"
        elif space_group.startswith("A") and "b" in space_group:
            space_group = space_group.replace("b", "e")
        space_group = space_group.replace("_", "")
        space_group = ase_sg.Spacegroup(space_group)
        sg_num = space_group.no
    else:
        raise ValueError("`space_group` needs to be of type str or int.")
    return sg_num


def transform_to_str(space_group):
    """
    Parse the space group from its number to the symbol.

    Parameters
    ----------
    space_group : int or str
        Space group of the crystal.

    Returns
    -------
    sg_str : str
        Symbol of the space group.
    """
    if isinstance(space_group, int):
        sg_ase = ase_sg.Spacegroup(space_group)
        sg_str = sg_ase.symbol.replace(" ", "")
    else:
        sg_str = space_group
    return sg_str


def get_lattice_type(space_group):
    """
    Return the crystal system of a space group given by a string or number.
    """
    from warnings import warn

    warn(
        "This function will be removed, please use `get_crystal_system` instead.",
        DeprecationWarning,
        2,
    )
    return get_crystal_system(space_group)


def get_crystal_system(space_group):
    """
    Return the crystal system of a space group given by a string or number.

    Parameters
    ----------
    space_group : int or str
        Space group of the crystal.

    Returns
    -------
    crystal_system : str
        The crystal system of the space group, e.g. 'tetragonal'.
    """
    sg_num = transform_to_nr(space_group)

    bounds = {
        "triclinic": (0, 3),
        "monoclinic": (2, 16),
        "orthorhombic": (15, 75),
        "tetragonal": (74, 143),
        "trigonal": (142, 168),
        "hexagonal": (167, 195),
        "cubic": (194, 231),
    }
    crystal_system = ""

    for crystal_system0, bound in bounds.items():
        if bound[0] < sg_num < bound[1]:
            crystal_system = crystal_system0

    return crystal_system
