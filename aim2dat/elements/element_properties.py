"""Module to retrieve physical and chemical properties of elements."""

# Internal libraray imports
from aim2dat.elements.data import symbols, names, element_groups, val_electrons
import aim2dat.elements.atomic_radii as atomic_radii
import aim2dat.elements.electronegativity as electronegativity
from aim2dat.elements.atomic_masses import atomic_masses


def _check_element(element):
    """Check if the input element can be processed."""
    if isinstance(element, str):
        element = element.capitalize()
        if element in names:
            el_number = names.index(element) + 1
            el_symbol = symbols[el_number - 1]
            el_name = element
        elif element in symbols:
            el_number = symbols.index(element) + 1
            el_symbol = element
            el_name = names[el_number - 1]
        else:
            raise ValueError(f"Element '{element}' could not be found.")
    else:
        try:
            el_number = int(element)
        except TypeError:
            raise TypeError(f"Element '{element}' needs to have the type str or int.")
        el_symbol = symbols[el_number - 1]
        el_name = names[el_number - 1]
    return el_number, el_symbol, el_name


def get_atomic_radius(element, radius_type="covalent"):
    """
    Return the covalent or van der Waals radius of the element. The following sources are
    used for different radius types:

    * ``'covalent'`` are from :doi:`10.1039/B801115J`.
    * ``'vdw'`` are from :doi:`10.1039/C3DT50599E`.
    * ``'chen_manz'`` are from :doi:`10.1039/C9RA07327B`.
    * ``'vdw_charry_tkatchenko'`` are from :doi:`10.26434/chemrxiv-2024-m3rtp-v2`.

    Parameters
    ----------
    element : str or int
        Atomic number, name or symbol of the element.
    radius_type : str (optional)
        Radius type. Valid options are ``'covalent'``, ``'vdw'``, ``'chen_manz'``,
        or ``'vdw_charry_tkatchenko'``.

    Returns
    -------
    radius : float
        Atomic radius of the element.

    Raises
    ------
    ValueError
        If ``radius_type`` is not supported or has the wrong format.
    """
    if radius_type in dir(atomic_radii):
        return getattr(atomic_radii, radius_type)[_check_element(element)[1]]
    else:
        raise ValueError(f"Radius type '{radius_type}' not supported.")


def get_electronegativity(element, scale="pauling"):
    """
    Return the electronegativity of the element.

    Parameters
    ----------
    element : str or int
        Atomic number, name or symbol of the element.
    scale : str (optional)
        Electronegativity scale. Supported values are ``'pauling'`` and ``'allen'``.

    Returns
    -------
    float or None
        Electronegativity of the element.
    """
    if scale in dir(electronegativity):
        return getattr(electronegativity, scale)[_check_element(element)[1]]
    else:
        raise ValueError(f"Scale '{scale}' not supported.")


def get_atomic_number(element):
    """
    Return atomic number of the element from element symbol or name.

    Parameters
    ----------
    element : str or int
        Atomic number, name or symbol of the element.

    Returns
    -------
    int
        Atomic number of the element.
    """
    return _check_element(element)[0]


def get_element_symbol(element):
    """
    Return symbol of the element from element number or name.

    Parameters
    ----------
    element : str or int
        Atomic number, name or symbol of the element.

    Returns
    -------
    str
        Symbol of the element.
    """
    return _check_element(element)[1]


def get_atomic_mass(element):
    """
    Return atomic mass of the element from the atomic number, element symbol or name.

    Parameters
    ----------
    element : str or int
        Atomic number, name or symbol of the element.

    Returns
    -------
    int
        Atomic mass of the element.
    """
    return atomic_masses[_check_element(element)[1]]


def get_val_electrons(element):
    """
    Return number of valence electrons of the element from the atomic number, element symbol
    or name.

    Parameters
    ----------
    element : str or int
        Atomic number, name or symbol of the element.

    Returns
    -------
    int
        Number of valence electrons of the element.
    """
    return val_electrons[_check_element(element)[1]]


def get_element_groups(element):
    """
    Return groups that contain the element from the atomic number, element symbol or name.

    Parameters
    ----------
    element : str or int
        Atomic number, name or symbol of the element.

    Returns
    -------
    groups : set
        Set of groups.
    """
    return set(element_groups[_check_element(element)[1]])


def get_group(group_label):
    """
    Return all elements in the group.

    Parameters
    ----------
    group_label : str
        Group label.

    Returns
    -------
    elements : set
        Set of element symbols..
    """
    elements = []
    for el, groups in element_groups.items():
        if group_label in groups:
            elements.append(el)
    return set(elements)
