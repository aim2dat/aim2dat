"""Module to retrieve physical and chemical properties of elements."""

# Third party library imports
from ase.data import (
    chemical_symbols,
    atomic_masses,
    atomic_numbers,
    atomic_names,
)

# Internal libraray imports
import aim2dat.utils.data as internal_data


def _check_element(element):
    """Check if the input element can be processed."""
    if isinstance(element, str):
        element = element.capitalize()
        if element in atomic_names:
            el_number = atomic_names.index(element)
            el_symbol = chemical_symbols[el_number]
            el_name = element
        elif element in chemical_symbols:
            el_number = atomic_numbers[element]
            el_symbol = element
            el_name = atomic_names[el_number]
        else:
            raise ValueError(f"Element '{element}' could not be found.")
    else:
        try:
            el_number = int(element)
        except TypeError:
            raise TypeError(f"Element '{element}' needs to have the type str or int.")
        el_symbol = chemical_symbols[el_number]
        el_name = atomic_names[el_number]
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
    el_number, element, _ = _check_element(element)
    if radius_type in dir(internal_data.atomic_radii):
        radius = getattr(internal_data.atomic_radii, radius_type)[element]
    else:
        raise ValueError(f"Radius type '{radius_type}' not supported.")
    return radius


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
    _, element, _ = _check_element(element)

    if scale in dir(internal_data.electronegativity):
        electronegativity = getattr(internal_data.electronegativity, scale)[element]
    else:
        raise ValueError(f"Scale '{scale}' not supported.")
    return electronegativity


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
    element_number, _, _ = _check_element(element)
    return element_number


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
    _, element_symbol, _ = _check_element(element)
    return element_symbol


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
    element_number, _, _ = _check_element(element)
    return atomic_masses[element_number]


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
    _, element, _ = _check_element(element)
    return internal_data.val_electrons[element]


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
    _, element, _ = _check_element(element)
    return set(internal_data.element_groups[element])


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
    for el, groups in internal_data.element_groups.items():
        if group_label in groups:
            elements.append(el)
    return set(elements)
