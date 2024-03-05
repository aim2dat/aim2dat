"""Module to retrieve physical and chemical properties of elements."""

# Standard library imports
import os

# Third party library imports
from ase.data import (
    chemical_symbols,
    atomic_masses,
    atomic_numbers,
    atomic_names,
    covalent_radii,
    vdw_radii,
)

# Internal libraray imports
from aim2dat.io.yaml import load_yaml_file


_groups_data = dict(load_yaml_file(os.path.dirname(__file__) + "/data_files/element_groups.yaml"))


element_groups = []
for groups in _groups_data.values():
    element_groups += groups
element_groups = set(element_groups)


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
    # else:
    #     raise TypeError(f"Element {element} needs to have the type str or int.")
    return el_number, el_symbol, el_name


def get_atomic_radius(element, radius_type="covalent"):
    """
    Return the covalent or van der Waals radius of the element (imported from ase).

    Parameters
    ----------
    element : str or int
        Atomic number, name or symbol of the element.
    radius_type : str (optional)
        Radius type. Valid options are 'covalent' or 'vdw'.

    Returns
    -------
    radius : float
        Atomic radius of the element.
    """
    el_number, _, _ = _check_element(element)

    if radius_type == "covalent":
        radius = covalent_radii[el_number]
    elif radius_type == "vdw":
        radius = vdw_radii[el_number]
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
    electronegativity : float or None
        Electronegativity of the element.
    """
    _, element, _ = _check_element(element)
    file_path = os.path.dirname(__file__) + "/data_files/electronegativity.yaml"
    en_dict = load_yaml_file(file_path)

    if scale in en_dict:
        electronegativity = en_dict[scale][element]
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
    element_number : int
        Atomic number of the element.
    """
    element_number, _, _ = _check_element(element)
    return atomic_masses[element_number]


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
    return set(_groups_data[element])


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
    for el, groups in _groups_data.items():
        if group_label in groups:
            elements.append(el)
    return set(elements)
