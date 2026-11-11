"""
Utils for the EnumlibCalcjob.
"""

# Standard library imports
import re

# Internal library imports
from aim2dat.elements.element_properties import _check_element


def check_kinds(kind_names):
    """Check whether all kinds can be processed."""
    kinds_failed = []

    for kind in kind_names:
        element = re.sub(r"[0-9]", "", kind)
        try:
            _check_element(element)
        except (ValueError, TypeError):
            kinds_failed.append(kind)

    return kinds_failed


def get_kindnames(structure, to_enumerate):
    """
    Get kind names from structure.
    """
    if isinstance(to_enumerate, list):
        kind_names = list({x for l0 in to_enumerate for x in l0})

    elif isinstance(to_enumerate, dict):
        kind_names = list({x for l0 in to_enumerate.values() for x in l0})
        kind_names += list(
            {
                el
                for el in structure.get_site_kindnames()
                if el not in kind_names and el not in to_enumerate
            }
        )

    return kind_names


# More methods may be added when new features are integrated in the Calcjob
