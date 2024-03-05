"""Space group analysis module."""

# Internal library imports
from aim2dat.ext_interfaces import _return_ext_interface_modules


def determine_space_group(*args, **kwargs):
    """Determine space group."""
    return _return_ext_interface_modules("spglib")._space_group_analysis(*args, **kwargs)
