"""Module to read hdf5 files."""

# Standard library imports
from typing import TYPE_CHECKING, List

# Internal library imports
from aim2dat.io.utils import read_structure
from aim2dat.ext_interfaces import _return_ext_interface_modules

if TYPE_CHECKING:
    from aim2dat.strct.structure import Structure


@read_structure(r".*\.h(df)?5")
def read_hdf5_structure(file_path: str) -> List[dict]:
    """
    Read hdf5 structure file.

    Parameters
    ----------
    file_path : str
        Path to hdf5 file.

    Returns
    -------
    list
        List of dictionaries, each representing a molecular/crystalline structure.
    """
    backend_module = _return_ext_interface_modules("hdf5")
    return backend_module._import_from_hdf5_file(file_path)


def write_hdf5_structure(file_path: str, structures: List["Structure"]):
    """
    Write a list of structures to file.

    Parameters
    ----------
    file_path : str
        Path to hdf5 file.
    structures : list
        List of structures.
    """
    backend_module = _return_ext_interface_modules("hdf5")
    backend_module._store_in_hdf5_file(file_path, structures)
