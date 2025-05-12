"""Handle optional external interfaces."""

# Standard library imports
import importlib.metadata as il_metadata
import importlib

# Third party library imports
from packaging import version


_ext_interfaces_deps = {
    "aiida": [("aiida-core", "2.0", "3.0")],
    "ase_atoms": [("ase", "3.22.0", None)],
    "dscribe": [("dscribe", "1.2.2", None)],
    "graphviz": [("graphviz", "0.19.1", None)],
    "hdf5": [("h5py", "3.7.0", None)],
    "mofxdb": [("mofdb_client", None, None)],
    "mp_openapi": [("requests", None, None), ("msgpack", "1.0.2", None), ("boto3", "1.25", None)],
    "oqmd": [("qmpy_rester", None, None)],
    "openmm": [("openmm", "8.2.0", None)],
    "optimade": [("requests", None, None)],
    "pandas": [("pandas", None, None)],
    "phonopy": [("phonopy", "2.17.1", None)],
    "pymatgen": [("pymatgen", "2022.02.03", None)],
    "pyxtal": [("pyxtal", None, None)],
    "spglib": [("spglib", "1.16.1", None)],
}


def _return_ext_interface_modules(ext_interfaces):
    """Import interface module(s) to optional dependencies."""
    if isinstance(ext_interfaces, str):
        ext_interfaces = [ext_interfaces]
    backend_modules = []
    for ext_int in ext_interfaces:
        _check_package_dependencies(_ext_interfaces_deps[ext_int])
        backend_modules.append(importlib.import_module("aim2dat.ext_interfaces." + ext_int))
    if len(backend_modules) == 1:
        return backend_modules[0]
    else:
        return backend_modules


def _check_package_dependencies(dependencies):
    """Check package dependencies."""
    for dep, l_vers_limit, u_vers_limit in dependencies:
        try:
            pkg_vers = il_metadata.version(dep)
        except il_metadata.PackageNotFoundError:
            raise ImportError(f"The library `{dep}` needs to be installed to use this function.")
        if l_vers_limit is not None and version.parse(pkg_vers) < version.parse(l_vers_limit):
            raise ImportError(
                f"The version of the library `{dep}` is incompatible "
                f"({pkg_vers} < {l_vers_limit})."
            )
        if u_vers_limit is not None and version.parse(pkg_vers) > version.parse(u_vers_limit):
            raise ImportError(
                f"The version of the library `{dep}` is incompatible: "
                f"{pkg_vers} > {l_vers_limit}."
            )
