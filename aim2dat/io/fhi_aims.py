"""
Module of functions to read/write files of the FHI-aims code.
"""

# Standard library imports
from typing import TYPE_CHECKING, Union

# Internal library imports
from aim2dat.io.utils import (
    read_structure,
    read_total_dos,
    read_multiple,
    write_structure,
    custom_open,
)
from aim2dat.elements import get_element_symbol

if TYPE_CHECKING:
    from aim2dat.strct import Structure, StructureCollection


def _check_for_soc_files(folder_path, soc):
    no_soc_suffix = False
    if soc and all(val is None for val in folder_path["soc"]):
        raise ValueError(
            "Spin-orbit coupling activated but the files don't have " + "the proper naming scheme."
        )
    if not soc and any(val == ".no_soc" for val in folder_path["soc"]):
        no_soc_suffix = True
    return no_soc_suffix


@read_structure(r".*\.in(\.next_step)?", preset_kwargs=None)
def read_fhiaims_geometry_file(file_path: str) -> dict:
    """
    Read geometry file.

    Parameters
    ----------
    file_path : str
        File path or file content.

    Returns
    -------
    dict
        Dictionary containing the structural information.
    """
    cell = []
    kinds = []
    elements = []
    positions = []
    is_cartesian = ""
    with custom_open(file_path, "r") as f_obj:
        for line in f_obj:
            line = line.strip()
            if line.startswith("lattice_vector"):
                cell.append([float(val) for val in line.split()[1:4]])
            elif line.startswith("atom"):
                line_sp = line.split()
                if line_sp[0] == "atom_frac":
                    if is_cartesian is True:
                        raise ValueError(
                            "Cannot parse geometry file with mixed cartesian and scaled positions."
                        )
                    is_cartesian = False
                elif is_cartesian is False:
                    raise ValueError(
                        "Cannot parse geometry file with mixed cartesian and scaled positions."
                    )
                else:
                    is_cartesian = True
                kinds.append(line_sp[4])
                try:
                    elements.append(get_element_symbol(line_sp[4]))
                except ValueError:
                    elements.append(
                        get_element_symbol(
                            "".join(i for i in line_sp[4].split("_")[0] if not i.isdigit())
                        )
                    )
                positions.append([float(val) for val in line.split()[1:4]])
    structure = {
        "elements": elements,
        "kinds": kinds,
        "positions": positions,
        "pbc": False,
        "is_cartesian": is_cartesian,
    }
    if len(cell) > 0:
        structure["cell"] = cell
        structure["pbc"] = True
    return structure


@write_structure(r".*\.in(\.next_step)?", preset_kwargs=None)
def write_fhiaims_geometry_file(
    file_path: str,
    structure: Union["Structure", "StructureCollection", list],
    use_scaled_pos: bool = False,
):
    """
    Write FHI-aims geometry file.

    Parameters
    ----------
    file_path : str
        Path to geometry file.
    structure : aim2dat.strct.Structure, aim2dat.strct.StructureCollection, list
        Structure object, StructureCollection object or list of Structure objects. For the latter
        two cases the first item/structure is written to file.
    use_scaled_pos : bool
        Whether to write scaled atom positions using the ``'atom_frac'`` key.
    """
    if isinstance(structure, list) or type(structure).__name__ == "StructureCollection":
        structure = structure[0]

    with open(file_path, "w") as f_obj:
        if structure.cell is not None:
            for cell_v in structure.cell:
                f_obj.write(
                    f"lattice_vector {cell_v[0]:16.8f} {cell_v[1]:16.8f} {cell_v[2]:16.8f}\n"
                )
        for el, kind, pos in structure.iter_sites(
            get_kind=True, get_cart_pos=not use_scaled_pos, get_scaled_pos=use_scaled_pos
        ):
            species_name = el if kind is None else "_".join(kind.split(" "))
            frac_suffix = "_frac" if use_scaled_pos else ""
            f_obj.write(
                f"atom{frac_suffix} {pos[0]:16.8f} {pos[1]:16.8f} {pos[2]:16.8f} {species_name}\n"
            )


@read_multiple(r".*band.*\.out(?P<soc>\.no_soc)?$", is_read_band_strct_method=True)
def read_fhiaims_band_structure(folder_path: str, soc: bool = False) -> dict:
    """
    Read band structure files from FHI-aims.
    Spin-polarized calculations are not yet supported.

    Parameters
    ----------
    folder_path : str
        Path to the folder of the band structure files.
    soc : bool (optional)
        Whether spin-orbit coupling is activated. The default value is ``False``.

    Returns
    -------
    band_structure : dict
        Dictionary containing the k-path and th eigenvalues as well as the occupations.
    """
    no_soc_suffix = _check_for_soc_files(folder_path, soc)

    indices = [(val, idx) for idx, val in enumerate(folder_path["file_path"])]
    indices.sort(key=lambda point: point[0])
    _, indices = zip(*indices)
    occupations = []
    bands = []
    kpoints = []
    for idx in indices:
        if (folder_path["soc"][idx] is None and no_soc_suffix) or (
            folder_path["soc"][idx] is not None and not no_soc_suffix
        ):
            continue

        with custom_open(folder_path["file_path"][idx], "r") as bandfile:
            for line in bandfile:
                l_split = line.split()
                nrbands = int((len(l_split) - 4) * 0.5)
                kpoints.append([float(l_split[idx]) for idx in range(1, 4)])
                bands.append([float(l_split[2 * idx + 5]) for idx in range(nrbands)])
                occupations.append([float(l_split[2 * idx + 4]) for idx in range(nrbands)])
    return {"kpoints": kpoints, "unit_y": "eV", "bands": bands, "occupations": occupations}


@read_total_dos(r".*KS_DOS_total*\.dat(?P<soc>\.no_soc)?$")
def read_fhiaims_total_dos(file_path: str) -> dict:
    """
    Read the total density of states from FHI-aims.

    Parameters
    ----------
    file_path : str
        Path of the output-file of FHI-aims containing the total density of states.

    Returns
    -------
    pdos : dict
        Dictionary containing the projected density of states for each atom.
    """
    energy = []
    tdos = []
    with custom_open(file_path, "r") as tdos_file:
        for line in tdos_file:
            if not line.startswith("#"):
                energy.append(float(line.split()[0]))
                tdos.append(float(line.split()[1]))
    return {"energy": energy, "tdos": tdos, "unit_x": "eV"}


@read_multiple(
    r".*atom_proj[a-z]*_dos_(spin_(?P<spin>\S\S))?(?P<kind>[^0-9]+)"
    + r"(?P<site_index>\d+)(?P<raw>_raw)?\.dat(?P<soc>\.no_soc)?$",
    is_read_proj_dos_method=True,
)
def read_fhiaims_proj_dos(folder_path: str, soc: bool = False, load_raw: bool = False) -> dict:
    """
    Read the atom projected density of states from FHI-aims.

    Parameters
    ----------
    folder_path : str
        Path to the folder of the pdos files or list of pdos files or path to a pdos file.
    soc : bool (optional)
        Whether spin-orbit coupling is activated. The default value is ``False``.
    load_raw : bool (optional)
        Load files with appendix 'raw'. The default value is ``False``.

    Returns
    -------
    pdos : dict
        Dictionary containing the projected density of states for each atom.
    """
    no_soc_suffix = _check_for_soc_files(folder_path, soc)

    # Iterate over files and quantum numbers:
    dict_labels = ["s", "p", "d", "f", "g", "h", "i"]
    spin_labels = {"up": "_alpha", "dn": "_beta"}
    used_indices = {}
    atomic_pdos = []
    energy = []
    indices = [(val, idx) for idx, val in enumerate(folder_path["site_index"])]
    indices.sort(key=lambda point: point[0])
    _, indices = zip(*indices)
    for idx in indices:
        if (folder_path["raw"][idx] is None and load_raw) or (
            folder_path["raw"][idx] is not None and not load_raw
        ):
            continue
        if (folder_path["soc"][idx] is None and no_soc_suffix) or (
            folder_path["soc"][idx] is not None and not no_soc_suffix
        ):
            continue

        pdos0 = {}
        energy = []
        with custom_open(folder_path["file_path"][idx], "r") as pdos_file:
            spin_suffix = spin_labels.get(folder_path["spin"][idx], "")
            for line in pdos_file:
                if line.split()[0] != "#" and len(line.strip()) != 0:
                    energy.append(float(line.split()[0]))
                    for value_idx, value in enumerate(line.split()[2:]):
                        pdos0.setdefault(dict_labels[value_idx] + spin_suffix, []).append(
                            float(value)
                        )

        site_index = int(folder_path["site_index"][idx])
        if site_index in used_indices:
            atomic_pdos[used_indices[site_index]].update(pdos0)
        else:
            # In FHI-aims, we only have "species" which can refer to an element or to
            # a specific kind. Here, we try to account for that:
            kind = folder_path["kind"][idx]
            kind_sp = kind.split("_")
            try:
                el = get_element_symbol(kind_sp[0])
            except ValueError:
                el = None
            if el is None:
                pdos0["kind"] = kind
            elif el == kind:
                pdos0["element"] = kind
            else:
                pdos0["kind"] = kind
                pdos0["element"] = kind
            used_indices[site_index] = len(atomic_pdos)
            atomic_pdos.append(pdos0)
    return {"energy": energy, "pdos": atomic_pdos, "unit_x": "eV"}


@read_multiple(r"band.*\.out(?P<soc>\.no_soc)?$")
def read_band_structure(folder_path: str, soc: bool = False) -> dict:
    """
    Read band structure files from FHI-aims.
    Spin-polarized calculations are not yet supported.

    Notes
    -----
        This function is deprecated and will be removed, please use
        `aim2dat.io.read_fhiaims_band_structure` instead.

    Parameters
    ----------
    folder_path : str
        Path to the folder of the band structure files.
    soc : bool (optional)
        Whether spin-orbit coupling is activated. The default value is ``False``.

    Returns
    -------
    band_structure : dict
        Dictionary containing the k-path and th eigenvalues as well as the occupations.
    """
    from warnings import warn

    warn(
        "This function will be removed, please use "
        + "`aim2dat.io.read_fhiaims_band_structure` instead.",
        DeprecationWarning,
        2,
    )
    return read_fhiaims_band_structure(folder_path=folder_path, soc=soc)


def read_total_density_of_states(file_name):
    """
    Read the total density of states from FHI-aims.

    Notes
    -----
        This function is deprecated and will be removed, please use
        `aim2dat.io.read_fhiaims_total_dos` instead.

    Parameters
    ----------
    file_name : str
        Path of the output-file of FHI-aims containing the total density of states.

    Returns
    -------
    pdos : dict
        Dictionary containing the projected density of states for each atom.
    """
    from warnings import warn

    warn(
        "This function will be removed, please use `aim2dat.io.read_fhiaims_total_dos` instead.",
        DeprecationWarning,
        2,
    )
    return read_fhiaims_total_dos(file_path=file_name)


@read_multiple(
    r".*atom_proj[a-z]*_dos_(spin_(?P<spin>\S\S))?(?P<kind>[^0-9]+)"
    + r"(?P<site_index>\d+)(?P<raw>_raw)?\.dat(?P<soc>\.no_soc)?$",
)
def read_atom_proj_density_of_states(folder_path, soc=False, load_raw=False):
    """
    Read the atom projected density of states from FHI-aims.

    Notes
    -----
        This function is deprecated and will be removed, please use
        `aim2dat.io.read_fhiaims_proj_dos` instead.

    Parameters
    ----------
    folder_path : str
        Path to the folder of the pdos files or list of pdos files or path to a pdos file.
    soc : bool (optional)
        Whether spin-orbit coupling is activated. The default value is ``False``.
    load_raw : bool (optional)
        Load files with appendix 'raw'. The default value is ``False``.

    Returns
    -------
    pdos : dict
        Dictionary containing the projected density of states for each atom.
    """
    from warnings import warn

    warn(
        "This function will be removed, please use `aim2dat.io.read_fhiaims_proj_dos` instead.",
        DeprecationWarning,
        2,
    )
    return read_fhiaims_proj_dos(folder_path=folder_path, soc=soc, load_raw=load_raw)
