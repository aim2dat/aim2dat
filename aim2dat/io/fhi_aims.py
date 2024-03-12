"""
Module of functions to read output-files of FHI-aims.
"""

# Standard library imports
import os
import re

from aim2dat.io.decorators import read_multiple


def _check_for_soc_files(soc, folder_path):
    no_soc_suffix = False
    if all(val is None for val in folder_path["soc"]):
        raise ValueError(
            "Spin-orbit coupling activated but the files don't have " +
            "the proper naming scheme."
        )
    if not soc and any(val == ".no_soc" for val in folder_path["soc"]):
        no_soc_suffix = True
    return no_soc_suffix



@read_multiple(r".*\.out(?P<soc>\.no_soc)?$")
def read_band_structure(folder_path, soc=False):
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
    no_soc_suffix = _check_for_soc_files(soc, folder_path)

    indices = [(val, idx) for idx, val in enumerate(folder_path["file_name"])]
    indices.sort(key=lambda point: point[0])
    _, indices = zip(*indices)
    occupations = []
    bands = []
    kpoints = []
    for idx in indices:
        if (folder_path["soc"][idx] is None and no_soc_suffix) or (folder_path["soc"][idx] is not None and not no_soc_suffix):
            continue

        with open(folder_path["file"][idx], "r") as bandfile:
            for line_idx, line in enumerate(bandfile):
                nrbands = int((len(line.split()) - 4) * 0.5)
                band0 = []
                occupation0 = []
                for band_idx in range(nrbands):
                    occupation0.append(float(line.split()[2 * band_idx + 4]))
                    band0.append(float(line.split()[2 * band_idx + 5]))
                kpoints.append([float(line.split()[idx]) for idx in range(1, 4)])
                bands.append(band0)
                occupations.append(occupation0)
    return {"kpoints": kpoints, "unit_y": "eV", "bands": bands, "occupations": occupations}


def read_total_density_of_states(file_name):
    """
    Read the total density of states from FHI-aims.

    Parameters
    ----------
    file_name : str
        Path of the output-file of FHI-aims containing the total density of states.

    Returns
    -------
    pdos : dict
        Dictionary containing the projected density of states for each atom.
    """
    energy = []
    tdos = []
    with open(file_name, "r") as tdos_file:
        for line in tdos_file:
            if not line.startswith("#"):
                energy.append(float(line.split()[0]))
                tdos.append(float(line.split()[1]))
    return {"energy": energy, "tdos": tdos, "unit_x": "eV"}


@read_multiple(r".*atom_proj[a-z]*_dos_(?P<kind>[a-zA-Z]+\d+)(?P<raw>_raw)?\.dat(?P<soc>\.no_soc)?$")
def read_atom_proj_density_of_states(folder_path, soc=False, load_raw=False):
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
    no_soc_suffix = _check_for_soc_files(soc, folder_path)

    # Iterate over files and quantum numbers:
    dict_labels = ["s", "p", "d", "f"]
    atomic_pdos = []
    energy = []
    indices = [(val, idx) for idx, val in enumerate(folder_path["file_name"])]
    indices.sort(key=lambda point: point[0])
    _, indices = zip(*indices)
    for idx in indices:
        if (folder_path["raw"][idx] is None and  load_raw) or (folder_path["raw"][idx] is not None and not load_raw):
            continue
        if (folder_path["soc"][idx] is None and no_soc_suffix) or (folder_path["soc"][idx] is not None and not no_soc_suffix):
            continue

        pdos0 = {"element": re.split(r"(\d+)", folder_path["kind"][idx])[0]}
        energy = []
        with open(folder_path["file"][idx], "r") as pdos_file: #TODO change to own manager:
            for line in pdos_file:
                if line.split()[0] != "#" and len(line.strip()) != 0:
                    energy.append(float(line.split()[0]))
                    for value_idx, value in enumerate(line.split()[2:]):
                        if dict_labels[value_idx] in pdos0:
                            pdos0[dict_labels[value_idx]].append(float(value))
                        else:
                            pdos0[dict_labels[value_idx]] = [float(value)]
        atomic_pdos.append(pdos0)
    return {"energy": energy, "pdos": atomic_pdos, "unit_x": "eV"}
