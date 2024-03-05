"""
Module of functions to read output-files of CP2K.
"""

# Standard library imports
import os
import re

# Internal library imports
from aim2dat.io.auxiliary_functions import _extract_file_names
from aim2dat.aiida_workflows.cp2k.parser_utils import RestartStructureParser, PDOSParser


def read_band_structure(file_name):
    """
    Read band structure file from CP2K.

    Parameters
    ----------
    file_name : str
        Path of the output-file of CP2K containing the band structure.

    Returns
    -------
    band_structure : dict
        Dictionary containing the k-path and the eigenvalues as well as the occupations.
    """
    kpoints = []
    bands = [[], []]
    occupations = [[], []]
    path_labels = []
    point_idx = -1
    spin_idx = 0
    special_p2 = None
    is_spin_pol = False
    with open(file_name, "r") as bands_file:
        for line in bands_file:
            l_splitted = line.split()
            if line.startswith("#  Special point 1"):
                if special_p2 is not None:
                    path_labels.append((point_idx, special_p2))
                if l_splitted[-1] != "specifi":
                    path_labels.append((point_idx + 1, l_splitted[-1]))
            elif line.startswith("#  Special point 2") and l_splitted[-1] != "specifi":
                special_p2 = l_splitted[-1]
            elif line.startswith("#  Point"):  # and "Spin 1" in line:
                if "Spin 1" in line:
                    spin_idx = 0
                    point_idx += 1
                    for idx in range(2):
                        bands[idx].append([])
                        occupations[idx].append([])
                    kpoints.append([float(l_splitted[idx]) for idx in range(5, 8)])
                else:
                    spin_idx = 1
                    is_spin_pol = True
            elif not line.startswith("#"):
                bands[spin_idx][point_idx].append(float(l_splitted[1]))
                occupations[spin_idx][point_idx].append(float(l_splitted[2]))
        if special_p2 is not None:
            path_labels.append((point_idx, special_p2))
    if not is_spin_pol:
        bands = bands[0]
        occupations = occupations[0]
    return {
        "kpoints": kpoints,
        "unit_y": "eV",
        "bands": bands,
        "occupations": occupations,
        "path_labels": path_labels,
    }


def read_atom_proj_density_of_states(folder_path):
    """
    Read the atom projected density of states from CP2K.

    Parameters
    ----------
    folder_path : str
        Path to the folder of the pdos files.

    Returns
    -------
    pdos : dict
        Dictionary containing the projected density of states for each kind.
    """
    if len(folder_path) > 0:
        folder_path += "/"

    pattern = re.compile(r"^[a-zA-Z0-9\.\_]*-([A-Z]*)?_*k\d+-\d+\.pdos$")
    files = [
        file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))
    ]
    pdos_files, file_info = _extract_file_names(files, pattern)
    if len(pdos_files) == 0:
        raise ValueError("No pDOS files found.")

    parser = PDOSParser()
    for file_name, spin in zip(pdos_files, file_info):
        file_content = open(folder_path + file_name, "r").read()
        parser.parse_pdos(file_content, spin)
    return parser.pdos


def read_optimized_structure(folder_path):
    """
    Read optimized structures from 'restart'-files.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the CP2K ouput-files.

    Returns
    -------
    dict or list
        Dictionary containing the structural information. In case of a farming job a list of
        dictionaries is returned. In case several calculations have been run in the same folder a
        nested dictionary is returned.
    """
    pattern = re.compile(r"(^\S*)?-1\.restart$")
    if len(folder_path) > 0:
        folder_path += "/"
    files = [
        file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))
    ]
    restart_files, file_info = _extract_file_names(files, pattern)
    structures = {}
    for restart_file, f_info in zip(restart_files, file_info):
        restart_content = open(folder_path + restart_file, "r").read()
        str_parser = RestartStructureParser(restart_content)
        new_structures = str_parser.retrieve_output_structure()
        if len(new_structures) == 0:
            continue
        elif len(new_structures) == 1:
            structures[f_info] = new_structures[0]
        else:
            structures[f_info] = new_structures
    return list(structures.values())[0] if len(structures.values()) == 1 else structures
