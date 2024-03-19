"""
Functions to read band structure and pDOS files of CP2K.
"""

# Internal library imports
from aim2dat.io.cp2k.legacy_parser import PDOSParser
from aim2dat.io.utils import read_multiple, custom_open


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
    with custom_open(file_name, "r") as bands_file:
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


@read_multiple(r".*-(?P<spin>[A-Z]+)?_?k.*\.pdos$")
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
    # TODO order pdos better..
    indices = [(val, idx) for idx, val in enumerate(folder_path["file_name"])]
    indices.sort(key=lambda point: point[0])
    _, indices = zip(*indices)
    parser = PDOSParser()
    for idx in indices:  # file_p, spin in zip(folder_path["file"], folder_path["spin"]):
        file_content = custom_open(folder_path["file"][idx], "r").read()
        parser.parse_pdos(file_content, folder_path["spin"][idx])
    return parser.pdos
