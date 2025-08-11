"""
Functions to read band structure and pDOS files of CP2K.
"""

# Internal library imports
import aim2dat.units as units
import aim2dat.io.utils as io_utils


@io_utils.read_band_structure(r".*\.bs$")
def read_cp2k_band_structure(file_path: str) -> dict:
    """
    Read band structure file from CP2K.

    Parameters
    ----------
    file_path : str
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
    with io_utils.custom_open(file_path, "r") as bands_file:
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


@io_utils.read_multiple(
    r".*-(?P<spin>[A-Z]+)?_?(?:k\d|list\d).*\.pdos$", is_read_proj_dos_method=True
)
def read_cp2k_proj_dos(folder_path: str) -> dict:
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
    all_orbitals = [
        "s",
        "px",
        "py",
        "pz",
        "d-2",
        "d-1",
        "d0",
        "d+1",
        "d+2",
        "f-3",
        "f-2",
        "f-1",
        "f0",
        "f+1",
        "f+2",
        "f+3",
        "g-4",
        "g-3",
        "g-2",
        "g-1",
        "g0",
        "g+1",
        "g+2",
        "g+3",
        "g+4",
        "i-5",
        "i-4",
        "i-3",
        "i-2",
        "i-1",
        "i0",
        "i+1",
        "i+2",
        "i+3",
        "i+4",
        "i+5",
    ]

    indices = [(val, idx) for idx, val in enumerate(folder_path["file_name"])]
    indices.sort(key=lambda point: point[0])
    _, indices = zip(*indices)
    pdos = []
    efermi = None
    kinds = {}
    for idx in indices:
        spin_suffix = ""
        if folder_path["spin"][idx] is not None:
            spin_suffix = "_" + folder_path["spin"][idx].lower()
        energy = []
        occupation = []
        with io_utils.custom_open(folder_path["file_path"][idx], "r") as dos_file:
            line_1 = dos_file.readline().split()
            line_2 = dos_file.readline().split()
            if "list" in line_1:
                kind = line_1[5]
            else:
                kind = line_1[6]
            efermi = float(line_1[-2]) * units.energy.Hartree
            orbital_labels = line_2[5:]
            single_pdos = {orb + spin_suffix: [] for orb in all_orbitals if orb in orbital_labels}
            single_pdos["kind"] = kind
            for line in dos_file:
                line_sp = line.split()
                energy.append(float(line_sp[1]) * units.energy.Hartree)
                occupation.append(float(line_sp[2]))
                for orb, val in zip(orbital_labels, line_sp[3:]):
                    single_pdos[orb + spin_suffix].append(float(val))
        if kind in kinds:
            pdos[kinds[kind]].update(single_pdos)
        else:
            kinds[kind] = len(pdos)
            pdos.append(single_pdos)
    return {
        "energy": energy,
        "occupation": occupation,
        "pdos": pdos,
        "unit_x": "eV",
        "e_fermi": efermi,
    }


def read_band_structure(file_path: str) -> dict:
    """
    Read band structure file from CP2K.

    Notes
    -----
        This function is deprecated and will be removed, please use
        `aim2dat.io.read_cp2k_band_structure` instead.

    Parameters
    ----------
    file_path : str
        Path of the output-file of CP2K containing the band structure.

    Returns
    -------
    band_structure : dict
        Dictionary containing the k-path and the eigenvalues as well as the occupations.
    """
    from warnings import warn

    warn(
        "This function will be removed, please use `aim2dat.io.read_cp2k_band_structure` instead.",
        DeprecationWarning,
        2,
    )
    return read_cp2k_band_structure(file_path=file_path)


def read_atom_proj_density_of_states(folder_path: str) -> dict:
    """
    Read the atom projected density of states from CP2K.

    Notes
    -----
        This function is deprecated and will be removed, please use `aim2dat.io.read_cp2k_proj_dos`
        instead.

    Parameters
    ----------
    folder_path : str
        Path to the folder of the pdos files.

    Returns
    -------
    pdos : dict
        Dictionary containing the projected density of states for each kind.
    """
    from warnings import warn

    warn(
        "This function will be removed, please use `aim2dat.io.read_cp2k_proj_dos` instead.",
        DeprecationWarning,
        2,
    )
    return read_cp2k_proj_dos(folder_path=folder_path)
