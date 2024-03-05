"""
Module of functions to read output-files of FHI-aims.
"""

# Standard library imports
import os
import re


def _extract_file_names(files, pattern):
    # Need to include spin-polarized case...
    # Use auxiliary functions...
    found_files = []
    file_numbers = []
    for file in files:
        file_nr = pattern.findall(file)
        if file_nr:
            found_files.append(file)
            file_numbers.append(file_nr[0])
    if len(found_files) > 0 and len(file_numbers) == len(found_files):
        found_files, file_numbers = zip(*sorted(zip(found_files, file_numbers)))
    return found_files, file_numbers


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
    if len(folder_path) > 0:
        folder_path += "/"

    # Define file-name patterns:
    pattern = re.compile(r"^band(\d*)?\.out$")
    pattern_no_soc = re.compile(r"^band(\d*)?\.out\.no_soc$")

    # Check the files in the folder:
    files = [
        file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))
    ]

    if soc:
        if "band1001.out.no_soc" not in files:
            raise ValueError(
                "Spin-orbit coupling activated but the files don't have "
                "the right naming scheme."
            )

        band_structure_files, _ = _extract_file_names(files, pattern)
    else:
        if "band1001.out.no_soc" in files:
            band_structure_files, _ = _extract_file_names(files, pattern_no_soc)
        else:
            band_structure_files, _ = _extract_file_names(files, pattern)

    if len(band_structure_files) == 0:
        raise ValueError("No band structure files found.")

    occupations = []
    bands = []
    kpoints = []
    for file_idx, file in enumerate(band_structure_files):
        bandfile = open(folder_path + file, "r")

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
        bandfile.close()
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


def read_atom_proj_density_of_states(folder_path, soc=False, load_raw=False):
    """
    Read the atom projected density of states from FHI-aims.

    Parameters
    ----------
    folder_path : str
        Path to the folder of the pdos files.
    soc : bool (optional)
        Whether spin-orbit coupling is activated. The default value is ``False``.
    load_raw : bool (optional)
        Load files with appendix 'raw'. The default value is ``False``.


    Returns
    -------
    pdos : dict
        Dictionary containing the projected density of states for each atom.
    """
    dict_labels = ["s", "p", "d", "f"]
    energy = []
    atomic_pdos = []

    if len(folder_path) > 0:
        folder_path += "/"

    # Define file-name patterns:
    if load_raw:
        pattern = re.compile(r"^atom_proj_dos_([a-zA-Z]*\d*)?_raw\.dat$")
        pattern_no_soc = re.compile(r"^atom_proj_dos_([a-zA-Z]*\d*)?_raw\.dat\.no_soc$")
    else:
        pattern = re.compile(r"^atom_proj[a-z]*_dos_([a-zA-Z]*\d*)?\.dat$")
        pattern_no_soc = re.compile(r"^atom_proj[a-z]*_dos_([a-zA-Z]*\d*)?\.dat.no_soc$")

    # Check the files in the folder:
    files = [
        file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))
    ]

    if soc:
        if all([file.endswith(".dat") for file in files]):
            raise ValueError(
                "Spin-orbit coupling activated but the files don't have "
                "the right naming scheme."
            )

        pdos_files, element_nrs = _extract_file_names(files, pattern)
    else:
        if any([file.endswith(".no_soc") for file in files]):
            pdos_files, element_nrs = _extract_file_names(files, pattern_no_soc)
        else:
            pdos_files, element_nrs = _extract_file_names(files, pattern)

    if len(pdos_files) == 0:
        raise ValueError("No pDOS files found.")
    print(pdos_files)

    # Iterate over files and quantum numbers:
    elements = [re.split(r"(\d+)", el_nr)[0] for el_nr in element_nrs]
    for el_idx, (element, file_name) in enumerate(zip(elements, pdos_files)):
        pdos0 = {"element": element}
        with open(folder_path + file_name, "r") as pdos_file:
            for line in pdos_file:
                if line.split()[0] != "#" and len(line.strip()) != 0:
                    if el_idx == 0:
                        energy.append(float(line.split()[0]))
                    for value_idx, value in enumerate(line.split()[2:]):
                        if dict_labels[value_idx] in pdos0:
                            pdos0[dict_labels[value_idx]].append(float(value))
                        else:
                            pdos0[dict_labels[value_idx]] = [float(value)]
        atomic_pdos.append(pdos0)
    return {"energy": energy, "pdos": atomic_pdos, "unit_x": "eV"}
