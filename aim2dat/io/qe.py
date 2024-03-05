"""
Module of functions to read output-files of Quantum ESPRESSO.
"""

# Standard library imports
import os
import re

# Internal library imports
from aim2dat.io.auxiliary_functions import _extract_file_names
from aim2dat.utils.units import length


def read_input_structure(file_name):
    """
    Read structure from the Quantum ESPRESSO input file.
    ibrav parameters are not yet fully supported.

    Parameters
    ----------
    file_name : str
        Path of the input-file of Quantum ESPRESSO containing structural data.

    Returns
    -------
    dict
        Dictionary containing the structural information.
    """

    def read_system_namelist(file_content, line_idx):
        patterns = {
            "ibrav": (re.compile(r"^\s*ibrav\s*=\s*(\d)?.*$"), int),
            "A": (re.compile(r"^\s*A\s*=\s*([0-9]*\.?[0-9]*)?.*$"), float),
            "B": (re.compile(r"^\s*B\s*=\s*([0-9]*\.?[0-9]*)?.*$"), float),
            "C": (re.compile(r"^\s*C\s*=\s*([0-9]*\.?[0-9]*)?.*$"), float),
        }
        celldm_pattern = re.compile(r"^\s*celldm\((\d)?\)\s*=\s*([-+]?[0-9]*\.?[0-9]*)?.*$")

        namelist_finished = False
        unit_cell = None
        qe_cell_p = {}
        while not namelist_finished:
            if file_content[line_idx].startswith("!") or file_content[line_idx].startswith("#"):
                line_idx += 1
                continue
            for label, (pattern, match_type) in patterns.items():
                if label not in qe_cell_p:
                    found_match = pattern.match(file_content[line_idx])
                    if found_match is not None:
                        qe_cell_p[label] = match_type(found_match.groups()[0])
            celldm_match = celldm_pattern.match(file_content[line_idx])
            if celldm_match is not None:
                qe_cell_p[int(celldm_match.groups()[0])] = float(celldm_match.groups()[1])
            if "/" in file_content[line_idx]:
                namelist_finished = True
            line_idx += 1
        # TO-DO: implement more ibrav-parameters:
        if qe_cell_p["ibrav"] == 8:
            if 1 in qe_cell_p and 2 in qe_cell_p and 3 in qe_cell_p:
                unit_cell = [
                    [qe_cell_p[1], 0.0, 0.0],
                    [0.0, qe_cell_p[1] * qe_cell_p[2], 0.0],
                    [0.0, 0.0, qe_cell_p[1] * qe_cell_p[3]],
                ]
            elif "A" in qe_cell_p and "B" in qe_cell_p and "C" in qe_cell_p:
                unit_cell = [
                    [qe_cell_p["A"], 0.0, 0.0],
                    [0.0, qe_cell_p["B"], 0.0],
                    [0.0, 0.0, qe_cell_p["C"]],
                ]
            else:
                raise ValueError(f"Could not retrieve unit cell from ibrav {qe_cell_p['ibrav']}.")
        elif qe_cell_p["ibrav"] in (
            1,
            2,
            3,
            -3,
            4,
            5,
            -5,
            6,
            7,
            8,
            9,
            -9,
            91,
            10,
            11,
            12,
            -12,
            13,
            -13,
            14,
        ):
            raise ValueError(f"ibrav {qe_cell_p['ibrav']} not yet implemented...")
        else:
            unit_cell = qe_cell_p[1] if 1 in qe_cell_p else 1.0
        return line_idx, unit_cell

    def read_cell_parameters(file_content, line_idx, conv_factor):
        pattern = re.compile(
            r"^\s*([-+]?[0-9]*\.?[0-9]*)?\s*([-+]?[0-9]*\.?[0-9]*)?"
            r"\s*([-+]?[0-9]*\.?[0-9]*)?\s*$"
        )
        card_finished = False
        unit_cell = []
        while not card_finished:
            match = pattern.match(file_content[line_idx])
            if match is not None and any(match.groups()):
                unit_cell.append([float(m_pos) * conv_factor for m_pos in match.groups()])
            else:
                card_finished = True
            line_idx += 1
        return line_idx, unit_cell

    def read_atomic_positions(file_content, line_idx):
        pattern = re.compile(
            r"^\s*(\w+)?\s*([-+]?[0-9]*\.?[0-9]*)?\s*([-+]?"
            r"[0-9]*\.?[0-9]*)?\s*([-+]?[0-9]*\.?[0-9]*)?.*$"
        )
        card_finished = False
        conv_factor = 1.0
        is_cartesian = True
        positions = []
        elements = []
        if "bohr" in file_content[line_idx - 1].lower():
            conv_factor = length.Bohr
        elif "crystal" in file_content[line_idx - 1].lower():
            is_cartesian = False

        while not card_finished:
            match = pattern.match(file_content[line_idx])
            if match is not None and any(match.groups()):
                try:
                    positions.append([float(m_pos) * conv_factor for m_pos in match.groups()[1:]])
                    elements.append(match.groups()[0])
                except ValueError:
                    card_finished = True
            else:
                card_finished = True
            line_idx += 1
        return line_idx, elements, positions, is_cartesian

    struct_dict = {"pbc": [True, True, True]}
    with open(file_name, "r") as input_file:
        file_content = input_file.read().splitlines()
    line_idx = 0
    while line_idx < len(file_content):  # line in enumerate(file_content):
        if "&SYSTEM" in file_content[line_idx]:
            line_idx, struct_dict["cell"] = read_system_namelist(file_content, line_idx + 1)
        if "CELL_PARAMETERS" in file_content[line_idx] and isinstance(struct_dict["cell"], float):
            conv_factor = struct_dict["cell"]
            if len(file_content[line_idx].split()) > 1:
                if file_content[line_idx].split()[-1].lower() == "bohr":
                    conv_factor = length.Bohr
                elif file_content[line_idx].split()[-1].lower() == "alat":
                    conv_factor *= length.Bohr
            line_idx, struct_dict["cell"] = read_cell_parameters(
                file_content, line_idx + 1, conv_factor
            )
        if "ATOMIC_POSITIONS" in file_content[line_idx]:
            (
                line_idx,
                struct_dict["elements"],
                struct_dict["positions"],
                struct_dict["is_cartesian"],
            ) = read_atomic_positions(file_content, line_idx + 1)
        line_idx += 1
    return struct_dict


def read_band_structure(file_name):
    """
    Read band structure file from Quantum ESPRESSO.
    Spin-polarized calculations are not yet supported.

    Parameters
    ----------
    file_name : str
        Path of the output-file of Quantum ESPRESSO containing the band structure.

    Returns
    -------
    band_structure : dict
        Dictionary containing the k-path and th eigenvalues as well as the occupations.
    """
    kpoints = []
    bands = []
    with open(file_name, "r") as bands_file:
        nr_bands = 0
        nr_kpoints = 0
        current_bands = []
        parse_kpoint = True
        for line in bands_file:
            # Catch the number of bands and k-points at the beginning of the file:
            if line.startswith(" &plot"):
                nr_bands = int(line.split()[2][:-1])
                nr_kpoints = int(line.split()[4])
                print("nr of bands:", nr_bands)
                print("nr of kpoints:", nr_kpoints)
                parse_kpoint = True
            # Parse k-point:
            elif parse_kpoint:
                kpoints.append(
                    [float(line.split()[0]), float(line.split()[1]), float(line.split()[2])]
                )
                parse_kpoint = False
            else:
                current_bands += [float(eigenvalue) for eigenvalue in line.split()]
                if len(current_bands) == nr_bands:
                    parse_kpoint = True
                    bands.append(current_bands)
                    current_bands = []
    return {"kpoints": kpoints, "unit_y": "eV", "bands": bands}


def read_total_density_of_states(file_name):
    """
    Read the total density of states from Quantum ESPRESSO.

    Parameters
    ----------
    file_name : str
        Path of the output-file of Quantum ESPRESSO containing the total density of states.

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


def read_atom_proj_density_of_states(folder_path):
    """
    Read the projected density of states from Quantum ESPRESSO.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the pdos ouput-files.

    Returns
    -------
    pdos : dict
        Dictionary containing the projected density of states for each atom.
    """
    quantum_numbers = {
        "s": ("s"),
        "p": ("px", "py", "pz"),
        "d": ("d-2", "d-1", "d0", "d+1", "d+2"),
        "f": (),  # magnetic qn need to be added here
    }

    energy = []
    atomic_pdos = []
    kind_indices = {}

    if len(folder_path) > 0:
        folder_path += "/"

    pattern = re.compile(
        r"^[a-zA-Z0-9\.\_]*pdos_atm#(\d*)?\(([a-zA-Z]*)?\)\_wfc\#(\d*)?\(([a-z])?\)$"
    )
    # Check the files in the folder:
    files = [
        file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))
    ]
    pdos_files, file_info = _extract_file_names(files, pattern)
    if len(pdos_files) == 0:
        raise ValueError("No pDOS files found.")

    for file_name, f_info in sorted(zip(pdos_files, file_info), key=lambda x: int(x[1][0])):
        # Check which kind the pdos belongs to:
        if f_info[0] not in kind_indices:
            kind_indices[f_info[0]] = len(atomic_pdos)
            atomic_pdos.append({"kind": f_info[1] + "_" + f_info[0]})

        # Read pdos, we only read the orbital contributions here, the summation is performed in
        # the plotting-class:
        with open(folder_path + file_name, "r") as pdos_file:
            # The energy is only parsed from the first file, we assume the same energy range:
            parse_energy = False
            if len(energy) == 0:
                parse_energy = True

            qn_labels = quantum_numbers[f_info[3]]

            # Create empty lists for each quantum number:
            for qn_label in qn_labels:
                atomic_pdos[kind_indices[f_info[0]]][f_info[2] + "_" + qn_label] = []

            # Iterate over lines and fill the list:
            for line in pdos_file:
                if not line.startswith("#"):
                    if parse_energy:
                        energy.append(float(line.split()[0]))
                    for qn_idx in range(len(qn_labels)):
                        qn_label = f_info[2] + "_" + qn_labels[qn_idx]

                        # Fix bug in output when exponential is too small, e.g.: 0.292-105 instead
                        # of 0.292E-105
                        float_number = 0.0
                        try:
                            float_number = float(line.split()[2 + qn_idx])
                        except ValueError:
                            pass
                        atomic_pdos[kind_indices[f_info[0]]][qn_label].append(float_number)

    return {"energy": energy, "pdos": atomic_pdos, "unit_x": "eV"}
