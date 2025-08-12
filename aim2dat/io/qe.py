"""
Module of functions to read output-files of Quantum ESPRESSO.
"""

# Standard library imports
import re
import xml.etree.ElementTree as ET
import copy

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.io.utils import (
    read_structure,
    read_total_dos,
    read_band_structure,
    read_multiple,
    custom_open,
)
from aim2dat.io.base_parser import transform_str_value
from aim2dat.units import length
from aim2dat.utils.dict_tools import dict_create_tree


@read_structure(r".*\.xml", preset_kwargs={"extract_structures": True})
def read_qe_xml(
    file_path: str,
    extract_structures: bool = False,
    strct_type: str = None,
    strct_include: list = None,
    raise_error: bool = True,
) -> dict:
    """
    Read xml output file.

    Parameters
    ----------
    file_path : str
        Path of the xml-file of Quantum ESPRESSO.
    extract_structures : bool
        Whether to extract alls crystal structures and add them to the output dictionary with the
        key ``'structures'``.
    strct_type : str
        Type of extracted structure(s). Supported options are ``'input'``, ``'output'`` or
        ``'steps'``.
    strct_include : list
        List of dictionary keys that are included in the structure attributes.
    raise_error : bool
        Whether to raise an error if a flaw is detected in the output file.

    Returns
    -------
    dict
        Output dictionary.
    """

    def _parse_atomic_structure(element):
        outp_dict = {
            "system": {
                "nat": int(element.attrib["nat"]),
                "alat": float(element.attrib["alat"]),
            }
        }
        if "bravais_index" in element.attrib:
            outp_dict["system"]["ibrav"] = int(element.attrib["bravais_index"])
        for c1 in element:
            if c1.tag == "atomic_positions":
                outp_dict["atomic_positions"] = [
                    [c.attrib["name"]] + [float(val) for val in c.text.split()] for c in c1
                ]
            elif c1.tag == "cell":
                outp_dict["cell"] = [[float(val) for val in c.text.split()] for c in c1]
        return outp_dict

    def _parse_kpoints(element):
        if element[0].tag == "nks":
            return {
                "nks": int(element[0].text),
                "k_points": [
                    [float(v) for v in c.text.split()] + [float(c.attrib["weight"])]
                    for c in element[1:]
                ],
            }
        elif element[0].tag == "monkhorst_pack":
            return {
                "automatic": [
                    int(element[0].attrib[k]) for k in ["nk1", "nk2", "nk3", "k1", "k2", "k3"]
                ]
            }

    def _parse_array(element):
        dims = [int(val) for val in reversed(element.attrib["dims"].split())]
        arr = np.array([float(val) for val in element.text.split()]).reshape(dims)
        return [v.tolist() for v in arr]

    def _parse_general_info(element, outp_dict):
        outp_dict["general_info"] = {
            c.tag: c.attrib for c in element if c.tag in ["creator", "created"]
        }

    def _parse_parallel_info(element, outp_dict):
        outp_dict["parallel_info"] = {c.tag: int(c.text) for c in element}

    def _parse_input(element, outp_dict):
        bfgs_mapping = {
            "ndim": "bfgs_ndim",
            "trust_radius_min": "trust_radius_min",
            "trust_radius_max": "trust_radius_max",
            "trust_radius_init": "trust_radius_ini",
            "w1": "w_1",
            "w2": "w_2",
        }
        inp_dict = outp_dict.setdefault("input", {"system": {}})
        for child in element:
            if child.tag == "control_variables":
                inp_dict["control"] = {c.tag: transform_str_value(c.text) for c in child}
            elif child.tag == "atomic_species":
                inp_dict["system"]["ntyp"] = int(child.attrib["ntyp"])
                inp_dict["atomic_species"] = {
                    c.attrib["name"]: (float(c[0].text), c[1].text) for c in child
                }
            elif child.tag == "atomic_structure":
                strct_dict = _parse_atomic_structure(child)
                strct_dict["cell_parameters"] = strct_dict.pop("cell")
                inp_dict["system"].update(strct_dict.pop("system"))
                inp_dict.update(strct_dict)
            elif child.tag == "dft":
                for c1 in child:
                    if c1.tag == "functional":
                        inp_dict["system"]["input_dft"] = c1.text
                    elif c1.tag == "vdW":
                        inp_dict["system"].update({c.tag: transform_str_value(c.text) for c in c1})
            elif child.tag in ["spin", "bands"]:
                inp_dict[child.tag] = {c.tag: transform_str_value(c.text) for c in child}
            elif child.tag == "basis":
                inp_dict["system"].update({c.tag: transform_str_value(c.text) for c in child})
            elif child.tag == "electron_control":
                inp_dict["electrons"] = {c.tag: transform_str_value(c.text) for c in child}
            elif child.tag == "k_points_IBZ":
                inp_dict["k_points"] = _parse_kpoints(child)
            elif child.tag == "ion_control":
                inp_dict["ions"] = {}
                for c1 in child:
                    # TODO take care of molecular dynamics:
                    if c1.tag == "bfgs":
                        inp_dict["ions"].update(
                            {bfgs_mapping[c.tag]: transform_str_value(c.text) for c in c1}
                        )
                    else:
                        inp_dict["ions"][c1.tag] = transform_str_value(c1.text)
            elif child.tag == "cell_control":
                inp_dict["cell"] = {}
                for c1 in child:
                    if c1.tag == "cell_do_free":
                        inp_dict["cell"]["cell_dofree"] = transform_str_value(c1.text)
                    else:
                        inp_dict["cell"][c1.tag] = transform_str_value(c1.text)
            elif child.tag == "symmetry_flags":
                inp_dict["system"].update({c.tag: transform_str_value(c.text) for c in child})
            elif child.tag == "free_positions":
                inp_dict["free_positions"] = _parse_array(child)

    def _parse_step(element, outp_dict):
        step_dict = {}
        for c1 in element:
            if c1.tag == "scf_conv":
                step_dict["scf_conv"] = {c.tag: transform_str_value(c.text) for c in c1}
            elif c1.tag == "atomic_structure":
                step_dict.update(_parse_atomic_structure(c1))
            elif c1.tag == "total_energy":
                step_dict["total_energy"] = {c.tag: float(c.text) for c in c1}
            elif "rank" in c1.attrib and "dims" in c1.attrib:
                step_dict[c1.tag] = _parse_array(c1)
            # elif c1.tag == "stress":

        outp_dict.setdefault("steps", []).append(step_dict)

    def _parse_output(element, outp_dict):
        output = outp_dict.setdefault("output", {})
        for c1 in element:
            if c1.tag == "convergence_info":
                for c2 in c1:
                    output[c2.tag] = {c.tag: transform_str_value(c.text) for c in c2}
            elif c1.tag in [
                "algorithmic_info",
                "boundary_conditions",
                "magnetization",
                "total_energy",
            ]:
                output[c1.tag] = {c.tag: transform_str_value(c.text) for c in c1}
            elif c1.tag == "atomic_structure":
                output.update(_parse_atomic_structure(c1))
            elif c1.tag == "symmetries":
                output["symmetry_operations"] = []
                for c2 in c1:
                    if c2.tag == "symmetry":
                        for c3 in c2:
                            if c3.tag == "rotation":
                                rotation = _parse_array(c3)
                            elif c3.tag == "fractional_translation":
                                translation = [float(val) for val in c3.text.split()]
                        output["symmetry_operations"].append([rotation, translation])
            elif c1.tag == "basis_set":
                output[c1.tag] = {}
                dict_create_tree(outp_dict, ["input", "system"])
                for c2 in c1:
                    if c2.tag == "fft_grid":
                        outp_dict["input"]["system"].update(
                            {f"nr{i}": int(c2.attrib[f"nr{i}"]) for i in range(1, 4)}
                        )
                    elif c2.tag == "fft_smooth":
                        outp_dict["input"]["system"].update(
                            {f"nrs{i}": int(c2.attrib[f"nr{i}"]) for i in range(1, 4)}
                        )
                    elif c2.tag == "reciprocal_lattice":
                        output[c1.tag][c2.tag] = [
                            [float(val) for val in c3.text.split()] for c3 in c2
                        ]
                    else:
                        output[c1.tag][c2.tag] = transform_str_value(c2.text)
            elif c1.tag == "band_structure":
                output[c1.tag] = {}
                for c2 in c1:
                    if c2.tag == "starting_k_points":
                        output[c1.tag][c2.tag] = _parse_kpoints(c2)
                    elif c2.tag == "ks_energies":
                        for c3 in c2:
                            if c3.tag in ["eigenvalues", "occupations"]:
                                output[c1.tag].setdefault(c3.tag, []).append(
                                    [float(val) for val in c3.text.split()]
                                )
                            elif c3.tag == "k_point":
                                output[c1.tag].setdefault("kpoints", []).append(
                                    [float(val) for val in c3.text.split()]
                                )
                                output[c1.tag].setdefault("weights", []).append(
                                    float(c3.attrib["weight"])
                                )
                    else:
                        output[c1.tag][c2.tag] = transform_str_value(c2.text)
            elif c1.tag == "forces":
                output["forces"] = _parse_array(c1)

    def _parse_timing_info(element, outp_dict):
        outp_dict["timing_info"] = {
            c1.attrib["label"]: {c.tag: float(c.text) for c in c1} for c1 in element
        }

    def _extract_structure(inp_dict, label=None, incl=None):
        cell_key = "cell_parameters" if "cell_parameters" in inp_dict else "cell"
        strct_dict = {
            "label": label,
            "elements": [],
            "positions": [],
            "cell": [[v1 * length.Bohr for v1 in v0] for v0 in inp_dict[cell_key]],
            "pbc": True,
            "is_cartesian": True,
            "attributes": {
                k: val
                for k, val in inp_dict.items()
                if k not in ["atomic_positions", cell_key, "forces"]
            },
        }
        if incl is not None:
            strct_dict["attributes"].update(copy.deepcopy(incl))
        for el, p1, p2, p3 in inp_dict["atomic_positions"]:
            strct_dict["elements"].append(el)
            strct_dict["positions"].append((p1 * length.Bohr, p2 * length.Bohr, p3 * length.Bohr))
        if "forces" in inp_dict:
            strct_dict["site_attributes"] = {"forces": [tuple(val) for val in inp_dict["forces"]]}
        return strct_dict

    tree = ET.parse(file_path)
    outp_dict = {}
    for child in tree.getroot():
        if child.tag == "exit_status":
            exit_status = int(child.text)
            outp_dict["exit_status"] = int(child.text)
            if raise_error and exit_status != 0:
                raise ValueError(
                    f"Calculation did not finish properly, exit status: {exit_status}. "
                    "To obtain output, set `raise_error` to False."
                )
        elif child.tag == "closed":
            pass
        else:
            locals()[f"_parse_{child.tag}"](child, outp_dict)
    if extract_structures:
        if strct_type is None:
            if "steps" in outp_dict:
                strct_type = "steps"
            elif "output" in outp_dict:
                strct_type = "output"
            else:
                strct_type = "input"

        label = (
            ""
            if outp_dict["input"]["control"]["prefix"] is None
            else outp_dict["input"]["control"]["prefix"]
        )
        incl = (
            {}
            if strct_include is None
            else {k: v for k, v in outp_dict.items() if k in strct_include}
        )
        if strct_type == "steps":
            outp_dict["structures"] = [
                _extract_structure(d, label="_".join([label, f"step_{i}"]), incl=incl)
                for i, d in enumerate(outp_dict[strct_type])
            ]
        else:
            outp_dict["structures"] = [
                _extract_structure(
                    outp_dict[strct_type], label="_".join([label, strct_type]), incl=incl
                )
            ]
    return outp_dict


@read_structure(r".*\.in(p)?$")
def read_qe_input_structure(file_path: str) -> dict:
    """
    Read structure from the Quantum ESPRESSO input file.
    ibrav parameters are not yet fully supported.

    Parameters
    ----------
    file_path : str
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
    with custom_open(file_path, "r") as input_file:
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


@read_band_structure(r".*bands\.dat$")
def read_qe_band_structure(file_path: str) -> dict:
    """
    Read band structure file from Quantum ESPRESSO.
    Spin-polarized calculations are not yet supported.

    Parameters
    ----------
    file_path : str
        Path of the output-file of Quantum ESPRESSO containing the band structure.

    Returns
    -------
    band_structure : dict
        Dictionary containing the k-path and th eigenvalues as well as the occupations.
    """
    kpoints = []
    bands = []
    with custom_open(file_path, "r") as bands_file:
        nr_bands = 0
        current_bands = []
        parse_kpoint = True
        for line in bands_file:
            line_split = line.split()
            # Catch the number of bands and k-points at the beginning of the file:
            if line.startswith(" &plot"):
                nr_bands = int(line_split[2][:-1])
                parse_kpoint = True
            # Parse k-point:
            elif parse_kpoint:
                kpoints.append([float(line_split[0]), float(line_split[1]), float(line_split[2])])
                parse_kpoint = False
            else:
                current_bands += [float(eigenvalue) for eigenvalue in line_split]
                if len(current_bands) == nr_bands:
                    parse_kpoint = True
                    bands.append(current_bands)
                    current_bands = []
    return {"kpoints": kpoints, "unit_y": "eV", "bands": bands}


@read_total_dos(r".*dos\.dat$")
def read_qe_total_dos(file_path: str) -> dict:
    """
    Read the total density of states from Quantum ESPRESSO.

    Parameters
    ----------
    file_path : str
        Path of the output-file of Quantum ESPRESSO containing the total density of states.

    Returns
    -------
    pdos : dict
        Dictionary containing the projected density of states for each atom.
    """
    energy = []
    tdos = []
    e_fermi = None
    with custom_open(file_path, "r") as tdos_file:
        for line in tdos_file:
            line_split = line.split()
            if not line.startswith("#"):
                energy.append(float(line_split[0]))
                tdos.append(float(line_split[1]))
            else:
                e_fermi = float(line_split[-2])
    return {"energy": energy, "tdos": tdos, "unit_x": "eV", "e_fermi": e_fermi}


@read_multiple(
    pattern=r"^.*pdos_atm#(?P<at_idx>\d*)?\((?P<el>[a-zA-Z]*)"
    + r"?\)\_wfc\#(?P<orb_idx>\d*)?\((?P<orb>[a-z])?\)$",
    is_read_proj_dos_method=True,
)
def read_qe_proj_dos(folder_path):
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

    indices = [(val, idx) for idx, val in enumerate(folder_path["at_idx"])]
    indices.sort(key=lambda point: point[0])
    _, indices = zip(*indices)
    for idx in indices:
        # Get regex details:
        at_idx = folder_path["at_idx"][idx]
        el = folder_path["el"][idx]
        orb = folder_path["orb"][idx]
        orb_idx = folder_path["orb_idx"][idx]

        # Check which kind the pdos belongs to:
        if at_idx not in kind_indices:
            kind_indices[at_idx] = len(atomic_pdos)
            atomic_pdos.append({"kind": el + "_" + at_idx})

        # The energy is only parsed from the first file, we assume the same energy range:
        parse_energy = len(energy) == 0

        # Read pdos, we only read the orbital contributions here, the summation is performed in
        # the plotting-class:
        with custom_open(folder_path["file_path"][idx], "r") as pdos_file:

            # Get inof from regex:
            qn_labels = quantum_numbers[orb]

            # Create empty lists for each quantum number:
            for qn_label in qn_labels:
                atomic_pdos[kind_indices[at_idx]][orb_idx + "_" + qn_label] = []

            # Iterate over lines and fill the list:
            for line in pdos_file:
                if not line.startswith("#"):
                    line_split = line.split()
                    if parse_energy:
                        energy.append(float(line_split[0]))
                    for qn_idx in range(len(qn_labels)):
                        qn_label = orb_idx + "_" + qn_labels[qn_idx]

                        # Fix bug in output when exponential is too small, e.g.: 0.292-105 instead
                        # of 0.292E-105
                        float_number = 0.0
                        try:
                            float_number = float(line_split[2 + qn_idx])
                        except ValueError:
                            pass
                        atomic_pdos[kind_indices[at_idx]].setdefault(qn_label, []).append(
                            float_number
                        )

    return {"energy": energy, "pdos": atomic_pdos, "unit_x": "eV"}


@read_structure(r".*\.xml", preset_kwargs={"extract_structures": True})
def read_xml(
    file_name: str,
    extract_structures: bool = False,
    strct_type: str = None,
    strct_include: list = None,
):
    """
    Read xml output file.

    Notes
    -----
        This function is deprecated and will be removed, please use `aim2dat.io.read_qe_xml`
        instead.

    Parameters
    ----------
    file_name : str
        Path of the xml-file of Quantum ESPRESSO.
    extract_structures : bool
        Whether to extract alls crystal structures and add them to the output dictionary with the
        key ``'structures'``.
    strct_type : str
        Type of extracted structure(s). Supported options are ``'input'``, ``'output'`` or
        ``'steps'``.
    strct_include : list
        List of dictionary keys that are included in the structure attributes.

    Returns
    -------
    dict
        Output dictionary.
    """
    from warnings import warn

    warn(
        "This function will be removed, please use `aim2dat.io.read_qe_xml` instead.",
        DeprecationWarning,
        2,
    )
    return read_qe_xml(
        file_path=file_name,
        extract_structures=extract_structures,
        strct_type=strct_type,
        strct_include=strct_include,
    )


@read_structure(r".*\.in(p)?$")
def read_input_structure(file_name):
    """
    Read structure from the Quantum ESPRESSO input file.
    ibrav parameters are not yet fully supported.

    Notes
    -----
        This function is deprecated and will be removed, please use
        `aim2dat.io.read_qe_input_structure` instead.

    Parameters
    ----------
    file_name : str
        Path of the input-file of Quantum ESPRESSO containing structural data.

    Returns
    -------
    dict
        Dictionary containing the structural information.
    """
    from warnings import warn

    warn(
        "This function will be removed, please use `aim2dat.io.read_qe_input_structure` instead.",
        DeprecationWarning,
        2,
    )
    return read_qe_input_structure(file_path=file_name)


def read_band_structure(file_name):
    """
    Read band structure file from Quantum ESPRESSO.
    Spin-polarized calculations are not yet supported.

    Notes
    -----
        This function is deprecated and will be removed, please use
        `aim2dat.io.read_qe_band_structure` instead.

    Parameters
    ----------
    file_name : str
        Path of the output-file of Quantum ESPRESSO containing the band structure.

    Returns
    -------
    band_structure : dict
        Dictionary containing the k-path and th eigenvalues as well as the occupations.
    """
    from warnings import warn

    warn(
        "This function will be removed, please use `aim2dat.io.read_qe_band_structure` instead.",
        DeprecationWarning,
        2,
    )
    return read_qe_band_structure(file_path=file_name)


def read_total_density_of_states(file_name):
    """
    Read the total density of states from Quantum ESPRESSO.

    Notes
    -----
        This function is deprecated and will be removed, please use `aim2dat.io.read_qe_total_dos`
        instead.

    Parameters
    ----------
    file_name : str
        Path of the output-file of Quantum ESPRESSO containing the total density of states.

    Returns
    -------
    pdos : dict
        Dictionary containing the projected density of states for each atom.
    """
    from warnings import warn

    warn(
        "This function will be removed, please use `aim2dat.io.read_qe_total_dos` instead.",
        DeprecationWarning,
        2,
    )
    return read_qe_total_dos(file_path=file_name)


@read_multiple(
    pattern=r"^.*pdos_atm#(?P<at_idx>\d*)?\((?P<el>[a-zA-Z]*)"
    + r"?\)\_wfc\#(?P<orb_idx>\d*)?\((?P<orb>[a-z])?\)$"
)
def read_atom_proj_density_of_states(folder_path):
    """
    Read the projected density of states from Quantum ESPRESSO.

    Notes
    -----
        This function is deprecated and will be removed, please use `aim2dat.io.read_qe_proj_dos`
        instead.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the pdos ouput-files.

    Returns
    -------
    pdos : dict
        Dictionary containing the projected density of states for each atom.
    """
    from warnings import warn

    warn(
        "This function will be removed, please use `aim2dat.io.read_qe_proj_dos` instead.",
        DeprecationWarning,
        2,
    )
    return read_qe_proj_dos(folder_path=folder_path)
