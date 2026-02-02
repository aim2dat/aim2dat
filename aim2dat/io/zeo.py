"""
Read and write zeo++ files.
"""

# Internal library imports
from aim2dat.io.utils import read_structure, write_structure, custom_open, parse_to_str
from aim2dat.utils.strct import _get_cell_from_lattice_p
from aim2dat.utils.space_groups import transform_to_nr


@read_structure(r".*(\.cssr|\.v1|\.cuc)", preset_kwargs=None)
def read_zeo_file(file_path: str):
    """
    Read zeo++ file.

    Parameters
    ----------
    file_path : str
        Path to the zeo++ file.

    Returns
    -------
    dict
        Dictionary containing structural information.

    Raises
    ------
    ValueError
        Could not detect file format, supported formats are 'cssr', 'v1' and 'cuc'.
    """
    is_cuc_file = False
    is_v1_file = False
    is_cssr_file = False
    in_cssr_header = False
    label = None
    cell = []
    cell_p = None
    elements = []
    positions = []
    attributes = {}

    with custom_open(file_path, "r") as f_obj:
        for line in f_obj:
            line = line.strip()
            if line.startswith("#") or len(line) == 0:
                continue

            line_sp = line.split()
            if line.startswith("Processing:"):
                is_cuc_file = True
                label = line_sp[1] if line_sp[1].lower() != "none" else label
            elif line.startswith("Unit cell vectors:"):
                is_v1_file = True
            elif not is_cssr_file and not is_cuc_file and not is_v1_file:
                is_cssr_file = True
                in_cssr_header = True
                cell_p = [float(v) for v in line.split()[:3]]
            elif is_cuc_file:
                if line.startswith("Unit_cell"):
                    cell = _get_cell_from_lattice_p(*[float(v) for v in line.split()[1:7]])
                else:
                    line_sp = line.split()
                    elements.append(line_sp[0])
                    positions.append([float(v) for v in line.split()[1:4]])
            elif is_v1_file:
                line_sp = line.split()
                if line.startswith(("va=", "vb=", "vc=")):
                    cell.append([float(v) for v in line_sp[1:4]])
                elif len(line_sp) == 4:
                    elements.append(line_sp[0])
                    positions.append([float(v) for v in line.split()[1:4]])
            elif is_cssr_file:
                line_sp = line.split()
                # if len(line_sp) < 3:
                #    pass
                if in_cssr_header:
                    if len(line_sp) < 3:
                        in_cssr_header = False
                    else:
                        cell = _get_cell_from_lattice_p(
                            *(cell_p + [float(v) for v in line.split()[:3]])
                        )
                        attributes["space_group"] = transform_to_nr(" ".join(line_sp[5:]))
                else:
                    if len(line_sp) < 3:
                        label = line_sp[1] if line_sp[1].lower() != "none" else label
                    else:
                        elements.append(line_sp[1])
                        positions.append([float(v) for v in line.split()[2:5]])
            else:
                raise ValueError(
                    "Could not detect file format, supported formats are 'cssr', 'v1' and 'cuc'."
                )
    return {
        "label": label,
        "elements": elements,
        "positions": positions,
        "cell": cell,
        "pbc": True,
        "is_cartesian": is_v1_file,
        "attributes": attributes,
    }


@write_structure(r".*(\.cssr|\.v1|\.cuc)", preset_kwargs=None)
def write_zeo_file(file_path: str, structure: dict):
    """
    Write structure to an input file for zeo++.

    Parameters
    ----------
    file_path : str
        Path to the zeo-file. Possible endings are ``'.cssr'``, ``'.v1'``, or ``'.cuc'``.
    structure : dict
        Structure which is written to the file.

    Raises
    ------
    ValueError
        Invalid file format. Allowed formats are: ``'.cssr'``, ``'.v1'``, or ``'.cuc'``.
    """
    if str(file_path).endswith(".cssr"):
        output = [" ".join(map(parse_to_str, structure.cell_lengths))]
        output.append
        output.append(
            " ".join(map(parse_to_str, structure.cell_angles))
            + " SPGR = "
            + structure.calc_space_group()["space_group"]["international_short"]
        )
        output.append(f"{len(structure.positions)} 0")
        output.append(f"0 {structure.label}")
        for idx, (el, pos) in enumerate(structure.iter_sites(get_scaled_pos=True)):
            output.append(
                " ".join(
                    [
                        parse_to_str(val, add_space_front=i != 1)
                        for i, val in enumerate([idx + 1, el] + list(pos) + 9 * [0])
                    ]
                )
            )

    elif str(file_path).endswith(".v1"):
        output = ["Unit cell vectors:"]
        for a, vec in zip(["va", "vb", "vc"], structure.cell):
            output.append(
                " ".join([f"{a}="] + [parse_to_str(v, add_space_front=True) for v in vec])
            )
        output.append(f"{len(structure.positions)}")
        for el, pos in structure.iter_sites(get_cart_pos=True):
            output.append(
                " ".join(
                    [
                        parse_to_str(val, add_space_front=i > 0)
                        for i, val in enumerate([el] + list(pos))
                    ]
                )
            )

    elif str(file_path).endswith(".cuc"):
        output = [f"Processing: {structure.label}"]
        output.append(
            "Unit_cell: "
            + " ".join(map(parse_to_str, structure.cell_lengths))
            + " "
            + " ".join(map(parse_to_str, structure.cell_angles))
        )
        for el, pos in structure.iter_sites(get_scaled_pos=True):
            output.append(
                " ".join(
                    [
                        parse_to_str(val, add_space_front=i != 1)
                        for i, val in enumerate([el] + list(pos))
                    ]
                )
            )
    else:
        raise ValueError(
            "Invalid file format. Allowed formats are: ``'.cssr'``, ``'.v1'``, or ``'.cuc'``."
        )

    with open(file_path, "w") as f_obj:
        for line in output:
            f_obj.write(line + "\n")
