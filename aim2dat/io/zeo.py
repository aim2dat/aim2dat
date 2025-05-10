"""
Input operations for zeo++.
"""


def write_zeo_file(file_path: str, struct_dict: dict):
    """
    Write structure dictonaryto an input file for zeo++.

    Parameters
    ----------
    file_path : str
        Path to the zeo-file. Possible endings are ``'.cssr'``, ``'.v1'``, or ``'.cuc'``.
    struct_dict : dict
        Structure which is written to the file.

    Raises
    ------
    ValueError
        Invalid file format. Allowed formats are: ``'.cssr'``, ``'.v1'``, or ``'.cuc'``.
    """
    if file_path.endswith(".cssr"):
        output = [" ".join(map(str, struct_dict.cell_lengths))]
        output.append
        output.append(
            " ".join(map(str, struct_dict.cell_angles))
            + " SPGR = "
            + struct_dict.calc_space_group()["space_group"]["international_short"]
        )
        output.append(f"{len(struct_dict.positions)} 0")
        output.append(f"0 {struct_dict.label}")
        for i, el_pos in enumerate(zip(struct_dict.elements, struct_dict.scaled_positions)):
            output.append(
                f"  {i+1} {el_pos[0]} {el_pos[1][0]} {el_pos[1][1]} {el_pos[1][2]} "
                + " ".join(map(str, 9 * [0]))
            )

    elif file_path.endswith(".v1"):
        output = ["Unit cell vectors:"]
        for a, vec in zip(["va", "vb", "vc"], struct_dict.cell):
            output.append(f"{a}= {vec[0]} {vec[1]} {vec[2]}")
        output.append(f"{len(struct_dict.positions)}")
        for el, pos in zip(struct_dict.elements, struct_dict.positions):
            output.append(f"{el} {pos[0]} {pos[1]} {pos[2]}")

    elif file_path.endswith(".cuc"):
        output = [f"Processing: {struct_dict.label}"]
        output.append(
            "Unit_cell: "
            + " ".join(map(str, struct_dict.cell_lengths))
            + " "
            + " ".join(map(str, struct_dict.cell_angles))
        )
        for el, pos in zip(struct_dict.elements, struct_dict.scaled_positions):
            output.append(f"{el} {pos[0]} {pos[1]} {pos[2]}")

    else:
        raise ValueError(
            "Invalid file format. Allowed formats are: ``'.cssr'``, ``'.v1'``, or ``'.cuc'``."
        )

    with open(file_path, "w") as file:
        file.write("\n".join(output))
