"""Wrapper functions for the ase Atoms object."""

# Standard library imports
import os
import re

# Third party library imports
from ase import Atoms
from ase.io import read, write


def _extract_structure_from_atoms(atoms):
    """Extract a dictionary with structural parameters from the ase Atoms object."""
    positions = []
    elements = []
    kinds = []
    tags_sum = 0
    for atom in atoms:
        elements.append(atom.symbol)
        positions.append([float(atom.position[idx]) for idx in range(3)])
        kinds.append(f"{atom.symbol}{atom.tag}")
        tags_sum += atom.tag

    # if tags_sum == 0:
    #     kinds = elements

    structure_dict = {
        "elements": elements,
        "kinds": kinds if tags_sum != 0 else None,
        "positions": positions,
        "pbc": atoms.get_pbc().tolist(),
        "is_cartesian": True,
    }
    if any(structure_dict["pbc"]):
        structure_dict["cell"] = [cell_v.tolist() for cell_v in atoms.cell.array]
    return structure_dict


def _create_atoms_from_structure(structure):
    """Create ase atoms object from structure dictionary."""
    tags = [0] * len(structure.elements)
    if structure.kinds is not None:
        tags = []
        for k in structure.kinds:
            tag = re.findall(r"\d+", k)
            if tag:
                tags.append(int(tag[0]))
            else:
                tags.append(0)

    return Atoms(
        structure.elements,
        positions=structure.positions,
        cell=structure.cell,
        pbc=structure.pbc,
        tags=tags,
    )


def _load_structure_from_file(file_path):
    """
    Load structure from file using the ase implementation.

    As for cif-files a tempory
    """

    def add_line_to_temp_files(temp_files, lines_str, file_indices):
        """Add line to active tempory cif file contents."""
        for file_idx in file_indices:
            temp_files[file_idx] += lines_str

    def check_partial_occupancy(loop_content, loop_tags, temp_files, file_indices):
        """
        Check partial occupancy of sites and create separate files for different configurations.
        """
        if "_atom_site_occupancy" in loop_tags and "_atom_site_label" in loop_tags:
            label_idx = loop_tags.index("_atom_site_label")
            occ_idx = loop_tags.index("_atom_site_occupancy")
            site_labels = []
            loop_lines = []
            occupancies = []
            common_content = ""
            curr_line = ""
            for line in loop_content.split("\n"):
                if line.strip().startswith("_") or "loop_" in line:
                    common_content += line + "\n"
                elif len(curr_line.split()) < len(loop_tags):
                    curr_line += " " + line
                else:
                    site_labels.append(curr_line.split()[label_idx])
                    loop_lines.append(curr_line)
                    occupancies.append(curr_line.split()[occ_idx])
                    curr_line = line
            ind_content = []
            for site_label, site_occ, site_line in zip(site_labels, occupancies, loop_lines):
                site_occ = site_occ.replace("(", "")
                site_occ = site_occ.replace(")", "")
                # Not sure if this is general enough...
                if float(site_occ) < 1.0:
                    conf_idx = site_label.count("'")
                    while len(ind_content) <= conf_idx:
                        ind_content.append("")
                    ind_content[conf_idx] += site_line + "\n"
                else:
                    common_content += site_line + "\n"
            if len(ind_content) > 1:
                orig_file_idx = len(temp_files) - 1
                for ind_cont in ind_content[1:]:
                    file_indices.append(len(temp_files))
                    temp_files.append(temp_files[orig_file_idx] + common_content + ind_cont)
                for idx0 in range(orig_file_idx + 1):
                    temp_files[idx0] += common_content + ind_content[0]
                return True
            else:
                return False
        else:
            return False

    relevant_cif_tags = [
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
        "_symmetry_cell_setting",
        "_symmetry_space_group_name_H-M",
        "_space_group_name_H-M_alt",
        "_space_group_IT_number",
        "_space_group_symop_operation_xyz",
        "_symmetry_equiv_pos_site_id",
        "_symmetry_equiv_pos_as_xyz",
        "_atom_site_type_symbol",
        "_atom_site_label",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]

    file_format = file_path.split(".")[-1]
    if file_format.lower() == "cif":
        structures = []
        loop_pattern = re.compile(r"^\s*loop_\s+([\s\S]*)?$")
        # Creates temporary file with correct formating that is deleted afterwards...
        with open(file_path, "r") as f1:
            in_loop = False
            add_loop = False
            loop_length = 0
            loop_content = []
            act_file_indices = []
            temp_files = []
            for line_idx, line in enumerate(f1):
                found_match = loop_pattern.match(line)

                # Check loop_ sections:
                if found_match is not None:
                    # Reset variables with a new loop:
                    add_loop = False
                    loop_content = ""
                    in_loop = True
                    current_loop_tags = []

                    # Check first loop tags:
                    match = found_match.groups()[0]
                    loop_content += "\nloop_\n"
                    for loop_tag in match.split():
                        current_loop_tags.append(loop_tag)
                        if loop_tag in relevant_cif_tags:
                            add_loop = True
                        loop_length += 1
                        loop_content += "      " + loop_tag + "\n"
                elif in_loop:
                    if line.strip() == "":
                        # In the case of partial occupancy we would like to have two
                        # files (or more)..
                        in_loop = False
                        if check_partial_occupancy(
                            loop_content, current_loop_tags, temp_files, act_file_indices
                        ):
                            pass
                        elif add_loop:
                            add_line_to_temp_files(temp_files, loop_content, act_file_indices)
                        add_line_to_temp_files(temp_files, line, act_file_indices)
                    elif line.strip().startswith("_"):
                        current_loop_tags += line.split()
                        if any(cif_tag == line.split()[0] for cif_tag in relevant_cif_tags):
                            add_loop = True
                        loop_length += len(line.split())
                        loop_content += line
                    elif line.strip().startswith(";"):
                        pass
                    else:
                        loop_content += line
                # Check relevant single tags
                elif line.strip().startswith("_"):
                    if "_symmetry_space_group_name_H-M" in line:
                        # Not sure if this works for all cases...
                        symbol = "".join(line.split()[1:])
                        symbol = symbol.replace("1", "")
                        add_line_to_temp_files(
                            temp_files,
                            "_symmetry_space_group_name_H-M " + symbol + "\n",
                            act_file_indices,
                        )
                    elif any(cif_tag == line.split()[0] for cif_tag in relevant_cif_tags):
                        add_line_to_temp_files(temp_files, line, act_file_indices)
                # Check beginning of new file:
                elif line.strip().startswith("data"):
                    act_file_indices = [len(temp_files)]
                    temp_files.append(f"data_temp_cif_{len(temp_files) - 1}\n\n")
            if in_loop:
                if check_partial_occupancy(
                    loop_content, current_loop_tags, temp_files, act_file_indices
                ):
                    pass
                elif add_loop:
                    add_line_to_temp_files(temp_files, loop_content, act_file_indices)

        for file_idx, temp_file_content in enumerate(temp_files):
            if len(temp_file_content.strip().split()) < 2:
                continue
            temp_filename = file_path + "_" + str(file_idx) + "_temp.cif"
            with open(temp_filename, "w") as temp_f:
                temp_f.write(temp_file_content)
            atoms = read(temp_filename)
            structures.append(_extract_structure_from_atoms(atoms))
            os.remove(temp_filename)
    else:
        structures = [_extract_structure_from_atoms(read(file_path))]
    return structures


def _write_structure_to_file(struct_dict, file_path):
    """Write structure to file using the ase implementation."""
    write(file_path, _create_atoms_from_structure(struct_dict))
