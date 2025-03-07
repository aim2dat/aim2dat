"""
Functions to read output-files of critic2.
"""

# Standard library imports
import re

# Internal library imports
from aim2dat.io.utils import custom_open


def read_critic2_stdout(file_path: str) -> dict:
    """
    Read standard output file.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    dict
        Results.
    """
    result_dict = {"plane_files": []}
    with custom_open(file_path, "r") as stdout_file:
        pc_section = False
        for line in stdout_file:
            line_splitted = line.split()
            if line.startswith("+ critic2"):
                if "version" in line_splitted[-2]:
                    result_dict["critic2_version"] = line_splitted[-1]
                    result_dict["critic2_branch"] = line_splitted[-3][1:-2]
                else:
                    result_dict["critic2_version"] = line_splitted[-2]
                    result_dict["critic2_branch"] = line_splitted[-4][1:-2]
            if line.startswith("* Yu-Trinkle integration"):
                result_dict["method"] = "Yu-Trinkle integration"
            elif line.startswith("* Henkelman et al. integration"):
                result_dict["method"] = "Henkelmann et al. integration"
            if line.startswith("* Integrated atomic properties"):
                pc_section = True
                result_dict["partial_charges"] = []
            elif pc_section and line.startswith("--------"):
                pc_section = False
            elif pc_section and not line.startswith("#"):
                element = line.split()[3].replace("_", "")
                result_dict["partial_charges"].append(
                    {"element": element, "population": float(line.split()[9])}
                )
            if line.startswith("* PLANE written to file:"):
                result_dict["plane_files"].append(line.split()[-1])
            if line.startswith("ERROR"):
                result_dict["aborted"] = True
                result_dict["error"] = line
                break
            if line.startswith("CRITIC2 ended successfully"):
                result_dict["nwarnings"] = int(line.split()[-4][1:])
                result_dict["ncomments"] = int(line.split()[-2])
            elif line.startswith("CRITIC2 ended "):
                result_dict["aborted"] = True
    return result_dict


def read_critic2_plane(file_path: str) -> dict:
    """
    Read output plane file.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    dict
        plane details.
    """
    unit_pattern = re.compile(r"^[\S\s]+\(units=([a-z]+)?\S+$")
    plane = {"coordinates": [], "values": [], "coordinates_unit": None}
    with custom_open(file_path, "r") as plane_file:
        for line in plane_file:
            line_splitted = line.split()
            if line.startswith("#"):
                match = unit_pattern.match(line)
                if match is not None:
                    plane["coordinates_unit"] = match.groups()[0]
            elif line.strip() == "":
                continue
            elif len(line_splitted) > 5:
                plane["coordinates"].append((float(line_splitted[3]), float(line_splitted[4])))
                field_values = [float(line_val) for line_val in line_splitted[5:]]
                if len(field_values) > 1:
                    plane["values"].append(tuple(field_values))
                else:
                    plane["values"].append(field_values[0])
    return plane
