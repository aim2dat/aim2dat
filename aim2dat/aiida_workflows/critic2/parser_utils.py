"""Auxiliary functions for the critic2 parser and io functions."""

# Standard library imports
import re


def _parse_plane_file(handle):
    """Parse plane files - only scalar fields are supported so far."""
    unit_pattern = re.compile(r"^[\S\s]+\(units=([a-z]+)?\S+$")
    output_string = handle.read()
    plane = {"coordinates": [], "values": [], "coordinates_unit": None}
    for line in output_string.splitlines():
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


def _parse_stdout_file(handle):
    """Parse partial charge files."""
    output_string = handle.read()
    result_dict = {"plane_files": []}
    pc_section = False
    for line in output_string.splitlines():
        line_splitted = line.split()
        if line.startswith("+ critic2"):
            try:
                result_dict["critic2_version"] = float(line_splitted[-1])
                result_dict["critic2_branch"] = line_splitted[-3][1:-2]
            except ValueError:
                result_dict["critic2_version"] = float(line_splitted[-2])
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
