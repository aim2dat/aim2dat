"""
Module of functions to read/write xyz files.
"""

# Standard library imports
import warnings
from typing import TYPE_CHECKING, Union, List

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.io.utils import custom_open, read_structure, write_structure

if TYPE_CHECKING:
    from aim2dat.strct import Structure, StructureCollection


_STRING_DELIMITERS = ["'", '"']
_VALUE_DELIMITERS = [",", " "]
_TYPE_MAPPING = {
    "r": float,
    "i": int,
    "s": str,
}
_PROPERTIES_MAPPING = {
    "species": "elements",
    "pos": "positions",
    "tags": "kinds",
}


def _parse_comment_line(line):
    def _cut_off_delimiters(str_value):
        if str_value[0] in _STRING_DELIMITERS:
            str_value = str_value[1:]
        if str_value[-1] in _STRING_DELIMITERS:
            str_value = str_value[:-1]
        return str_value.strip()

    columns = [("elements", 0, 1, str), ("positions", 1, 4, float)]
    if "=" in line:
        keys = []
        values = []
        eq_pos = [-1] + [pos for pos, char in enumerate(line) if char == "="] + [len(line) - 1]
        for idx, p in enumerate(eq_pos[1:]):
            substr = line[eq_pos[idx] + 1 : p].strip()
            if idx == 0:
                keys.append(_cut_off_delimiters(substr))
            elif idx == len(eq_pos) - 2:
                values.append(_cut_off_delimiters(substr))
            else:
                for val_del in _VALUE_DELIMITERS:
                    if val_del in substr:
                        del_pos = len(substr) - list(reversed(substr)).index(val_del)
                        if del_pos < len(substr) - 1:
                            values.append(_cut_off_delimiters(substr[: del_pos - 1].strip()))
                            keys.append(_cut_off_delimiters(substr[del_pos:].strip()))
                        break
        add_pars = {"attributes": {key: val for key, val in zip(keys, values)}}
        for attr in list(add_pars["attributes"].keys()):
            if attr.lower() == "lattice":
                value = add_pars["attributes"].pop(attr).split()
                if len(value) != 9:
                    raise ValueError("'Lattice' needs to have 9 numbers separated by space.")
                add_pars["cell"] = [
                    [float(val) for val in value[i * 3 : (i + 1) * 3]] for i in range(3)
                ]
                add_pars["pbc"] = True
            elif attr.lower() == "pbc":
                value = [
                    val.lower().startswith("t") for val in add_pars["attributes"].pop(attr).split()
                ]
                if len(value) != 3:
                    raise ValueError("'pbc' needs to have 3 booleans separated by space.")
                add_pars["pbc"] = value
            elif attr.lower() == "properties":
                value = add_pars["attributes"].pop(attr).split(":")
                if len(value) % 3 != 0:
                    raise ValueError(
                        "'Properties' needs to have a multiple of 3 entries separated by ':'."
                    )
                columns = []
                col_count = 0
                for i in range(0, len(value), 3):
                    label = (
                        _PROPERTIES_MAPPING[value[i]]
                        if value[i] in _PROPERTIES_MAPPING
                        else value[i]
                    )
                    columns.append(
                        (
                            label,
                            col_count,
                            col_count + int(value[i + 2]),
                            _TYPE_MAPPING[value[i + 1].lower()],
                        )
                    )
                    col_count += int(value[i + 2])
    else:
        add_pars = {"pbc": False, "attributes": {"comment": line}}
    return columns, add_pars


def _parse_column_value(column_key, column_value, parsed_values, n_values=None, v_type=None):
    # Length check
    if isinstance(column_value, (list, tuple, np.ndarray)):
        n_values0 = len(column_value)
    else:
        n_values0 = 1
        column_value = [column_value]
    if n_values and n_values0 != n_values:
        warnings.warn(f"Cannot add '{column_key}' due to length mismatch.")
        return False, None, None

    output_value = []
    for v in column_value:
        if v_type is None:
            if isinstance(v, float):
                v_type = "r"
            elif isinstance(v, int):
                v_type = "i"
            elif isinstance(v, str):
                v_type = "s"
            else:
                warnings.warn(
                    f"Cannot add '{column_key}' since the values cannot"
                    + " be cast into str, int or float."
                )
                return False, None, None

        v_type0 = v_type

        try:
            v = _TYPE_MAPPING[v_type0](v)
            if v_type0 == "r":
                v = f"{v:16.8f}"
            elif v_type0 == "i":
                v = str(v)
                v = "".join(" " for _ in range(8 - len(v))) + v
            else:
                v = str(v)
                if " " in v:
                    warnings.warn(f"Cannot add '{column_key}' since the values have white spaces.")
                    return False, None, None
        except ValueError:
            warnings.warn(f"Cannot add '{column_key}' since the values have different types.")
            return False, None, None

        output_value.append(v)
    parsed_values.append(output_value)
    return True, n_values0, v_type0


def _create_columns(columns, n_sites):
    mapping = {"elements": "species", "positions": "pos"}
    comment = "Properties="

    outp_lines = ["" for _ in range(n_sites)]
    for key, val in columns:
        parsed_values = []
        n_values = None
        v_type = None
        for v in val:
            can_add, n_values, v_type = _parse_column_value(
                key, v, parsed_values, n_values, v_type
            )
            if not can_add:
                break
        if not can_add:
            continue

        comment += f"{mapping.get(key, key)}:{v_type.upper()}:{n_values}:"
        for i, val in enumerate(parsed_values):
            for v in val:
                white_spaces = " " + " ".join("" for _ in range(3 - len(v)))
                outp_lines[i] += v + white_spaces
    outp_lines = [line.rstrip() for line in outp_lines]
    return comment[:-1], outp_lines


@read_structure(r".*\.xyz", preset_kwargs=None)
def read_xyz_file(file_path: str) -> List[dict]:
    """
    Read xyz file.

    Parameters
    ----------
    file_path : str
        Path to the xyz-file.

    Returns
    -------
    list
        List of dictionaries containing structural information.
    """
    structures = []
    with custom_open(file_path, "r") as f_obj:
        n_atoms = None
        columns = None
        in_col_section = False
        for line in f_obj:
            line = line.strip()
            if n_atoms is None and line == "":
                continue

            if n_atoms is None:
                n_atoms = [0, int(line)]
                structures.append({"site_attributes": {}})
            elif not in_col_section:
                columns, add_pars = _parse_comment_line(line)
                structures[-1].update(add_pars)
                in_col_section = True
            else:
                line_sp = line.split()
                for key, start_idx, end_idx, val_type in columns:
                    val = [val_type(v) for v in line_sp[start_idx:end_idx]]
                    if len(val) == 1:
                        val = val[0]
                    if key in _PROPERTIES_MAPPING.values():
                        structures[-1].setdefault(key, []).append(val)
                    else:
                        structures[-1]["site_attributes"].setdefault(key, []).append(val)
                n_atoms[0] += 1
                if n_atoms[0] == n_atoms[1]:
                    n_atoms = None
                    columns = None
                    in_col_section = False
    print(structures)
    return structures


@write_structure(r".*\.xyz", preset_kwargs=None)
def write_xyz_file(
    file_path: str,
    structures: Union["Structure", "StructureCollection", list],
    include_attributes: list = None,
    exclude_attributes: list = None,
    include_site_attributes: list = None,
    exclude_site_attributes: list = None,
):
    """
    Write xyz file.

    file_path : str
        Path to xyz file.
    structures : aim2dat.strct.Structure, aim2dat.strct.StructureCollection, list
        Structure object, StructureCollection object or list of Structure objects.
    include_attributes : list
        List of attributes that are written to file.
    exclude_attributes : list
        List of attributes that are not written to file.
    include_site_attributes : list
        List of site attributes that are written to file.
    exclude_site_attributes : list
        List of site attributes that are not written to file.
    """
    structures = [structures] if type(structures).__name__ == "Structure" else structures
    exclude_attributes = [] if exclude_attributes is None else exclude_attributes
    exclude_site_attributes = [] if exclude_site_attributes is None else exclude_site_attributes
    with open(file_path, "w") as f_obj:
        for strct in structures:
            comment_line = ""
            if strct.cell is not None:
                comment_line += (
                    'Lattice="' + " ".join([str(v) for val in strct.cell for v in val]) + '" '
                )

            columns = [("elements", strct.elements), ("positions", strct.positions)]
            if any(k is not None for k in strct.kinds) and "kinds" not in exclude_site_attributes:
                columns.append(("tags", strct.kinds))
            if include_site_attributes is not None:
                for attr in include_site_attributes:
                    if attr in strct.site_attributes:
                        columns.append((attr, strct.site_attributes[attr]))
            else:
                for attr, value in strct.site_attributes.items():
                    if attr not in exclude_site_attributes:
                        columns.append((attr, strct.site_attributes[attr]))

            comment, column_lines = _create_columns(columns, len(strct))

            comment_line += comment
            if include_attributes is not None:
                for attr in include_attributes:
                    if attr in strct.attributes:
                        val = str(strct.attributes[attr])
                        if " " in val or " " in attr:
                            warnings.warn(
                                f"Cannot add '{attr}' since the values have white spaces."
                            )
                            continue
                        comment_line += f" {attr}={val}"
            else:
                for attr, val in strct.attributes.items():
                    if attr not in exclude_attributes:
                        val = str(val)
                        if " " in val or " " in attr:
                            warnings.warn(
                                f"Cannot add '{attr}' since the values have white spaces."
                            )
                            continue
                        comment_line += f" {attr}={val}"
            comment_line += ' pbc="' + " ".join("T" if v else "F" for v in strct.pbc) + '"'

            f_obj.write(f"{len(strct)}\n")
            f_obj.write(comment_line + "\n")
            for col in column_lines:
                f_obj.write(col + "\n")
