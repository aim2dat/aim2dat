"""Interface functions for the pandas library."""

# Third party library imports
import pandas as pd

# Internal library imports
from aim2dat.chem_f import transform_list_to_dict


def _parse_el_concentrations(pd_series_dict, dtypes, structure, all_elements):
    chem_formula = transform_list_to_dict(structure["elements"])
    for element in all_elements:
        dtypes["el_conc_" + element] = float
        if "el_conc_" + element not in pd_series_dict:
            pd_series_dict["el_conc_" + element] = []
        if element in chem_formula:
            pd_series_dict["el_conc_" + element].append(
                chem_formula[element] / sum(chem_formula.values())
            )
        else:
            pd_series_dict["el_conc_" + element].append(0.0)


def _parse_nr_atoms_per_el(pd_series_dict, dtypes, structure, all_elements):
    chem_formula = transform_list_to_dict(structure["elements"])
    for element in all_elements:
        dtypes["nr_atoms_" + element] = int
        if "nr_atoms_" + element not in pd_series_dict:
            pd_series_dict["nr_atoms_" + element] = []
        if element in chem_formula:
            pd_series_dict["nr_atoms_" + element].append(chem_formula[element])
        else:
            pd_series_dict["nr_atoms_" + element].append(0)


def _parse_nr_atoms(pd_series_dict, dtypes, structure, all_elements):
    dtypes["nr_atoms"] = int
    if "nr_atoms" not in pd_series_dict:
        pd_series_dict["nr_atoms"] = []
    pd_series_dict["nr_atoms"].append(len(structure["elements"]))


def _parse_coordination(pd_series_dict, dtypes, structure, all_elements):
    coord_labels = [
        "nrs_avg",
        "nrs_stdev",
        "nrs_max",
        "nrs_min",
        "distance_avg",
        "distance_stdev",
        "distance_max",
        "distance_min",
    ]
    elements = set(structure.elements)
    for el1 in all_elements:
        for el2 in all_elements:
            for coord_l in coord_labels:
                if (
                    el1 in elements
                    and el2 in elements
                    and "coordination" in structure.function_args
                    and (el1, el2) in structure.extras["coordination"][coord_l]
                ):
                    value = structure.extras["coordination"][coord_l][(el1, el2)]
                else:
                    value = pd.NA
                label = f"coord_{coord_l}_{el1}-{el2}"
                if label not in pd_series_dict:
                    pd_series_dict[label] = []
                pd_series_dict[label].append(value)


def _parse_general_attribute(pd_series_dict, dtypes, units, entry, dict_key):
    if dict_key not in pd_series_dict:
        pd_series_dict[dict_key] = []
        dtypes[dict_key] = None
        units[dict_key] = None
    value = pd.NA
    if entry is not None:
        value = entry.get(dict_key, pd.NA)
    if isinstance(value, dict):
        if "unit" in value:
            units[dict_key] = value["unit"]
        if "value" in value:
            value = value["value"]
    pd_series_dict[dict_key].append(value)
    if value is not None and dtypes[dict_key] != float:
        dtypes[dict_key] = type(value)


def _create_strct_c_pandas_df(strct_c, exclude_columns):
    exclude_columns = [] if exclude_columns is None else exclude_columns
    column_types = {
        "el_concentrations": _parse_el_concentrations,
        "nr_atoms": _parse_nr_atoms,
        "nr_atoms_per_el": _parse_nr_atoms_per_el,
        "coordination": _parse_coordination,
    }
    all_elements = strct_c.get_all_elements()
    all_attributes = strct_c.get_all_attribute_keys()
    pd_series_dict = {"label": [], "structure": []}
    dtypes = {"label": str}
    units = {}
    for strct in strct_c:
        pd_series_dict["label"].append(strct.label)
        if "structure" not in exclude_columns:
            pd_series_dict["structure"].append(strct)
        for column_type, process_function in column_types.items():
            if column_type not in exclude_columns:
                process_function(pd_series_dict, dtypes, strct, all_elements)
        for key0 in all_attributes:
            if key0 not in exclude_columns:
                _parse_general_attribute(pd_series_dict, dtypes, units, strct["attributes"], key0)
    return _turn_dict_into_pandas_df(pd_series_dict, units=units)


def _delete_empty_columns(pd_series_dict, dtypes=None):
    """
    Delete empty columns from series dictionary.
    """
    columns_to_del = []
    for col, values in pd_series_dict.items():
        if any(isinstance(val, (tuple, list)) for val in values):
            continue
        if all(val is None or pd.isna(val) for val in values):
            columns_to_del.append(col)
    for col in columns_to_del:
        del pd_series_dict[col]
        if dtypes is not None:
            del dtypes[col]


def _turn_dict_into_pandas_df(pd_series_dict, dtypes=None, units=None):
    if units is not None:
        for column, unit in units.items():
            if unit is not None:
                pd_series_dict[f"{column} ({unit})"] = pd_series_dict.pop(column)
                if dtypes is not None:
                    dtypes[f"{column} ({unit})"] = dtypes.pop(column)
    for column in pd_series_dict.keys():
        pd_series_dict[column] = [pd.NA if val is None else val for val in pd_series_dict[column]]
    _delete_empty_columns(pd_series_dict, dtypes)
    data_frame = pd.DataFrame(pd_series_dict)
    if dtypes is not None:
        data_frame = data_frame.astype(dtypes)
    return data_frame


def _apply_color_map(data_frame, color_map):
    return data_frame.style.map(color_map)
