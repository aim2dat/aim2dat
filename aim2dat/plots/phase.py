"""
Module plotting quantities with respect to the chemical composition.
"""

# Standard library imports
import re
import math

# from statistics import mean

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.plots.base_plot import _BasePlot
from aim2dat.ext_interfaces.import_opt_dependencies import _return_ext_interface_modules
from aim2dat.fct.hull import get_convex_hull, get_minimum_maximum_points
from aim2dat.chem_f import (
    transform_str_to_dict,
    transform_list_to_dict,
    transform_dict_to_str,
    transform_dict_to_latexstr,
)
import aim2dat.utils.space_groups as utils_sg
from aim2dat.elements import get_element_symbol


def PhaseDiagram(*args, **kwargs):
    """Depreciated PhaseDiagram class."""
    from warnings import warn

    warn(
        "This class will be removed, please use `PhasePlot` instead.",
        DeprecationWarning,
        2,
    )
    return PhasePlot(*args, **kwargs)


def _get_concentration(entry, elements):
    conc = None
    if all([el in elements for el in entry["chem_formula"]]):
        conc = 0.0
        if elements[0] in entry["chem_formula"]:
            conc = entry["chem_formula"][elements[0]] / sum(entry["chem_formula"].values())
    return conc


class PhasePlot(_BasePlot):
    """
    Plot the formation energy of binary and ternary material systems.

    Attributes
    ----------
    show_convex_hull : bool
        Whether to calculate and show the convex hull in the plot.
    """

    _crystal_system_mapping = {
        "triclinic": 0,
        "monoclinic": 1,
        "orthorhombic": 2,
        "tetragonal": 3,
        "trigonal": 4,
        "hexagonal": 5,
        "cubic": 6,
    }
    _supported_plot_types = ["scatter", "numbers"]
    _default_y_labels = {
        "formation_energy": r"$E_{form}$ in eV/atom",
        "stability": "Stability in eV/atom",
        "numbers": "Nr. of structures",
    }

    def __init__(
        self,
        plot_type="scatter",
        plot_property="formation_energy",
        show_crystal_system=False,
        show_convex_hull=True,
        show_lower_hull=False,
        show_upper_hull=False,
        top_labels=[],
        hist_bin_size=0.1,
        **kwargs,
    ):
        """Initialize object."""
        _BasePlot.__init__(self, **kwargs)
        self.plot_type = plot_type
        self.plot_property = plot_property
        self.show_crystal_system = show_crystal_system
        self.show_convex_hull = show_convex_hull
        self.show_lower_hull = show_lower_hull
        self.show_upper_hull = show_upper_hull
        self.top_labels = top_labels
        self.hist_bin_size = hist_bin_size
        self._all_elements = []
        self._elements = None

    @property
    def elements(self):
        """
        list: List of elements that are included in the plot. If set to ``None`` all elements
            are included.
        """
        return self._elements

    @elements.setter
    def elements(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError("`elements` needs to be of type list or tuple.")
        elements = []
        for val0 in value:
            elements.append(get_element_symbol(val0))
        self._elements = elements

    @property
    def plot_type(self):
        """
        Specify plot type. Supported options are: ``'formation_energy'``, ``'stability'``,
        ``'band_gap'``, ``'direct_band_gap'`` and ``'numbers'``.
        """
        return self._plot_type

    @plot_type.setter
    def plot_type(self, value):
        if value not in self._supported_plot_types:
            raise ValueError(
                f"`plot_type` '{value}' is not suppported. Supported options are '"
                + "', '".join(self._supported_plot_types)
                + "'."
            )
        self._plot_type = value

    @property
    def show_crystal_system(self):
        """
        Show crystal system of the phases.
        """
        return self._show_crystal_system

    @show_crystal_system.setter
    def show_crystal_system(self, value):
        self._show_crystal_system = value

    @property
    def top_labels(self):
        """
        list or str or dict: Chemical formulas that are shown as labels in the plot.
        """
        return self._top_labels

    @top_labels.setter
    def top_labels(self, value):
        if not isinstance(value, (list, tuple)):
            value = [value]
        for f_idx, formula in enumerate(value):
            if isinstance(formula, str):
                value[f_idx] = transform_str_to_dict(formula)
            elif isinstance(formula, (list, tuple)):
                value[f_idx] = transform_list_to_dict(formula)
        self._top_labels = value

    def add_data_point(
        self,
        data_label,
        formula,
        formation_energy=None,
        stability=None,
        unit=None,
        space_group=None,
        attributes=None,
    ):
        """
        Add datapoint to the dataset.

        If the ``data_label`` does not exist, a new data set with label ``data_label`` is created.

        Parameters
        ----------
        data_label : str
            Internal label used to plot and compare multiple data sets.
        formula : dict
            Chemical formula of the material, e.g. ``{'Cs': 1, 'Sb': 2}``.
        formation_energy : float (optional)
            Formation energy of the material.
        stability : float (optional)
            Stability of the material.
        unit : str (optional)
            Unit of the formation energy and stability.
        space_group : str or int (optional)
            Space group of the material, as symbol or number. The default value is ``None``.
        attributes : dict (optional)
            Additional attributes of the material that can be plotted.
        """
        if isinstance(formula, (list, tuple)):
            formula = transform_list_to_dict(formula)
        elif isinstance(formula, str):
            formula = transform_str_to_dict(formula)
        elif isinstance(formula, dict):
            pass
        else:
            raise TypeError("`formula` needs to be of type list/tuple/str/dict.")

        entry = {"chem_formula": formula, "attributes": {}}
        if formation_energy is not None:
            entry["attributes"]["formation_energy"] = {"value": formation_energy, "unit": unit}
        if stability is not None:
            entry["attributes"]["stability"] = {"value": stability, "unit": unit}
        if space_group is None:
            entry["space_group"] = None
        else:
            entry["space_group"] = utils_sg.transform_to_nr(space_group)
        if attributes is not None:
            if not isinstance(attributes, dict):
                raise TypeError("`attributes` needs to be of type dict.")
            for attr_key, attr_val in attributes.items():
                if attr_key not in entry["attributes"]:
                    entry["attributes"][attr_key] = attr_val
        if data_label in self._data:
            self._data[data_label].append(entry)
        else:
            self._data[data_label] = [entry]
        for el in formula.keys():
            if el not in self._all_elements:
                self._all_elements.append(el)

    def import_from_structure_collection(self, data_label, structure_collection):
        """
        Import data from a StructureCollection object.

        Parameters
        ----------
        data_label : str
            Internal label used to plot and compare multiple data sets.
        structure_collection : aim2dat.strct.StructureCollection
            Instance of StructureCollection containing all structures.
        """
        for structure in structure_collection:
            dp_kwargs = {}
            for attr in ["formation_energy", "stability"]:
                value = structure["attributes"].get(attr, None)
                dp_kwargs[attr] = value
                if isinstance(value, dict):
                    dp_kwargs[attr] = value["value"]
                    if "unit" in value and "unit" not in dp_kwargs:
                        dp_kwargs["unit"] = value["unit"]
            dp_kwargs["space_group"] = structure["attributes"].get("space_group", None)
            dp_kwargs["attributes"] = structure["attributes"]
            self.add_data_point(
                data_label, transform_list_to_dict(structure["elements"]), **dp_kwargs
            )

    def import_from_pandas_df(
        self, data_label, data_frame, structure_column="optimized_structure"
    ):
        """
        Import data from pandas data frame.

        Parameters
        ----------
        data_label : str
            Internal label used to plot and compare multiple data sets.
        data_frame : pandas.DataFrame
            Pandas data frame containing the total energy or formation energy and the structural
            details.
        structure_column : str (optional)
            Column containing AiiDA structure nodes used to determine structural and compositional
            properties. The default value is ``'optimized_structure'``.
        """
        self._check_data_label(data_label)
        pattern = re.compile(r"([\w-]+)?\s*\(?(\w+)?\)?")
        attributes = {}

        comp_type = None
        comp_cols = {}
        sg_col = None
        for col in data_frame.columns:
            col_splitted = col.split("_")
            if structure_column == col:
                comp_type = "aiida_structure"
                comp_cols = col
            elif "chem_formula" in col and comp_type is None:
                comp_type = "chem_formula"
                comp_cols = col
            elif (
                "nr_atoms" in col
                and len(col_splitted) > 2
                and (comp_type is None or comp_type == "atoms_per_el")
            ):
                symbol = col_splitted[-1]
                comp_type = "atoms_per_el"
                comp_cols[symbol] = col
            elif "space_group" in col:
                sg_col = col
            else:
                found = pattern.findall(col)[0]
                if len(found) > 1:
                    label, unit = found
                else:
                    unit = None
                    label = found[0]
                attributes[label] = {"value": None, "unit": unit, "col": col}

        for _, row in data_frame.iterrows():
            chem_f = getattr(self, "_extract_formula_from_" + comp_type)(row, comp_cols)
            row_attr = {}
            for attr_label, attr_details in attributes.items():
                row_attr[attr_label] = {
                    "value": row[attr_details["col"]],
                    "unit": attr_details["unit"],
                }
            self.add_data_point(data_label, chem_f, attributes=row_attr, space_group=row[sg_col])

    def _prepare_to_plot(self, data_labels, subplot_assignment):
        plot_elements = self._elements
        if plot_elements is None:
            plot_elements = self._all_elements
        if len(plot_elements) != 2:
            raise NotImplementedError("Feature is not yet supported.")

        # TODO add y-label?
        self._auto_set_axis_properties(
            x_label=r"$x_{" + plot_elements[0] + r"}$",
        )
        data_sets = []
        label_tl = []
        values_tl = []
        for top_label in self._top_labels:
            if all([el in plot_elements for el in top_label.keys()]):
                x_value = 0.0
                if plot_elements[0] in top_label:
                    x_value = top_label[plot_elements[0]] / sum(top_label.values())
                label_tl.append(transform_dict_to_latexstr(top_label))
                values_tl.append(x_value)
                data_sets.append(
                    {
                        "x": x_value,
                        "color": "black",
                        "linestyle": "dashed",
                        "scaled": True,
                        "type": "vline",
                    }
                )
        create_function = getattr(self, "_create_2d_" + self.plot_type + "_data_sets")
        for data_set_idx, data_label in enumerate(data_labels):
            create_function(data_sets, data_set_idx, data_label, plot_elements)
        return (
            data_sets,
            None,
            None,
            None,
            None,
            [{"ticks": values_tl, "tick_labels": label_tl, "coord": "x"}],
        )

    @staticmethod
    def _extract_formula_from_aiida_structure(row, comp_cols):
        """Extract chemical formula from aiida structure nodes."""
        backend_module = _return_ext_interface_modules("aiida")
        struct = backend_module._load_data_node(row[comp_cols])
        return transform_str_to_dict(struct.get_formula())

    @staticmethod
    def _extract_formula_from_atoms_per_el(row, comp_cols):
        """Extract chemical formula from atoms-per-element lists."""
        chem_f = {}
        for symbol, col in comp_cols.items():
            chem_f[symbol] = row[col]
        return chem_f

    @staticmethod
    def _extract_formula_from_chem_formula(row, comp_cols):
        """Create entries list from list of chemical formulas."""
        return row[comp_cols]

    def _create_2d_scatter_data_sets(self, data_sets, ds_idx, data_label, elements):
        """Process entries for 2D phase diagram."""
        cs_map = {"Phases": 0}
        if self.show_crystal_system:
            cs_map = self._crystal_system_mapping
        entries = self._return_data_set(data_label)
        new_data_sets = []
        space_groups = []
        used_labels = []
        data_points = []

        for entry in entries:
            conc = _get_concentration(entry, elements)
            if conc is None:
                continue
            if self.plot_property not in entry["attributes"]:
                print(
                    f"Property '{self.plot_property}' missing, could not plot entry "
                    + transform_dict_to_str(entry["chem_formula"])
                    + "."
                )
                continue
            y_value = entry["attributes"][self.plot_property]
            if isinstance(y_value, dict):
                y_value = y_value["value"]
            data_points.append((conc, y_value))
            if self.show_crystal_system and entry["space_group"] is None:
                print(
                    "Space group missing, could not plot entry "
                    + transform_dict_to_str(entry["chem_formula"])
                    + "."
                )
                continue
            crystal_system = "Phases"
            if self.show_crystal_system:
                space_groups.append(entry["space_group"])
                crystal_system = utils_sg.get_crystal_system(entry["space_group"])
            data_set = {
                "x_values": [conc],
                "y_values": [y_value],
                "marker": cs_map[crystal_system],
                "linestyle": "none",
                "markerfacecolor": "none",
                "markeredgewidth": 1.7,
                "color": ds_idx,
                "legendgrouptitle_text": data_label,
                "legendgroup": data_label,
                "group_by": "color",
            }
            if crystal_system not in used_labels:
                data_set["label"] = crystal_system
                used_labels.append(crystal_system)
            new_data_sets.append(data_set)

        if self.show_crystal_system:
            zipped = list(zip(space_groups, new_data_sets))
            zipped.sort(key=lambda point: point[0])
            _, new_data_sets = zip(*zipped)
            new_data_sets = list(new_data_sets)

        # create_hulls
        show_hull = {}
        for hull_type in ["convex_hull", "lower_hull", "upper_hull"]:
            hull_attr = getattr(self, "show_" + hull_type)
            if not isinstance(hull_attr, bool):
                if len(hull_attr) > ds_idx:
                    hull_attr = hull_attr[ds_idx]
                else:
                    hull_attr = hull_attr[-1]
            show_hull[hull_type] = hull_attr
        if show_hull["convex_hull"]:
            x_values, y_values = get_convex_hull(data_points, upper_hull=False)
            new_data_sets.append(
                {
                    "x_values": list(x_values),
                    "y_values": list(y_values),
                    "color": ds_idx,
                    "legendgrouptitle_text": data_label,
                    "legendgroup": data_label,
                    "label": "Convex hull",
                    "group_by": "color",
                }
            )
        if show_hull["lower_hull"] or show_hull["upper_hull"]:
            x_values, min_values, max_values = get_minimum_maximum_points(data_points)
            if show_hull["lower_hull"] and show_hull["upper_hull"]:
                new_data_sets.append(
                    {
                        "x_values": x_values,
                        "y_values": min_values,
                        "y_values_2": max_values,
                        "color": ds_idx,
                        "alpha": ds_idx,
                        "use_fill_between": True,
                    }
                )
            else:
                if show_hull["lower_hull"]:
                    y_values = min_values
                else:
                    y_values = max_values
                new_data_sets.append(
                    {
                        "x_values": x_values,
                        "y_values": y_values,
                        "color": ds_idx,
                    }
                )
        data_sets += new_data_sets

    def _create_2d_numbers_data_sets(self, data_sets, _, data_label, elements):
        """Process data for bar plot."""
        cs_map = {"Phases": 0}
        if self.show_crystal_system:
            cs_map = self._crystal_system_mapping
        entries = self._return_data_set(data_label)
        x_values = [
            float(val) for val in np.arange(0.0, 1.0 + self.hist_bin_size, self.hist_bin_size)
        ]
        for crystal_system, color_idx in cs_map.items():
            data_sets.append(
                {
                    "x_values": x_values,
                    "bottom": [0.0] * len(x_values),
                    "heights": [0.0] * len(x_values),
                    "width": self.hist_bin_size - 0.005,
                    "type": "bar",
                    "color": color_idx,
                    "label": crystal_system,
                }
            )
        for entry in entries:
            if self.show_crystal_system and entry["space_group"] is None:
                print(
                    "Space group missing, could not plot entry "
                    + transform_dict_to_str(entry["chem_formula"])
                    + "."
                )
                continue
            conc = _get_concentration(entry, elements)
            if conc is None:
                continue

            crystal_system = "Phases"
            cs_idx = 0
            if self.show_crystal_system:
                crystal_system = utils_sg.get_crystal_system(entry["space_group"])
                cs_idx = self._crystal_system_mapping[crystal_system]
            x_idx = math.ceil((conc - 0.5 * self.hist_bin_size) / self.hist_bin_size)
            ds_idx = len(data_sets) - len(cs_map) + cs_idx
            data_sets[ds_idx]["heights"][x_idx] += 1
            for upper_idx in range(ds_idx + 1, len(data_sets)):
                data_sets[upper_idx]["bottom"][x_idx] += 1
