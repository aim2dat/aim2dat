"""Surface energy plots."""

# Standard library imports
import re

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.plots.base_plot import _BasePlot
from aim2dat.units import UnitConverter
from aim2dat.chem_f import transform_str_to_dict, transform_list_to_dict
import aim2dat.elements as utils_el


def _validate_miller_indices(m_indices):
    if not isinstance(m_indices, (list, tuple)):
        raise TypeError("`miller_indices` need to be of type list or tuple.")
    if len(m_indices) not in [3, 4]:
        raise ValueError("`miller_indices` need to have a length of 3 or 4.")
    return m_indices


def _check_map(dict_map, value):
    if len(dict_map) == 0:
        val_idx = 0
        dict_map[value] = val_idx
    elif value in dict_map:
        val_idx = dict_map[value]
    else:
        val_idx = max(list(dict_map.values())) + 1
        dict_map[value] = val_idx
    return val_idx


def _transform_miller_indices_to_str(m_indices):
    m_idx_str = "("
    for idx0 in m_indices:
        if idx0 < 0:
            m_idx_str += r"$\bar{" + str(idx0)[1] + "}$"
        else:
            m_idx_str += str(idx0)
    m_idx_str += ")"
    return m_idx_str


class SurfacePlot(_BasePlot):
    """
    Plot the surface energy with respect to the chemical potential.
    """

    _supported_plot_types = ["chem_potential", "excess_atoms"]
    _supported_ter_labeling_schemes = ["excess_ter_sto"]

    # _line_styles = ["solid", "dashed", "dotted", "dashdot"]
    _markers = ["o", "x", ">", "<"]

    def __init__(
        self,
        area_unit="angstrom",
        energy_unit="eV",
        plot_element=None,
        plot_type="chem_potential",
        plot_properties="surface_energy",
        plot_labels=None,
        show_x_label=True,
        show_y_label=True,
        **kwargs,
    ):
        """Initialize class."""
        _BasePlot.__init__(self, **kwargs)
        self._bulk_phase = None
        self._elemental_phases = {}
        self._data = {}

        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self.energy_unit = energy_unit
        self.area_unit = area_unit
        self.plot_element = plot_element
        self.plot_type = plot_type
        self.plot_properties = plot_properties

    @property
    def plot_type(self):
        """str: Plot type. Supported options are `'chem_potential'` or `'excess_atoms'`."""
        return self._plot_type

    @plot_type.setter
    def plot_type(self, value):
        if value not in self._supported_plot_types:
            raise ValueError(
                f"`plot_type` '{value}' is not supported. Supported values are '"
                + "' or '".join(self._supported_plot_types)
                + "'."
            )
        self._plot_type = value

    @property
    def plot_properties(self):
        """str, tuple or list: Properties that are plotted on the y-axis."""
        return self._plot_properties

    @plot_properties.setter
    def plot_properties(self, value):
        if isinstance(value, str):
            value = (value,)
        elif isinstance(value, (list, tuple)):
            value = tuple(value)
        else:
            raise TypeError(
                "`plot_properties` needs to be of type str or tuple/list of str objects."
            )
        if "surface_energy" in value and len(value) > 1:
            raise ValueError("'surface_energy' can only be plotted as a single property.")
        self._plot_properties = value

    @property
    def area_unit(self):
        """str: Unit of the surface area."""
        return self._area_unit

    @area_unit.setter
    def area_unit(self, value):
        self._area_unit = self._check_unit(value)

    @property
    def energy_unit(self):
        """str: Energy unit."""
        return self._energy_unit

    @energy_unit.setter
    def energy_unit(self, value):
        self._energy_unit = self._check_unit(value)

    @property
    def elemental_phases(self):
        """Elemental phases."""
        return self._elemental_phases

    @property
    def bulk_phase(self):
        """Bulk phase."""
        return self._bulk_phase

    def set_elemental_phase(self, element, total_energy, nr_atoms=1, unit="eV"):
        """
        Set elemental phase.

        Parameters
        ----------
        element : str, int
            Name, symbol or atomic number of the element.
        total_energy : float
            Total energy of the phase.
        nr_atoms : int (optional)
            Number of atoms of the phase. The default value is ``1``.
        unit : str (optional)
            Unit of the energy. The default value is ``'eV'``.
        """
        element = utils_el.get_element_symbol(element)
        self._elemental_phases[element] = {
            "total_energy": total_energy / nr_atoms,
            "unit": self._check_unit(unit),
        }

    def set_bulk_phase(self, formula, total_energy, unit="eV"):
        """
        Set bulk phase.

        Parameters
        ----------
        formula : dict, list or str
            Chemical formula of the bulk phase.
        total_energy : float
            Total energy of the phase.
        unit : str (optional)
            Unit of the energy. The default value is ``'eV'``.
        """
        self._bulk_phase = {
            "formula": self._check_formula(formula),
            "total_energy": total_energy,
            "unit": self._check_unit(unit),
        }

    def add_surface_facet(
        self,
        data_label,
        formula,
        total_energy,
        surface_area,
        miller_indices=None,
        termination_label=None,
        area_unit="angstrom",
        energy_unit="eV",
        **kwargs,
    ):
        """
        Add surface.

        Parameters
        ----------
        data_label : str
            Internal label used to plot and compare multiple data sets.
        formula : dict, list or str
            Chemical formula of the surface slab.
        total_energy : float
            Total energy of the surface slab.
        surface_area : float
            Surface area of the facet.
        area_unit : str (optional)
            Unit of the surface area. The default value is ``'angstrom'``.
        energy_unit : str (optional)
            Unit of the energy. The default value is ``'eV'``.
        """
        facet_data = {
            "formula": self._check_formula(formula),
            "total_energy": total_energy,
            "area": surface_area,
            "miller_indices": _validate_miller_indices(miller_indices),
            "ter": termination_label,
            "area_unit": self._check_unit(area_unit),
            "energy_unit": self._check_unit(energy_unit),
            "properties": kwargs,
        }
        if data_label in self._data:
            self._data[data_label].append(facet_data)
        else:
            self._data[data_label] = [facet_data]

    def import_from_pandas_df(
        self,
        data_label,
        data_frame,
        termination_labels=None,
        ter_labeling_scheme=None,
        extract_electronic_properties=False,
    ):
        """
        Import surface facets from a results pandas data frame of the workflow builder.

        Parameters
        ----------
        data_label : str
            Internal label used to plot and compare multiple data sets.
        data_frame : pandas.DataFrame
            Pandas data frame.
        termination_labels : list (optional)
            List of termination indicies. If set to ``None`` the termination index for each facet
            is extracted from the input SurfaceData of the workflow.
        ter_labeling_scheme : str (optional)
            Automatic labeling for surface slabs.
        extract_electronic_properties : bool (optional)
            Whether to extract the band gap and the ionization potential from the calculated
            band structure.
        """
        from aim2dat.plots import BandStructurePlot
        from aim2dat.ext_interfaces.aiida import _load_data_node

        energy_pattern = re.compile(r"total_energy\s+\((\w+)?\)")
        for col in data_frame.columns:
            if len(energy_pattern.findall(col)) > 0:
                energy_col = col
                energy_unit = energy_pattern.findall(col)[0]

        for row_idx, (_, row) in enumerate(data_frame.iterrows()):
            kwargs = {}
            inp_surface = _load_data_node(row["parent_node"])
            opt_slab = _load_data_node(row["optimized_structure"])
            p_cell_v = [
                v for idx, v in enumerate(opt_slab.cell) if idx != inp_surface.aperiodic_dir
            ]

            # if data_labels is None:
            #     dl = "(" + "".join([str(val) for val in inp_surface.miller_indices]) + ")"
            # else:
            #     dl = data_labels[row_idx]
            # miller_indices = inp_surface.miller_indices
            if termination_labels is None:
                t_label = "Ter. " + str(inp_surface.termination)
            else:
                t_label = termination_labels[row_idx]
            if "band_structure" in row:
                bs_plot = BandStructurePlot()
                bs_plot.import_from_aiida_bandsdata("_", row["band_structure"])
                gap_info = bs_plot.calculate_band_gap("_")
                kwargs["Band gap"] = gap_info["indirect_gap"]
                kwargs["Ionization potential"] = gap_info["vbm_energy"] * -1.0
            for key, value in row.items():
                if "e_fermi" in key:
                    # TODO add unit conversion
                    kwargs["Work function"] = row[key] * -1.0
                    break
            self.add_surface_facet(
                data_label,
                opt_slab.get_formula(),
                float(row[energy_col]),
                float(np.linalg.norm(np.cross(p_cell_v[0], p_cell_v[1]))),
                termination_label=t_label,
                miller_indices=inp_surface.miller_indices,
                area_unit="angstrom",
                energy_unit=energy_unit,
                **kwargs,
            )

    def _prepare_to_plot(self, data_labels, subplot_assignment):
        el1 = self.plot_element
        if el1 is None:
            el1 = list(self._elemental_phases.keys())[0]
        for el in self._elemental_phases.keys():
            if el != el1:
                el2 = el
        create_fct = getattr(self, "_create_" + self.plot_type + "_data_sets")
        data_sets, sec_axis = create_fct(data_labels, subplot_assignment, el1, el2)
        return (
            data_sets,
            None,
            None,
            None,
            None,
            sec_axis,
        )

    def _create_chem_potential_data_sets(self, data_labels, subplot_assignment, el1, el2):
        # Set axis labels:
        if self.show_x_label and self.x_label is None:
            self.x_label = (
                r"$\left(\mu_\mathrm{"
                + el1
                + r"} - \mu^\mathrm{bulk}_\mathrm{"
                + el1
                + r"}\right)$ in "
                + UnitConverter.plot_labels[self.energy_unit]
            )
        if (
            self.show_y_label
            and self.y_label is None
            and self.plot_properties[0] == "surface_energy"
        ):
            self.y_label = (
                "Surface energy in "
                + UnitConverter.plot_labels[self.energy_unit]
                + "/"
                + UnitConverter.plot_labels[self.area_unit]
                + r"$^\mathrm{2}$"
            )

        # Set boundaries:
        e_shift = UnitConverter.convert_units(
            self._elemental_phases[el1]["total_energy"],
            self._elemental_phases[el1]["unit"],
            self.energy_unit,
        )
        e_max = 0.0
        e_bulk = UnitConverter.convert_units(
            self._bulk_phase["total_energy"], self._bulk_phase["unit"], self.energy_unit
        )
        e_min = (
            e_bulk
            - self._bulk_phase["formula"][el2]
            * UnitConverter.convert_units(
                self._elemental_phases[el2]["total_energy"],
                self._elemental_phases[el2]["unit"],
                self.energy_unit,
            )
        ) / self._bulk_phase["formula"][el1] - e_shift
        values_tl = [e_min, e_max]
        labels_tl = [f"{el1}-poor", f"{el1}-rich"]

        # Create data sets:
        color_map = {}
        ls_map = {}
        data_sets = [
            [
                {"x": e_min, "color": "black", "linestyle": "--", "scaled": True, "type": "vline"},
                {"x": e_max, "color": "black", "linestyle": "--", "scaled": True, "type": "vline"},
            ]
            for _ in range(max(subplot_assignment) + 1)
        ]
        for dl, subp_a in zip(data_labels, subplot_assignment):
            if self.plot_properties[0] == "surface_energy":
                data_sets[subp_a] += self._process_surface_energy(
                    dl, el1, el2, e_bulk, e_min, e_max, e_shift, color_map, ls_map
                )
            else:
                data_sets[subp_a] += self._process_property(
                    dl, el1, el2, e_bulk, e_min, e_max, e_shift, color_map, ls_map
                )

        return data_sets, [{"ticks": values_tl, "tick_labels": labels_tl, "coord": "x"}]

    def _create_excess_atoms_data_sets(self, data_labels, subplot_assignment, el1, el2):
        ratio = self.bulk_phase["formula"][el1] / self.bulk_phase["formula"][el2]
        # print(ratio)
        if self.show_x_label and self.x_label is None:
            self.x_label = (
                f"Nr. excess {el1} atoms in #/{UnitConverter.plot_labels[self.area_unit]}"
                + r"$^\mathrm{2}$"
            )
        # TODO change to autoset properties...

        marker_map = {}
        color_map = {}
        data_sets = [{} for _ in range(max(subplot_assignment) + 1)]
        excess_maps = [{} for _ in range(max(subplot_assignment) + 1)]
        for subp_idx in range(max(subplot_assignment) + 1):
            for ds_idx, subp_a in enumerate(subplot_assignment):
                if subp_a != subp_idx:
                    continue

                for surface in self._data[data_labels[ds_idx]]:
                    label = data_labels[ds_idx]
                    group_label = ""
                    area = UnitConverter.convert_units(
                        surface["area"], surface["area_unit"], self.area_unit
                    )
                    excess = (
                        0.5 * (surface["formula"][el1] - ratio * surface["formula"][el2]) / area
                    )
                    customdata = [excess]
                    if "miller_indices" in surface:
                        m_idx_str = _transform_miller_indices_to_str(surface["miller_indices"])
                        customdata.append(m_idx_str)
                        label += " " + m_idx_str
                    else:
                        customdata.append(None)
                    if "ter" in surface:
                        group_label += surface["ter"]
                        customdata.append(surface["ter"])
                    else:
                        customdata.append(None)
                    for key in ["Ionization potential", "Work function", "Band gap"]:
                        customdata.append(surface["properties"].get(key, ""))
                    if excess in excess_maps[subp_idx]:
                        if group_label not in excess_maps[subp_idx][excess]["group_labels"]:
                            excess_maps[subp_idx][excess]["group_labels"].append(group_label)
                    else:
                        excess_maps[subp_idx][excess] = {"group_labels": [group_label]}
                    if label not in marker_map:
                        marker_map[label] = (
                            max(list(marker_map.values())) + 1 if len(marker_map) > 0 else 0
                        )
                    properties = []
                    for prop in self.plot_properties:
                        if prop not in surface["properties"]:
                            properties.append(None)
                            continue
                        if prop in excess_maps[subp_idx][excess]:
                            excess_maps[subp_idx][excess][prop] = (
                                min(
                                    [
                                        excess_maps[subp_idx][excess][prop][0],
                                        surface["properties"][prop],
                                    ]
                                ),
                                max(
                                    [
                                        excess_maps[subp_idx][excess][prop][1],
                                        surface["properties"][prop],
                                    ]
                                ),
                            )
                        else:
                            excess_maps[subp_idx][excess][prop] = (
                                surface["properties"][prop],
                                surface["properties"][prop],
                            )
                        properties.append(surface["properties"][prop])
                    if group_label in data_sets[subp_idx]:
                        data_sets[subp_idx][group_label].append(
                            (label, excess, properties, customdata)
                        )
                    else:
                        data_sets[subp_idx][group_label] = [
                            (label, excess, properties, customdata)
                        ]

        plot_data_sets = [[] for _ in range(max(subplot_assignment) + 1)]
        for excess_map, plot_ds, ds in zip(excess_maps, plot_data_sets, data_sets):
            excess_s = sorted(list(excess_map.keys()))
            # Plot hull for each property:
            for prop_idx, prop in enumerate(self.plot_properties):
                x_values = []
                y_values = [[], []]
                for excess in excess_s:
                    if prop not in excess_map[excess]:
                        continue
                    if all(v0 is not None for v0 in excess_map[excess][prop]):
                        x_values.append(excess)
                        for i0 in range(2):
                            y_values[i0].append(excess_map[excess][prop][i0])
                plot_ds.append(
                    {
                        "x_values": x_values,
                        "y_values": y_values[0],
                        # "y_values_2": y_values[0],
                        "type": "scatter",
                        "color": prop_idx,
                        "linestyle": prop_idx,
                        # "use_fill_between": True,
                        "alpha": prop_idx,
                    }
                )
                plot_ds.append(
                    {
                        "x_values": x_values,
                        "y_values": y_values[1],
                        # "y_values_2": y_values[0],
                        "type": "scatter",
                        "color": prop_idx,
                        "linestyle": prop_idx,
                        # "use_fill_between": True,
                        "alpha": prop_idx,
                    }
                )
            used_group_labels = []
            for excess in excess_s:
                for group_label in excess_map[excess]["group_labels"]:
                    if group_label in used_group_labels:
                        continue
                    used_group_labels.append(group_label)
                    for data_p in ds[group_label]:
                        if group_label in color_map:
                            color = color_map[group_label]
                        else:
                            color = len(self.plot_properties)
                            if len(color_map) > 0:
                                color = max(list(color_map.values())) + 1
                            color_map[group_label] = color
                        plot_ds.append(
                            {
                                "x_values": [data_p[1]] * len(self.plot_properties),
                                "y_values": data_p[2],
                                "linestyle": "none",
                                "type": "scatter",
                                "marker": marker_map[data_p[0]],
                                "color": color,
                                "label": data_p[0],
                                "legendgrouptitle_text": group_label,
                                "legendgroup": group_label,
                                "group_by": "color",
                                "customdata": [data_p[3]] * len(self.plot_properties),
                                "hovertemplate": "<b>Excess:</b> %{customdata[0]}<br>"
                                + "<b>Miller indices:</b> %{customdata[1]}<br>"
                                + "<b>Ter.:</b> %{customdata[2]}<br>"
                                + "<b>Ionization potential:</b> %{customdata[3]}<br>"
                                + "<b>Work function:</b> %{customdata[4]}<br>"
                                + "<b>Band gap:</b> %{customdata[5]}",
                            }
                        )
        if max(subplot_assignment) == 0:
            self._legend_order = (None, list(marker_map.keys()))
        return plot_data_sets, None

    def _process_surface_energy(
        self, dl, el1, el2, e_bulk, e_min, e_max, e_shift, color_map, ls_map
    ):
        data_sets = []
        for facet in self._data[dl]:
            m_indices_str = _transform_miller_indices_to_str(facet["miller_indices"])
            data_set = {
                "x_values": [float(val) for val in np.linspace(e_min, e_max, 500)],
                "type": "scatter",
                "color": _check_map(color_map, facet["ter"]),
                "legendgrouptitle_text": facet["ter"],
                "legendgroup": facet["ter"],
                "label": m_indices_str,
                "group_by": "color",
                "linestyle": _check_map(ls_map, m_indices_str),
                "linewidth": _check_map(ls_map, m_indices_str),
            }
            # TODO do conversions once and not at each step.
            data_set["y_values"] = [
                self._calculate_surf_energy(x_val, el1, el2, e_bulk, facet, e_shift)
                for x_val in data_set["x_values"]
            ]
            data_sets.append(data_set)
        return data_sets

    def _process_property(self, dl, el1, el2, e_bulk, e_min, e_max, e_shift, color_map, ls_map):
        data_sets = []
        x_values = [float(val) for val in np.linspace(e_min, e_max, 500)]
        for plot_idx, plot_label in enumerate(self.plot_properties):
            l_surf_e = {}
            extra_prop = {}
            for facet in self._data[dl]:
                if plot_label not in facet["properties"]:
                    continue
                surf_e = [
                    self._calculate_surf_energy(x_val, el1, el2, e_bulk, facet, e_shift)
                    for x_val in x_values
                ]
                if dl in l_surf_e:
                    for idx0, (surf_val1, surf_val2) in enumerate(zip(l_surf_e[dl], surf_e)):
                        if surf_val2 < surf_val1:
                            l_surf_e[dl][idx0] = surf_val2
                            extra_prop[dl][idx0] = facet["properties"][plot_label]
                else:
                    l_surf_e[dl] = surf_e
                    extra_prop[dl] = [facet["properties"][plot_label]] * 500

            for dl0, y_values in extra_prop.items():
                data_sets.append(
                    {
                        "x_values": x_values,
                        "y_values": y_values,
                        "type": "scatter",
                        "color": _check_map(color_map, facet["ter"]),
                        "label": plot_label,
                        "linestyle": plot_idx,
                        "linewidth": plot_idx,
                        "legendgrouptitle_text": dl0,
                        "legendgroup": dl0,
                        "group_by": "color",
                    }
                )
        return data_sets

    @staticmethod
    def _check_unit(value):
        value = value.lower()
        if value not in UnitConverter.available_units:
            raise ValueError(f"{value} as unit not supported.")
        return value

    @staticmethod
    def _check_formula(value):
        if isinstance(value, dict):
            pass
        elif isinstance(value, str):
            value = transform_str_to_dict(value)
        elif isinstance(value, list):
            value = transform_list_to_dict(value)
        else:
            raise TypeError("Could not process `formula`.")
        return value

    def _calculate_surf_energy(self, x_val, el1, el2, e_bulk, surface, e_shift):
        slab_el1 = surface["formula"][el1]
        slab_el2 = surface["formula"][el2]
        bulk_el1 = self._bulk_phase["formula"][el1]
        bulk_el2 = self._bulk_phase["formula"][el2]
        e_slab = UnitConverter.convert_units(
            surface["total_energy"], surface["energy_unit"], self.energy_unit
        )
        area = UnitConverter.convert_units(surface["area"], surface["area_unit"], self.area_unit)
        return (
            0.5
            * (
                e_slab
                - slab_el1 * (x_val + e_shift)
                - slab_el2 * (e_bulk - bulk_el1 * (x_val + e_shift)) / bulk_el2
            )
            / area
        )
