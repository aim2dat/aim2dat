"""
Module to plot the partial charges.
"""

# Third party library imports
from statistics import mean
import copy

# Internal library imports
from aim2dat.plots.base_plot import _BasePlot
from aim2dat.ext_interfaces import _return_ext_interface_modules


def _validate_plot_type(value):
    if not isinstance(value, str):
        raise TypeError("`pc_plot_type` needs to be of type str.")
    if value not in ["scatter", "bar"]:
        raise ValueError("`pc_plot_type` must contain 'scatter' or 'bar'.")
    return value


class PartialChargesPlot(_BasePlot):
    """
    Plot the partial charges.
    """

    _object_title = "Partial Charge Plot"

    def __init__(
        self, custom_linestyles=["solid"], pc_plot_type="scatter", pc_plot_order=[], **kwargs
    ):
        """Initialize class."""
        _BasePlot.__init__(self, custom_linestyles=custom_linestyles, **kwargs)
        self.pc_plot_type = pc_plot_type
        self.pc_plot_order = pc_plot_order

    @property
    def pc_plot_type(self):
        """
        str: plot type of the partial charge data sets,
             supported options are ``'scatter'``, ``'bar'``.
        """
        return self._pc_plot_type

    @pc_plot_type.setter
    def pc_plot_type(self, value):
        self._pc_plot_type = _validate_plot_type(value)

    @property
    def pc_plot_order(self):
        """
        list: List of plot assignments to order the plotted data.
        """
        return self._pc_plot_order

    @pc_plot_order.setter
    def pc_plot_order(self, value):
        self._pc_plot_order = value

    def import_partial_charges(
        self,
        data_label,
        partial_charges,
        valence_electrons,
        plot_label,
        x_label,
        custom_kind_dict=None,
    ):
        """
        Import partial charges.

        Parameters
        ----------
        data_label : str
            Internal label of the data set.
        partial_charges : list of dict
            List of dict containing elements and populations, e.g.
            ``[{"element": "H", "population": 0.9}, {"element": "O", "population": 7.4}]``
        valence_electrons : dict
            Valence electrons used for each element to calculate partial charges.
        plot_label : str
            Label in legend on how to plot data.
        x_label : str
            Label on x axes to sort data.
        custom_kind_dict : dict or None (optional)
            Group the partial charges by taking the average value and put custom labels, e.g.
            ``{"label_1": [0, 1, 2], "label_2": [[3, 4], [5]]}``.
            In case a nested list is given as a value the sum of the average value(s) is
            calculated.
        """
        self._check_data_label(data_label)
        if valence_electrons:
            partial_charges = copy.deepcopy(partial_charges)
            for population in partial_charges:
                population["charge"] = (
                    valence_electrons[population["element"]] - population["population"]
                )

        mean_charge = self._process_charge(partial_charges, custom_kind_dict)
        self._data[data_label] = {
            "plot_label": plot_label,
            "x_label": x_label,
            "mean_charge": mean_charge,
        }

    def import_from_aiida_list(
        self,
        data_label,
        pcdata,
        plot_label,
        x_label,
        custom_kind_dict=None,
    ):
        """
        Import partial charges.

        Parameters
        ----------
        data_label : str
            Internal label of the data set.
        pcdata : aiida.orm.list
            AiiDA data node containing the partial charges as list of dictionaries.
        plot_label : str
            Label in legend on how to plot data.
        x_label : str
            Label on x axes to sort data.
        custom_kind_dict : dict or None (optional)
            Group the partial charges by taking the average value and put custom labels, e.g.
            ``{"label_1": [0, 1, 2], "label_2": [[3, 4], [5]]}``.
            In case a nested list is given as a value the sum of the average value(s) is
            calculated.
        """
        self._check_data_label(data_label)
        backend_module = _return_ext_interface_modules("aiida")
        partial_charges = backend_module._load_data_node(pcdata).get_list()
        valence_electrons = None
        self.import_partial_charges(
            data_label,
            partial_charges,
            valence_electrons,
            plot_label,
            x_label,
            custom_kind_dict,
        )

    def _process_charge(self, partial_charges, custom_kind_dict):
        elements = list(dict.fromkeys([el["element"] for el in partial_charges]).keys())
        mean_charge = {}
        if custom_kind_dict:
            for el, indices in custom_kind_dict.items():
                if all(isinstance(i, list) for i in indices):
                    mean_charge[el] = 0.0
                    for idx in indices:
                        mean_charge[el] += mean([partial_charges[i]["charge"] for i in idx])
                elif any(isinstance(i, list) for i in indices):
                    raise ValueError(
                        "`custom_kind_dict` must contain dict of lists with int or "
                        + "lists of lists with int."
                    )
                else:
                    mean_charge[el] = mean([partial_charges[idx]["charge"] for idx in indices])
        else:
            for el in elements:
                mean_charge[el] = mean(
                    [charge["charge"] for charge in partial_charges if el in charge["element"]]
                )
        return mean_charge

    def _prepare_to_plot(self, data_labels, subplot_assignment):
        all_charge = {}
        x_tick_labels = []
        for data_label in data_labels:
            all_charge[data_label] = self._return_data_set(data_label)
            if all_charge[data_label]["x_label"] not in x_tick_labels:
                x_tick_labels.append(all_charge[data_label]["x_label"])
        x_ticks = [*range(0, len(x_tick_labels))]
        data_sets = self._generate_data_sets_for_plot(
            all_charge, x_tick_labels, subplot_assignment
        )
        self._auto_set_axis_properties(y_label=r"Partial charge in $e$")
        return data_sets, x_ticks, None, x_tick_labels, None, None

    def _generate_data_sets_for_plot(self, all_charge, x_tick_labels, subplot_assignment):
        data_labels = copy.deepcopy(self.pc_plot_order)
        plot_labels = []
        mean_charge = [all_charge[data]["mean_charge"] for data in all_charge]
        for label in mean_charge:
            for el in label.keys():
                if el not in data_labels:
                    data_labels.append(el)
        for data in all_charge:
            if all_charge[data]["plot_label"] not in plot_labels:
                plot_labels.append(all_charge[data]["plot_label"])
        data_sets, max_index = self._generate_data(
            data_labels, plot_labels, all_charge, x_tick_labels
        )
        if max(subplot_assignment) > 0 or len(subplot_assignment) != len(all_charge):
            new_data_sets = [[] for idx0 in range(max(subplot_assignment) + 1)]
            used = set()
            for idx, new_idx in enumerate(subplot_assignment):
                if new_idx in used:
                    for line in data_sets[idx]:
                        max_index += 1
                        line["color"] = max_index
                        line["marker"] = max_index
                        line["linestyle"] = max_index
                        line["label"] = line["label"] + "2"
                new_data_sets[new_idx] += data_sets[idx]
                used.add(new_idx)
            data_sets = new_data_sets
        return data_sets

    def _generate_data(self, data_labels, plot_labels, all_charge, x_tick_labels):
        max_index = 0
        data_sets = [[] for idx0 in range(len(data_labels))]
        for idx, label in enumerate(data_labels):
            for plot_label in plot_labels:
                plot_idx = plot_labels.index(plot_label)
                if plot_idx > max_index:
                    max_index = plot_idx
                y_values_height = [
                    all_charge[data]["mean_charge"].get(label)
                    for data in all_charge
                    if plot_label == all_charge[data]["plot_label"]
                ]
                x_values = [
                    x_tick_labels.index(all_charge[data]["x_label"])
                    for data in all_charge
                    if plot_label == all_charge[data]["plot_label"]
                ]
                data_set = {
                    "label": plot_label,
                    "type": self._pc_plot_type,
                    "color": plot_idx,
                    "marker": plot_idx,
                    "linestyle": plot_idx,
                }
                if self.pc_plot_type == "bar":
                    data_set.update(
                        {
                            "x_values": [
                                -0.45 + (0.5 + plot_idx) / len(plot_labels) * 0.9 + i
                                for i in x_values
                            ],
                            "heights": [
                                value if value is not None else 0 for value in y_values_height
                            ],
                            "width": 0.9 / len(plot_labels),
                        }
                    )
                elif self.pc_plot_type == "scatter":
                    data_set.update(
                        {
                            "x_values": x_values,
                            "y_values": y_values_height,
                        }
                    )

                data_sets[idx].append(data_set)
        return data_sets, max_index
