"""Plots to analyse the output of a phonopy calculations."""

# Internal library imports
from aim2dat.plots.base_plot import _BasePlot


class ThermalPropertiesPlot(_BasePlot):
    """
    Class to plot the thermal properties.

    """

    _supported_properties = {
        "free_energy": "Free energy",
        "entropy": "Entropy",
        "heat_capacity": "Heat capacity",
    }

    def __init__(
        self,
        plot_properties=["free_energy", "entropy", "heat_capacity"],
        x_label=[None, "Temperature in K"],
        y_label=["Free energy in kJ/mol", "Entropy and $C_V$ in J/K/mol"],
        subplot_nrows=2,
        subplot_ncols=1,
        subplot_sharex=True,
        subplot_tight_layout=True,
        **kwargs,
    ):
        """Initialize object."""
        self._plot_properties = plot_properties
        _BasePlot.__init__(
            self,
            x_label=x_label,
            y_label=y_label,
            subplot_nrows=subplot_nrows,
            subplot_ncols=subplot_ncols,
            subplot_sharex=subplot_sharex,
            subplot_tight_layout=subplot_tight_layout,
            **kwargs,
        )

    @property
    def plot_properties(self):
        """list or str: Plot properties."""
        return self._plot_properties

    @plot_properties.setter
    def plot_properties(self, value):
        if not isinstance(value, (tuple, list)):
            value = [value]
        for val in value:
            if val not in list(self._supported_properties.keys()):
                raise ValueError(f"'{val}' is not supported.")
        self._plot_properties = value

    def import_thermal_properties(
        self, data_label, temperatures, free_energy, entropy, heat_capacity
    ):
        """
        Import data set.

        Parameters
        ----------
        data_label : str
            Internal label of the data set.
        temperatures : list
            List of temperatures.
        free_energy : list
            List of free energy values.
        entropy : list
            List of entropy values.
        heat_capacity : list
            List of heat capacity values.
        """
        self._check_data_label(data_label)
        self._data[data_label] = {
            "free_energy": {
                "x_values": temperatures,
                "y_values": free_energy,
                "type": "scatter",
                "label": data_label,
            },
            "entropy": {
                "x_values": temperatures,
                "y_values": entropy,
                "type": "scatter",
                "label": data_label,
                "linestyle": "dashed",
            },
            "heat_capacity": {
                "x_values": temperatures,
                "y_values": heat_capacity,
                "type": "scatter",
                "label": data_label,
            },
        }

    def _prepare_to_plot(self, data_labels, subplot_assignment):
        data_sets_plot1 = []
        data_sets_plot2 = []
        for idx0, data_label in enumerate(data_labels):
            # distinguish the data sets:
            data_sets = self._return_data_set(data_label)
            for prop_label, data_set in data_sets.items():
                if prop_label in self.plot_properties:
                    data_set["color"] = idx0
                    if prop_label == "free_energy":
                        if len(data_labels) == 1:
                            data_set["label"] = self._supported_properties[prop_label]
                        data_sets_plot1.append(data_set)
                    else:
                        if len(data_labels) > 1:
                            data_set["group_by"] = "color"
                            data_set["legendgroup"] = data_set["label"]
                            data_set["legendgrouptitle_text"] = data_set["label"]
                            data_set["label"] = self._supported_properties[prop_label]
                        else:
                            data_set["label"] = self._supported_properties[prop_label]
                        data_sets_plot2.append(data_set)
        if len(data_sets_plot1) == 0:
            data_sets_plot = data_sets_plot2
        elif len(data_sets_plot2) == 0:
            data_sets_plot = data_sets_plot1
        else:
            data_sets_plot = [data_sets_plot1, data_sets_plot2]
        return data_sets_plot, None, None, None, None, None


class QHAPlot(_BasePlot):
    """Class to plot the quasi harmonic properties."""

    _supported_plot_properties = [
        "volume_temperature",
        "thermal_expansion",
        "gibbs_temperature",
        "bulk_modulus_temperature",
        "gruneisen_temperature",
    ]

    def __init__(
        self,
        plot_properties=["volume_temperature", "thermal_expansion"],
        selected_temperatures=(0.0, 200.0, 400.0, 600.0, 800.0, 1000.0),
        **kwargs,
    ):
        """Initialize object."""
        _BasePlot.__init__(self, **kwargs)
        self.plot_properties = plot_properties
        self.selected_temperatures = selected_temperatures

    @property
    def plot_properties(self):
        """
        list : Properties that are plotted.
        """
        return self._plot_properties

    @plot_properties.setter
    def plot_properties(self, value):
        if isinstance(value, str):
            value = [value]
        elif not isinstance(value, (list, tuple)):
            raise TypeError("`plot_properties` need to be of type str/list/tuple.")
        for val in value:
            if val not in self._supported_plot_properties:
                raise ValueError(f"'{val}' is not supported.")
        self._plot_properties = value

    def import_qha_properties(
        self,
        data_label,
        temperatures,
        volume_temperature=None,
        bulk_modulus_temperature=None,
        thermal_expansion=None,
        helmholtz_volume=None,
        volumes=None,
    ):
        """
        Import data set.

        Parameters
        ----------
        data_label : str
            Internal label of the data set.
        temperatures : list
            List of temperatures.
        volume_temperature : list
            List of volumes.
        bulk_modulus_temperature : list
            List of bulk moduli.
        thermal_expansion : list
            List of thermal expansion values.
        helmholtz_volume : list
            List of helmholtz volumes.
        volumes : list
            List of volumes.
        """
        self._check_data_label(data_label)
        data_set = {"temperatures": temperatures}
        if volume_temperature is not None:
            data_set["volume_temperature"] = volume_temperature
        if bulk_modulus_temperature is not None:
            data_set["bulk_modulus_temperature"] = bulk_modulus_temperature
        if thermal_expansion is not None:
            data_set["thermal_expansion"] = thermal_expansion
        if helmholtz_volume is not None:
            data_set["helmholtz_volume"] = helmholtz_volume
        if volumes is not None:
            data_set["volumes"] = volumes
        self._data[data_label] = data_set

    def _prepare_to_plot(self, data_labels, subplot_assignment):
        if isinstance(data_labels, str):
            data_labels = [data_labels]
        data_sets_plot = [[] for idx0 in self.plot_properties]
        for data_label in data_labels:
            data_set = self._return_data_set(data_label)
            for prop_idx, prop in enumerate(self.plot_properties):
                process_f = getattr(self, "_process_" + prop)
                data_sets_plot[prop_idx] += process_f(data_label, data_set)
        if len(data_sets_plot) == 1:
            data_sets_plot = data_sets_plot[0]
        return data_sets_plot, None, None, None, None, None

    def _process_volume_temperature(self, data_label, data_set):
        return self._process_scatter_data_set(data_label, data_set, "volume_temperature")

    def _process_thermal_expansion(self, data_label, data_set):
        return self._process_scatter_data_set(data_label, data_set, "thermal_expansion")

    def _process_gibbs_volume(self, data_label, data_set):
        return self._process_scatter_data_set(data_label, data_set, "gibbs_volume")

    def _process_bulk_modulus_temperature(self, data_label, data_set):
        return self._process_scatter_data_set(data_label, data_set, "bulk_modulus_temperature")

    def _process_gruneisen_temperature(self, data_label, data_set):
        return self._process_scatter_data_set(data_label, data_set, "gruneisen_temperature")

    @staticmethod
    def _process_scatter_data_set(data_label, data_set, plot_property):
        data_set_plot = []
        if plot_property in data_set:
            data_set_plot.append(
                {
                    "x_values": data_set["temperatures"],
                    "y_values": data_set[plot_property],
                    "label": data_label,
                    "type": "scatter",
                }
            )
        return data_set_plot
