"""Plot class for radial distribution functions."""

# Third party library imports
import numpy as np

# Internal library imports:
from aim2dat.plots.base_plot import _BasePlot
from aim2dat.units import UnitConverter


class PartialRDFPlot(_BasePlot):
    """
    Plot the partial radial distribution function.
    """

    def __init__(self, custom_linestyles=["solid", "dashed", "dotted", "dashdot"], **kwargs):
        """Initialize object."""
        _BasePlot.__init__(self, custom_linestyles=custom_linestyles, **kwargs)
        self._el_pairs_color_indices = {}
        self._x_unit = None

    @property
    def x_unit(self):
        """Set unit of the x coordinate."""
        return self._x_unit

    @x_unit.setter
    def x_unit(self, value):
        value = value.lower()
        if value not in UnitConverter.available_units:
            raise ValueError(f"{value} as unit not supported.")
        self._x_unit = value

    def import_ffingerprint(
        self,
        data_label,
        bins,
        fingerprints,
        x_unit=None,
    ):
        """
        Import F-Fingerprint functions.

        Parameters
        ----------
        data_label : str
            Internal label used to plot and compare multiple data sets.
        bins : list
            Bins of the distance.
        fingerprints : dict
            Dictionary with the keys being the element pairs as tuples and the fingerprint
            functions as values.
        x_unit : str or None (optional)
            Unit of the x-axis.
        """
        self._check_data_label(data_label)
        data_sets = []

        coord_factor, self._x_unit = self._set_unit_conv_factor(x_unit, self._x_unit)

        bins = np.array(bins) * coord_factor

        for el_pair, fingerprint in fingerprints.items():
            if el_pair not in self._el_pairs_color_indices:
                self._el_pairs_color_indices[el_pair] = (
                    max(list(self._el_pairs_color_indices.values()) + [-1]) + 1
                )
            data_sets.append(
                {
                    "x_values": bins,
                    "y_values": fingerprint,
                    "type": "scatter",
                    "color": self._el_pairs_color_indices[el_pair],
                    "legendgrouptitle_text": el_pair[0] + "-" + el_pair[1],
                    "legendgroup": el_pair[0] + "-" + el_pair[1],
                    "label": data_label,
                    "group_by": "color",
                }
            )
        self._data[data_label] = data_sets

    def _prepare_to_plot(self, data_labels, subplot_assignment):
        data_sets_plot = [[] for idx0 in range(max(subplot_assignment) + 1)]
        for idx, (data_label, subp_a) in enumerate(zip(data_labels, subplot_assignment)):
            data_sets = self._return_data_set(data_label)
            for data_set in data_sets:
                data_set["linestyle"] = idx
                if len(data_labels) == 1:
                    data_set["label"] = data_set["legendgroup"]
                    del data_set["legendgroup"]
                    del data_set["group_by"]
                    del data_set["legendgrouptitle_text"]
            data_sets_plot[subp_a] += data_sets

        x_label = f"Distance ({self._x_unit})"
        if self._x_unit in ["angstrom", "ang"]:
            x_label = r"Distance ($\mathrm{\AA}$)"
        self._auto_set_axis_properties(x_label=x_label)
        return data_sets_plot, None, None, None, None, None

    @staticmethod
    def _set_unit_conv_factor(input_unit, plot_unit):
        """Set unit conversion factor."""
        conv_factor = 1.0
        if plot_unit is None:
            plot_unit = input_unit
        elif input_unit is not None:
            conv_factor = UnitConverter.convert_units(1.0, input_unit, plot_unit)
        return conv_factor, plot_unit
