"""Classes to plot planar fields."""

# Standard library imports
import copy

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.plots.base_plot import _BasePlot
from aim2dat.units import UnitConverter


class PlanarFieldPlot(_BasePlot):
    """
    Plot scalar planar fields.
    """

    _supported_norms = {"symlog": "SymLogNorm", "log": "LogNorm"}
    _supported_plot_types = {"heatmap", "contour"}

    def __init__(self, show_x_label=True, show_y_label=True, **kwargs):
        """Initialize object."""
        _BasePlot.__init__(self, **kwargs)
        self._coordinates_unit = None
        self._values_unit = None
        self._norm = None
        self._plot_type = "heatmap"
        self.show_x_label = show_x_label
        self.show_y_label = show_y_label
        self.contour_filled = True
        self.contour_levels = None
        self.color_map = "RdBu_r"
        self.vmin = None
        self.vmax = None
        self.linthresh = 0.01

    @property
    def coordinates_unit(self):
        """Set unit of the two coordinates."""
        return self._coordinates_unit

    @coordinates_unit.setter
    def coordinates_unit(self, value):
        value = value.lower()
        if value not in UnitConverter.available_units:
            raise ValueError(f"{value} as unit not supported.")
        self._coordinates_unit = value

    @property
    def values_unit(self):
        """Set unit of the z-values."""
        return self._values_unit

    @values_unit.setter
    def values_unit(self, value):
        value = value.lower()
        if value not in UnitConverter.available_units:
            raise ValueError(f"{value} as unit not supported.")
        self._values_unit = value

    @property
    def norm(self):
        """Set norm of the z-values for matplotlib."""
        return self._norm

    @norm.setter
    def norm(self, value):
        if value not in self._supported_norms.keys():
            raise ValueError(
                f"{value} not supported. Supported norms are: "
                + ", ".join(self._supported_norms.keys())
                + "."
            )
        self._norm = value

    @property
    def plot_type(self):
        """Set plot-type."""
        return self._plot_type

    @plot_type.setter
    def plot_type(self, value):
        if value in self._supported_plot_types:
            self._plot_type = value
        else:
            raise ValueError(
                f"Plot type {value} is not supported. Supported plot types are: "
                + ", ".join(self._supported_plot_types)
                + "."
            )

    def import_field(
        self,
        data_label,
        coordinates,
        values,
        flip_lr=False,
        flip_ud=False,
        coordinates_unit=None,
        values_unit=None,
        text_labels=[],
    ):
        """
        Import field.

        Parameters
        ----------
        data_label : str
            Internal label used to plot and compare multiple data sets.
        coordinates : list
            Nested list of the coordinates.
        values : list
            List or nested list of the values.
        flip_lr : bool (optional)
            Whether to flip the field from left to right.
        flip_ud : bool (optional)
            Whether to flip the field from up to down.
        coordinates_unit : str (optional)
            Unit of coordinates.
        values_unit : str (optional)
            Unit of values.
        text_labels : list (optional)
            List of text labels.
        """
        self._check_data_label(data_label)
        text_labels = copy.deepcopy(text_labels)
        # - Check units:
        coord_factor, self._coordinates_unit = self._set_unit_conv_factor(
            coordinates_unit, self._coordinates_unit
        )
        val_factor, self._values_unit = self._set_unit_conv_factor(values_unit, self._values_unit)

        # - Create grid:
        coordinates = np.array(coordinates) * coord_factor
        x_values, y_values = zip(*coordinates)
        x_values = np.sort(np.unique(np.array(x_values)))
        x = x_values
        if flip_lr:
            x = x_values[::-1]
            x_max = np.max(x)
            for label in text_labels:
                label["x"] = x_max - label["x"]
        y_values = np.sort(np.unique(np.array(y_values)))
        y = y_values
        if flip_ud:
            y = y_values[::-1]
            y_max = np.max(y)
            for label in text_labels:
                label["y"] = y_max - label["y"]
        val_shape = (y.shape[0], x.shape[0])
        is_vector_field = False
        if isinstance(values[0], (tuple, list)):
            val_shape = (y.shape[0], x.shape[0], len(values[0]))
            is_vector_field = True
        values_grid = np.zeros(val_shape)

        # - Distribute points on grid.
        for coord, value in zip(coordinates, values):
            x_idx = np.where(x == coord[0])[0][0]
            y_idx = np.where(y == coord[1])[0][0]
            # print(x_idx, y_idx, coord)
            if is_vector_field:
                for val_idx, val0 in enumerate(value):
                    values_grid[y_idx][x_idx][val_idx] = val0
            else:
                values_grid[y_idx][x_idx] = value
        self._data[data_label] = {
            "x_values": x_values,
            "y_values": y_values,
            "z_values": values_grid,
            "is_vector_field": is_vector_field,
            "vmin": min(values),
            "vmax": max(values),
            "text_labels": [self._set_text_label(t_label) for t_label in text_labels],
        }

    def import_from_aiida_arraydata(
        self, data_label, planedata, flip_lr=False, flip_ud=False, values_unit=None, text_labels=[]
    ):
        """
        Import from aiida array data.

        Parameters
        ----------
        data_label : str
            Internal label used to plot and compare multiple data sets.
        flip_lr : bool (optional)
            Whether to flip the field from left to right.
        flip_ud : bool (optional)
            Whether to flip the field from up to down.
        values_unit : str (optional)
            Unit of values.
        text_labels : list (optional)
            List of text labels.
        """
        from aim2dat.ext_interfaces.aiida import _load_data_node

        planedata = _load_data_node(planedata)
        self.import_field(
            data_label,
            planedata.get_array("coordinates"),
            planedata.get_array("values"),
            coordinates_unit=planedata.get_attribute("coordinates_unit", None),
            flip_lr=flip_lr,
            flip_ud=flip_ud,
            values_unit=values_unit,
            text_labels=text_labels,
        )

    def _prepare_to_plot(self, data_labels, subplot_assignment):
        if self.backend == "plotly":
            print("Warning: logarithmic scales and vmin/vmax is not supported for this backend.")
        axis_label = self._set_axis_label()
        self._auto_set_axis_properties(x_label=axis_label, y_label=axis_label)
        data_sets = [[] for idx0 in range(max(subplot_assignment) + 1)]
        for data_label, subp_a in zip(data_labels, subplot_assignment):
            # for data_label in data_labels:
            data_sets[subp_a] += self._process_data_set_plot(data_label)
        return data_sets, None, None, None, None, None

    def _process_data_set_plot(self, data_label):
        data_set = self._return_data_set(data_label)
        if data_set.pop("is_vector_field"):
            print("Only scalar fields are supported so far.")
            return None

        if self.plot_type == "contour":
            data_set["filled"] = self.contour_filled
            if all(atr0 is not None for atr0 in [self.contour_levels, self.vmin, self.vmax]):
                data_set["levels"] = np.linspace(
                    data_set["vmin"], data_set["vmax"], self.contour_levels
                )
                # data_set["extend"] = "both"
            elif self.contour_levels is not None:
                data_set["levels"] = np.linspace(self.vmin, self.vmax, self.contour_levels)
            if self.norm is not None:
                if self.norm == "symlog":
                    data_set["symlog_scale"] = True
                    data_set["linthresh"] = self.linthresh
                    data_set["base"] = 10
                elif self.norm == "log":
                    data_set["log_scale"] = True
                    data_set["linthresh"] = self.linthresh
                    data_set["base"] = 10
        self._set_norm(data_set)
        text_labels = data_set.pop("text_labels", [])
        data_set["cmap"] = self.color_map
        data_set["type"] = self.plot_type
        return [data_set] + text_labels

    def _set_norm(self, data_set):
        """Set parameter for the matplotlib.colors.LogNorm or SymLogNorm."""
        if self._norm is not None:
            data_set["norm_type"] = self._supported_norms[self._norm]
            data_set["norm_args"] = {}
            if self.vmin is None:
                data_set["norm_args"]["vmin"] = data_set["vmin"]
            else:
                data_set["norm_args"]["vmin"] = self.vmin
            if self.vmax is None:
                data_set["norm_args"]["vmax"] = data_set["vmax"]
            else:
                data_set["norm_args"]["vmax"] = self.vmax
            if self._norm == "symlog":
                data_set["norm_args"]["linthresh"] = self.linthresh
            if self._norm == "log":
                data_set["norm_args"]["vmin"] = max(1e-21, data_set["norm_args"]["vmin"])
                data_set["norm_args"]["vmax"] = max(1e-21, data_set["norm_args"]["vmax"])
                print(data_set["norm_args"]["vmax"], data_set["norm_args"]["vmin"])
        del data_set["vmin"]
        del data_set["vmax"]

    def _set_axis_label(self):
        """Set axis labels."""
        return (
            UnitConverter._available_units[self.coordinates_unit].capitalize()
            + f" in {UnitConverter.plot_labels[self.coordinates_unit]}"
        )

    @staticmethod
    def _set_text_label(text_label):
        """Set text label."""
        if not isinstance(text_label, dict):
            raise TypeError("Text labels need to be of type dict.")
        for key0 in ["x", "y", "label"]:
            if key0 not in text_label:
                raise ValueError(f"Key '{key0}' needs to be in text label.")
        text_label["type"] = "text"
        text_label["ha"] = "center"
        text_label["va"] = "center"
        # text_label["weight"] = "bold"
        return text_label

    @staticmethod
    def _set_unit_conv_factor(input_unit, plot_unit):
        """Set unit conversion factor."""
        conv_factor = 1.0
        if plot_unit is None:
            plot_unit = input_unit
        elif input_unit is not None:
            conv_factor = UnitConverter.convert_units(1.0, input_unit, plot_unit)
        return conv_factor, plot_unit
