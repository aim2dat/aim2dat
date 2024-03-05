"""Mixin classes with additional features for plots."""

# Standard library imports
import math

# Third party library imports
import numpy as np
from scipy.signal import find_peaks

# Internal library imports
import aim2dat.fct.smearing as fct_smearing


def _add_line(
    extra_lines,
    x,
    y_min,
    y_max,
    y,
    x_min,
    x_max,
    color,
    line_style,
    scaled,
    foreground,
    subplot_assignment,
    data_label,
):
    if x is None:
        line_dict = {"y": y, "xmin": x_min, "xmax": x_max, "scaled": scaled, "type": "hline"}
    else:
        line_dict = {"x": x, "ymin": y_min, "ymax": y_max, "scaled": scaled, "type": "vline"}
    if color is not None:
        line_dict["color"] = color
    if line_style is not None:
        line_dict["linestyle"] = line_style
    if data_label is not None:
        subplot_assignment = [data_label]
    elif subplot_assignment is None:
        subplot_assignment = ["all"]
    if not isinstance(subplot_assignment, (list, tuple)):
        subplot_assignment = [subplot_assignment]
    for subp_a in subplot_assignment:
        if subp_a in extra_lines:
            extra_lines[subp_a].append(line_dict)
        else:
            extra_lines[subp_a] = [line_dict]


class _BaseExtras:
    def remove_additional_plot_elements(self):
        """Remove all added plot elements."""
        for extra in ["_plot_extras_lines_background", "_plot_extras_lines_foreground"]:
            if hasattr(self, extra):
                setattr(self, extra, {})


class _VLineMixin(_BaseExtras):
    def add_vline(
        self,
        x,
        y_min,
        y_max,
        color=None,
        line_style=None,
        scaled=False,
        foreground=False,
        subplot_assignment=None,
        data_label=None,
    ):
        """
        Add a vertical line to the plot.

        Parameters
        ----------
        x : float
            x-position of the line.
        y_min : float
            Bottom y-position of the line.
        y_max : float
            Top y-position of the line.
        color : str or None
            Color of the line.
        line_style : str or None
            Line style of the line.
        scaled : bool
            Whether the input is given in scaled positions.
        foreground : bool
            Whether to plot the line in front of the other elements.
        subplot_assignment : list or None
            Assignment of the line to individual subplots.
        """
        attr_suffix = "background"
        if foreground:
            attr_suffix = "foreground"
        if not hasattr(self, "_plot_extras_lines_" + attr_suffix):
            setattr(self, "_plot_extras_lines_" + attr_suffix, {})
        _add_line(
            getattr(self, "_plot_extras_lines_" + attr_suffix),
            x,
            y_min,
            y_max,
            None,
            None,
            None,
            color,
            line_style,
            scaled,
            foreground,
            subplot_assignment,
            data_label,
        )


class _HLineMixin(_BaseExtras):
    def add_hline(
        self,
        y,
        x_min,
        x_max,
        color=None,
        line_style=None,
        scaled=False,
        foreground=False,
        subplot_assignment=None,
        data_label=None,
    ):
        """
        Add a vertical line to the plot.

        Parameters
        ----------
        y : float
            y-position of the line.
        x_min : float
            Bottom x-position of the line.
        x_max : float
            Top x-position of the line.
        color : str or None
            Color of the line.
        line_style : str or None
            Line style of the line.
        scaled : bool
            Whether the input is given in scaled positions.
        foreground : bool
            Whether to plot the line in front of the other elements.
        subplot_assignment : list or None
            Assignment of the line to individual subplots.
        """
        attr_suffix = "background"
        if foreground:
            attr_suffix = "foreground"
        if not hasattr(self, "_plot_extras_lines_" + attr_suffix):
            setattr(self, "_plot_extras_lines_" + attr_suffix, {})
        _add_line(
            getattr(self, "_plot_extras_lines_" + attr_suffix),
            None,
            None,
            None,
            y,
            x_min,
            x_max,
            color,
            line_style,
            scaled,
            foreground,
            subplot_assignment,
            data_label,
        )


class _SmearingMixin:
    @property
    def smearing_method(self):
        """
        str: Method used to smear out the functions. Supported options are ``'gaussian'`` and
        ``'lorentzian'``.
        """
        return getattr(self, "_smearing_method", None)

    @smearing_method.setter
    def smearing_method(self, value):
        if value not in fct_smearing.AVAILABLE_SMEARING_METHODS:
            raise ValueError(
                f"Smearing method '{value}' is not supported. Available methods are: '"
                + "', '".join(fct_smearing.AVAILABLE_SMEARING_METHODS.keys())
                + "'."
            )
        self._smearing_method = value

    @property
    def smearing_delta(self):
        """
        float or None: Spacing between two values. If set to ``None`` the original function is
        used.
        """
        return getattr(self, "_smearing_delta", None)

    @smearing_delta.setter
    def smearing_delta(self, value):
        if value is not None:
            self._smearing_delta = float(value)

    @property
    def smearing_sigma(self):
        """
        float: Sigma value of the smearing distribution.
        """
        return getattr(self, "_smearing_sigma", None)

    @smearing_sigma.setter
    def smearing_sigma(self, value):
        self._smearing_sigma = float(value)

    def _apply_smearing(self, x_values, y_values):
        def sum_y_values(x_values, y_values, x_min, nr_bins, smearing_delta):
            y_smear = np.zeros(nr_bins)
            for x_value, y_value in zip(x_values, y_values):
                bin0 = round((x_value - x_min) / self.smearing_delta)
                y_smear[bin0] += y_value
            return y_smear

        if self.smearing_method is None:
            raise ValueError("`smearing_method` needs to be set.")
        if self.smearing_sigma is None:
            raise ValueError("`smearing_sigma` needs to be set.")

        y_values = np.array(y_values)
        if self.smearing_delta is not None:
            x_min = np.min(x_values) - 10.0 * self.smearing_sigma * self.smearing_delta
            X_max = np.max(x_values) + 10.0 * self.smearing_sigma * self.smearing_delta
            nr_bins = math.ceil((X_max - x_min) / self.smearing_delta)
            x_smear = np.linspace(x_min, X_max, nr_bins)
            if len(y_values.shape) == 1:
                y_values = sum_y_values(x_values, y_values, x_min, nr_bins, self.smearing_delta)
            else:
                y_smear = np.zeros((y_values.shape[0], nr_bins))
                for idx0, y_val in enumerate(y_values):
                    y_smear[idx0] = sum_y_values(
                        x_values, y_val, x_min, nr_bins, self.smearing_delta
                    )
                y_values = y_smear
            x_values = x_smear
        if len(y_values.shape) == 1:
            y_values = fct_smearing.apply_smearing(
                y_values, sigma=self.smearing_sigma, method=self.smearing_method
            )
        else:
            for idx0, y_val in enumerate(y_values):
                y_values[idx0] = fct_smearing.apply_smearing(
                    y_val, sigma=self.smearing_sigma, method=self.smearing_method
                )
        return x_values, y_values


class _PeakDetectionMixin:
    @property
    def peaks(self):
        return getattr(self, "_peaks", {})

    @peaks.setter
    def peaks(self, value):
        if len(value) != 2:
            raise ValueError(
                "`peaks` needs to be set by tuple consisting of data label and peaks."
            )
        if not hasattr(self, "_peaks"):
            self._peaks = {}
        self._peaks[value[0]] = value[1]

    @property
    def peak_height(self):
        return getattr(self, "_peak_height", None)

    @peak_height.setter
    def peak_height(self, value):
        self._peak_height = value

    @property
    def peak_threshold(self):
        return getattr(self, "_peak_threshold", None)

    @peak_threshold.setter
    def peak_threshold(self, value):
        self._peak_threshold = value

    @property
    def peak_distance(self):
        return getattr(self, "_peak_distance", None)

    @peak_distance.setter
    def peak_distance(self, value):
        self._peak_distance = value

    @property
    def peak_prominence(self):
        return getattr(self, "_peak_prominence", None)

    @peak_prominence.setter
    def peak_prominence(self, value):
        self._peak_prominence = value

    @property
    def peak_width(self):
        return getattr(self, "_peak_width", None)

    @peak_width.setter
    def peak_width(self, value):
        self._peak_width = value

    @property
    def peak_wlen(self):
        return getattr(self, "_peak_wlen", None)

    @peak_wlen.setter
    def peak_wlen(self, value):
        self._peak_wlen = value

    @property
    def peak_rel_height(self):
        return getattr(self, "_peak_rel_height", 0.5)

    @peak_rel_height.setter
    def peak_rel_height(self, value):
        self._peak_rel_height = value

    @property
    def peak_plateau_size(self):
        return getattr(self, "_peak_plateau_size", None)

    @peak_plateau_size.setter
    def peak_plateau_size(self, value):
        self._peak_plateau_size = value

    @property
    def peak_color(self):
        return getattr(self, "_peak_color", "red")

    @peak_color.setter
    def peak_color(self, value):
        self._peak_color = value

    @property
    def peak_y_min(self):
        return getattr(self, "_peak_y_min", 0)

    @peak_y_min.setter
    def peak_y_min(self, value):
        self._peak_y_min = value

    @property
    def peak_line_style(self):
        return getattr(self, "_peak_line_style", "solid")

    @peak_line_style.setter
    def peak_line_style(self, value):
        self._peak_line_style = value

    @property
    def peak_max_factor(self):
        return getattr(self, "_peak_max_factor", 0.9)

    @peak_max_factor.setter
    def peak_max_factor(self, value):
        self._peak_max_factor = value

    def _find_peaks(self, x_values, y_values, data_label):
        peaks, _ = find_peaks(
            y_values,
            height=self.peak_height,
            threshold=self.peak_threshold,
            distance=self.peak_distance,
            prominence=self.peak_prominence,
            width=self.peak_width,
            wlen=self.peak_wlen,
            rel_height=self.peak_rel_height,
            plateau_size=self.peak_plateau_size,
        )

        self.peaks = (
            data_label,
            {
                "x_values": [float(x_values[peak]) for peak in peaks],
                "y_values": [float(y_values[peak]) for peak in peaks],
            },
        )

        for peak in peaks:
            self.add_vline(
                x=x_values[peak],
                y_min=self.peak_y_min,
                y_max=self.peak_max_factor * y_values[peak],
                line_style=self.peak_line_style,
                color=self.peak_color,
                data_label=data_label,
            )
