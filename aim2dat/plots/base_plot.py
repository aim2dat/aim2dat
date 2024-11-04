"""Parent classes for plots."""

# Standard library imports
import importlib
import os
from types import MethodType
import copy
import abc

# Third party library imports
from matplotlib.colors import is_color_like
import numpy as np

# Internal library imports
import aim2dat.utils.print as utils_print
from aim2dat.ext_interfaces.import_opt_dependencies import _check_package_dependencies

style_sheet_path = "aim2dat/plots/matplotlib_style_sheets/custom_settings.mplstyle"
cwd = os.path.dirname(__file__)


def _validate_custom_colors(colors_list):
    if not isinstance(colors_list, (list, tuple)):
        raise TypeError("`custom_colors` must be a list/tuple of colors.")
    for color in colors_list:
        if not is_color_like(color):
            raise ValueError(f"The color '{color}' has the wrong format.")
    return tuple(colors_list)


def _validate_xy_label(label_type, value):
    if isinstance(value, str) or value is None:
        return value
    elif isinstance(value, (list, tuple)) and all(
        isinstance(val0, str) or val0 is None for val0 in value
    ):
        return tuple(value)
    else:
        raise TypeError(
            f"`{label_type}` must be of type str or a list/tuple consisting of str values."
        )


def _validate_xy_range(range_type, value):
    def check_range_tuple(value):
        if value is None:
            return value
        elif len(value) == 2 and all(
            isinstance(val0, (int, float)) or val0 is None for val0 in value
        ):
            return tuple(value)
        else:
            raise TypeError(
                f"`{range_type}` must be a nested list/tuple or a list/tuple of two numbers."
            )

    if value is None:
        return value
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"`{range_type}` must be `None` or of type list/tuple.")
    if value[0] is None or isinstance(value[0], (list, tuple)):
        return tuple(check_range_tuple(val0) for val0 in value)
    else:
        return check_range_tuple(value)


def _check_plot_param_single_values(values, n_subplots):
    output = []
    for val in values:
        if isinstance(val, (list, tuple)):
            val = list(val)
            while len(val) < n_subplots:
                val.append(val[-1])
            output.append(val[:n_subplots])
        else:
            output.append([val] * n_subplots)
    return tuple(output)


def _check_plot_param_list_values(values, n_subplots):
    output = []
    for val in values:
        if val is not None and (val[0] is None or isinstance(val[0], (list, tuple))):
            val = list(val)
            while len(val) < n_subplots:
                val.append(val[-1])
            output.append(val[:n_subplots])
        else:
            output.append([val] * n_subplots)
    return tuple(output)


class _BasePlot(abc.ABC):
    _object_title = "Base Plot"
    _allowed_backends = {
        "matplotlib": [("matplotlib", "3.6.0", None)],
        "plotly": [("plotly", None, None)],
    }
    _backend_function_list = [
        "_create_figure",
        "_finalize_plot",
        "_plot2d",
        "_set_subplot_parameters",
        "_create_secondary_axis",
        "_create_legend",
        "_create_colorbar",
        "_add_scatter",
        "_add_bar",
        "_add_heatmap",
        "_add_contour",
        "_add_vline",
        "_add_hline",
        "_add_text",
    ]
    _default_colors = [f"C{idx}" for idx in range(10)]
    _default_linestyles = ["solid", "dashed", "dotted", "dashdot", "solid", "solid"]
    _default_linewidths = [None]
    _default_markers = ["*", "o", ">", "s", "v", "h", "D"]
    _default_hatch = []
    _default_alpha = [0.5]

    def __init__(self, **kwargs):
        """Initialize class."""
        # Internal variables:
        self._figure = None
        self._axes = None
        self._data = {}

        # Backend options:
        self.backend = "matplotlib"
        # We have to see how that work with plotly and other libraries..
        self._style_sheet = cwd + "/matplotlib_style_sheets/custom_settings.mplstyle"

        # General:
        self._ratio = (7, 7)
        self._equal_aspect_ratio = False
        self._store_path = "./"
        self._store_plot = False
        self._show_plot = False
        self._show_legend = False
        self._show_grid = False
        self._show_colorbar = False
        self._x_label = None
        self._y_label = None
        self._x_range = None
        self._y_range = None

        # Legend:
        self._legend_loc = 1
        self._legend_bbox_to_anchor = (1, 1)
        self._legend_ncol = 1
        self._legend_sort_entries = False

        # Customization:
        self._custom_colors = None
        self._custom_linestyles = None
        self._custom_linewidths = None
        self._custom_markers = None
        self._custom_hatch = None
        self._custom_alpha = None
        self._custom_xticks = None
        self._custom_xticklabels = None
        self._custom_yticks = None
        self._custom_yticklabels = None
        # self.custom_legend_objects = None

        # Multiple plot:
        self._subplot_nrows = 1
        self._subplot_ncols = 1
        self._subplot_hspace = None
        self._subplot_wspace = None
        self._subplot_sharex = False
        self._subplot_sharey = False
        self._subplot_sup_title = None
        self._subplot_sup_x_label = None
        self._subplot_sup_y_label = None
        self._subplot_share_legend = False
        self._subplot_share_colorbar = False
        self._subplot_adjust = {}
        self._subplot_align_ylabels = False
        self._subplot_tight_layout = False
        self._subplot_gridspec_values = None
        self._subplot_gf_x = 1
        self._subplot_gf_y = 1
        self._subplot_center_last_row = False

        # auto set axis properties:
        self._auto_set_x_label = True
        self._auto_set_y_label = True

        for attr, val in kwargs.items():
            self.__setattr__(attr, val)

    def __str__(self):
        output_str = utils_print._print_title(self._object_title)
        output_str += "\n"
        output_str += utils_print._print_list("Data labels:", list(self._data.keys()))
        output_str += "\n"
        output_str += self._print_extra_properties()
        output_str += utils_print._print_hline()
        return output_str

    @property
    def data_labels(self):
        """
        list: List of labels for all data sets.
        """
        return tuple(self._data.keys())

    # Backend properties:
    @property
    def backend(self):
        """
        str: used backend library to plot the data. Supported values are ``"matplotlib"`` and
        ``"plotly"``.
        """
        return self._backend

    @backend.setter
    def backend(self, value):
        if value in self._allowed_backends:
            # Check if library is installed and import backend:
            _check_package_dependencies(self._allowed_backends[value])
            backend_module = importlib.import_module("aim2dat.plots." + value + "_backend")
            # Load all methods and add them as bounded methods to class:
            for fct_name in self._backend_function_list:
                self.__setattr__(fct_name, MethodType(getattr(backend_module, fct_name), self))
            # Set _backend string:
            self._backend = value
        else:
            raise ValueError(f"Backend '{value}' is not supported.")

    @property
    def style_sheet(self):
        """str: Custom matplotlib style sheet."""
        return self._style_sheet

    @style_sheet.setter
    def style_sheet(self, value):
        self._style_sheet = value

    # General properties:
    @property
    def ratio(self):
        """tuple or list: Length-to-width ratio of the plot given as a tuple of two numbers."""
        return self._ratio

    @ratio.setter
    def ratio(self, value):
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 2
            or any(not isinstance(val0, (int, float)) for val0 in value)
        ):
            raise TypeError("`ratio` must be a list/tuple consisting of two numbers.")
        self._ratio = tuple(value)

    @property
    def equal_aspect_ratio(self):
        """bool: Set equal aspect ratio of the plot(s)."""
        return self._equal_aspect_ratio

    @equal_aspect_ratio.setter
    def equal_aspect_ratio(self, value):
        if not isinstance(value, bool):
            raise TypeError("`equal_aspect_ratio` needs to be of type bool.")
        self._equal_aspect_ratio = value

    @property
    def store_path(self):
        """str: Path of the folder to store the plot. The default value is ``./``."""
        return self._store_path

    @store_path.setter
    def store_path(self, value):
        if value is None:
            value = "./"
        self._store_path = value

    @property
    def store_plot(self):
        """bool: Store plot. The default value is ``False``."""
        return self._store_plot

    @store_plot.setter
    def store_plot(self, value):
        if not isinstance(value, bool):
            raise TypeError("`store_plot` needs to be of type bool.")
        self._store_plot = value

    @property
    def show_plot(self):
        """bool: Show plot. The default value is ``False``."""
        return self._show_plot

    @show_plot.setter
    def show_plot(self, value):
        if not isinstance(value, bool):
            raise TypeError("`show_plot` needs to be of type bool.")
        self._show_plot = value

    @property
    def show_legend(self):
        """bool: Show legend. The default value is ``False``."""
        return self._show_legend

    @show_legend.setter
    def show_legend(self, value):
        if not isinstance(value, (tuple, list)):
            value = [value]
        if any(not isinstance(val, bool) for val in value):
            raise TypeError("`show_legend` needs to be of type bool or a list/tuple of type bool.")
        self._show_legend = tuple(value)

    @property
    def plot_grid(self):
        """
        bool: Whether to draw a grid in the plot. The default value is ``False``.
        """
        from warnings import warn

        warn(
            "This property is depreciated. Please use `show_grid` instead.",
            DeprecationWarning,
            2,
        )
        return self._show_grid

    @plot_grid.setter
    def plot_grid(self, value):
        from warnings import warn

        warn(
            "This property is depreciated. Please use `show_grid` instead.",
            DeprecationWarning,
            2,
        )
        self.show_grid = value

    @property
    def show_grid(self):
        """
        bool: Whether to draw a grid in the plot. The default value is ``False``.
        """
        return self._show_grid

    @show_grid.setter
    def show_grid(self, value):
        if not isinstance(value, bool):
            raise TypeError("`show_grid` needs to be of type bool.")
        self._show_grid = value

    @property
    def show_colorbar(self):
        """bool: Show colorbar (not supported by all plots)."""
        return self._show_colorbar

    @show_colorbar.setter
    def show_colorbar(self, value):
        self._show_colorbar = value

    @property
    def x_label(self):
        """
        str or None: Label of the x-axis. If ``None`` is given no label will be shown. The
        default value is ``None``.
        """
        return self._x_label

    @x_label.setter
    def x_label(self, value):
        self._x_label = _validate_xy_label("x_label", value)

    @property
    def y_label(self):
        """
        str or None: Label of the y-axis. If ``None`` is given no label will be shown. The
        default value is ``None``.
        """
        return self._y_label

    @y_label.setter
    def y_label(self, value):
        self._y_label = _validate_xy_label("y_label", value)

    @property
    def x_range(self):
        """
        tuple or list: Range of the x-axis. The default value is ``None``.
        """
        return self._x_range

    @x_range.setter
    def x_range(self, value):
        self._x_range = _validate_xy_range("x_range", value)

    @property
    def y_range(self):
        """
        tuple or list: Range of the y-axis. The default value is ``None``.
        """
        return self._y_range

    @y_range.setter
    def y_range(self, value):
        self._y_range = _validate_xy_range("y_range", value)

    # Legend properties:
    @property
    def legend_loc(self):
        """int: Location of the legend. The default value is ``1``."""
        return self._legend_loc

    @legend_loc.setter
    def legend_loc(self, value):
        self._legend_loc = value

    @property
    def legend_bbox_to_anchor(self):
        """tuple or list: Shift between box and anchor. The default value is ``(1, 1)``."""
        return self._legend_bbox_to_anchor

    @legend_bbox_to_anchor.setter
    def legend_bbox_to_anchor(self, value):
        self._legend_bbox_to_anchor = value

    @property
    def legend_ncol(self):
        """int: Columns of the legend (only supported for the matplotlib backend)."""
        return self._legend_ncol

    @legend_ncol.setter
    def legend_ncol(self, value):
        self._legend_ncol = value

    @property
    def legend_sort_entries(self):
        """bool: Sort entries of the legend."""
        return self._legend_sort_entries

    @legend_sort_entries.setter
    def legend_sort_entries(self, value):
        self._legend_sort_entries = value

    # Customization properties:
    @property
    def custom_colors(self):
        """list or tuple: Colors used in the plot."""
        return self._custom_colors

    @custom_colors.setter
    def custom_colors(self, value):
        self._custom_colors = _validate_custom_colors(value)

    @property
    def custom_linestyles(self):
        """
        list or tuple: Line styles used in the plot (This feature is not supported by all plot
        types).
        """
        return self._custom_linestyles

    @custom_linestyles.setter
    def custom_linestyles(self, value):
        self._custom_linestyles = value

    @property
    def custom_linewidths(self):
        """
        list or tuple: Line widths used in the plot (This feature is not supported by all plot
        types).
        """
        return self._custom_linewidths

    @custom_linewidths.setter
    def custom_linewidths(self, value):
        self._custom_linewidths = value

    @property
    def custom_markers(self):
        """
        list or tuple: Marker types used in the plot (This feature is not supported by all plot
        types).
        """
        return self._custom_markers

    @custom_markers.setter
    def custom_markers(self, value):
        self._custom_markers = value

    @property
    def custom_hatch(self):
        """float, list or tuple: Hatch value(s) controlling the hatch of plot elements."""
        return self._custom_hatch

    @custom_hatch.setter
    def custom_hatch(self, value):
        if not isinstance(value, (list, tuple)):
            value = (value,)
        self._custom_hatch = value

    @property
    def custom_alpha(self):
        """float, list or tuple: Alpha value(s) controlling the opacity of plot elements."""
        return self._custom_alpha

    @custom_alpha.setter
    def custom_alpha(self, value):
        if not isinstance(value, (list, tuple)):
            value = (value,)
        self._custom_alpha = value

    @property
    def custom_xticks(self):
        """list: List of values to set ticks on the x-axis."""
        return self._custom_xticks

    @custom_xticks.setter
    def custom_xticks(self, value):
        self._custom_xticks = value

    @property
    def custom_yticks(self):
        """list: List of values to set ticks on the y-axis."""
        return self._custom_yticks

    @custom_yticks.setter
    def custom_yticks(self, value):
        self._custom_yticks = value

    @property
    def custom_xticklabels(self):
        """list: List of labels for the ticks on the x-axis."""
        return self._custom_xticklabels

    @custom_xticklabels.setter
    def custom_xticklabels(self, value):
        self._custom_xticklabels = value

    @property
    def custom_yticklabels(self):
        """list: List of labels for the ticks on the y-axis."""
        return self._custom_yticklabels

    @custom_yticklabels.setter
    def custom_yticklabels(self, value):
        self._custom_yticklabels = value

    # Multiple plot properties:
    @property
    def subplot_sup_title(self):
        """str: Title of the whole figure."""
        return self._subplot_sup_title

    @subplot_sup_title.setter
    def subplot_sup_title(self, value):
        self._subplot_sup_title = value

    @property
    def subplot_sup_x_label(self):
        """str: x-label of the whole figure."""
        return self._subplot_sup_x_label

    @subplot_sup_x_label.setter
    def subplot_sup_x_label(self, value):
        self._subplot_sup_x_label = value

    @property
    def subplot_sup_y_label(self):
        """str: y-label of the whole figure."""
        return self._subplot_sup_y_label

    @subplot_sup_y_label.setter
    def subplot_sup_y_label(self, value):
        self._subplot_sup_y_label = value

    @property
    def subplot_nrows(self):
        """int: Number of rows. The default value is ``2``."""
        return self._subplot_nrows

    @subplot_nrows.setter
    def subplot_nrows(self, value):
        self._subplot_nrows = value

    @property
    def subplot_ncols(self):
        """int: Number of columns. The default value is ``1``."""
        return self._subplot_ncols

    @subplot_ncols.setter
    def subplot_ncols(self, value):
        self._subplot_ncols = value

    @property
    def subplot_gridspec(self):
        """list or tuple: Grid spec values."""
        return self._subplot_gridspec_values

    @subplot_gridspec.setter
    def subplot_gridspec(self, value):
        if not isinstance(value, list):
            raise TypeError("`subplot_gridspec` needs to be a list.")
        if any([(not isinstance(val, tuple)) or (len(val) != 4) for val in value]):
            raise ValueError("All elements of `subplot_gridspec` need to be a tuple of length 4.")

        value_array = np.array(value)
        if np.any(value_array[:, 0] >= value_array[:, 1]):
            raise ValueError("First row index needs to be larger than second row index.")
        if np.any(value_array[:, 2] >= value_array[:, 3]):
            raise ValueError("First column index needs to be larger than second column index.")
        if np.any(value_array[:, 1].max() > self.subplot_nrows):
            raise ValueError("Maximum row index is larger than number of rows.")
        if np.any(value_array[:, 3].max() > self.subplot_ncols):
            raise ValueError("Maximum column index is larger than number of columns.")

        specs = np.zeros((self.subplot_nrows, self.subplot_ncols))
        for val in value:
            specs[val[0] : val[1], val[2] : val[3]] += 1
        if np.any(specs > 1):
            raise ValueError("The specified gridspec contains overlapping subplots.")

        self._subplot_gridspec_values = value

    @property
    def subplot_hspace(self):
        """float: Vertical spacing between the subplots."""
        return self._subplot_hspace

    @subplot_hspace.setter
    def subplot_hspace(self, value):
        self._subplot_hspace = value

    @property
    def subplot_wspace(self):
        """float: Horizontal spacing between the subplots."""
        return self._subplot_wspace

    @subplot_wspace.setter
    def subplot_wspace(self, value):
        self._subplot_wspace = value

    @property
    def subplot_adjust(self):
        """: dict : Keyword arguments for the matplotlib ``subplots_adjust`` function."""
        return self._subplot_adjust

    @subplot_adjust.setter
    def subplot_adjust(self, value):
        self._subplot_adjust = value

    @property
    def subplot_share_legend(self):
        """bool: Merge legend items of all subplots."""
        return self._subplot_share_legend

    @subplot_share_legend.setter
    def subplot_share_legend(self, value):
        self._subplot_share_legend = value

    @property
    def subplot_share_colorbar(self):
        """bool: Use one common colorbar for all subplots."""
        return self._subplot_share_colorbar

    @subplot_share_colorbar.setter
    def subplot_share_colorbar(self, value):
        self._subplot_share_colorbar = value

    @property
    def subplot_sharex(self):
        """bool: Share the x-axis of subplots located in the same column."""
        return self._subplot_sharex

    @subplot_sharex.setter
    def subplot_sharex(self, value):
        self._subplot_sharex = value

    @property
    def subplot_sharey(self):
        """bool: Share the y-axis of subplots located in the same row."""
        return self._subplot_sharey

    @subplot_sharey.setter
    def subplot_sharey(self, value):
        self._subplot_sharey = value

    @property
    def subplot_tight_layout(self):
        """bool: Tight layout of plot. The default value is ``False``."""
        return self._subplot_tight_layout

    @subplot_tight_layout.setter
    def subplot_tight_layout(self, value):
        self._subplot_tight_layout = value

    @property
    def subplot_align_ylabels(self):
        """bool: Align y label of plot. The default value is ``False``."""
        return self._subplot_align_ylabels

    @subplot_align_ylabels.setter
    def subplot_align_ylabels(self, value):
        self._subplot_align_ylabels = value

    def auto_set_axis_properties(self, set_x_label=True, set_y_label=True):
        """
        Whether the axis labels and other axis properties are auto-generated.

        Parameters
        ----------
        set_x_label : bool
            Set x-axis label automatically.
        set_y_label : bool
            Set y-axis label automatically.
        """
        self._auto_set_x_label = set_x_label
        self._auto_set_y_label = set_y_label

    def create_default_gridspec(
        self, nrows, ncols, nplots, heights=1, widths=1, center_last_row=True
    ):
        """
        Create default grid for multiple plots.

        Parameters
        ----------
        nrows : Int
            Number of rows.
        ncols : int
            Number columns.
        nplots : int
            Number of subplots.
        heights : int (optional)
            Height of subplots.
        widths : int (optional)
            Width of subplots.
        center_last_row : bool (optional)
            Center the plots of the last row.
        """
        self._subplot_center_last_row = center_last_row
        last_row = nplots % ncols
        complete_rows = nplots // ncols
        gs_factor = widths
        if center_last_row and (ncols * widths - last_row) % 2 != 0:
            gs_factor *= 2

        grid_list = []
        for idx in range(nplots):
            grid_list.append(
                (
                    idx // ncols * heights,
                    (idx // ncols + 1) * heights,
                    (idx % ncols) * gs_factor,
                    (idx % ncols + 1) * gs_factor,
                )
            )

            if center_last_row and idx == complete_rows * ncols - 1:
                break

        if last_row == 0 or not center_last_row:
            self.subplot_nrows = nrows * heights
            self.subplot_ncols = ncols * gs_factor
            self._subplot_gf_x = gs_factor
            self._subplot_gf_y = heights
            self._subplot_gridspec_values = tuple(grid_list)
        else:
            start = ((ncols - last_row) * gs_factor) // 2
            for idx in range(start, start + last_row * gs_factor, gs_factor):
                grid_list.append(
                    (
                        (nrows - 1) * heights,
                        nrows * heights,
                        idx,
                        idx + gs_factor,
                    )
                )
            self.subplot_nrows = nrows * heights
            self.subplot_ncols = ncols * gs_factor
            self._subplot_gf_x = gs_factor
            self._subplot_gf_y = heights
            self._subplot_gridspec_values = tuple(grid_list)

    def reset_gridspec(self):
        """Reset gridspec settings."""
        self.subplot_nrows = int(self.subplot_nrows / self._subplot_gf_y)
        self.subplot_ncols = int(self.subplot_ncols / self._subplot_gf_x)
        self._subplot_gf_x = 1
        self._subplot_gf_y = 1
        self._subplot_gridspec_values = None

    def return_data_labels(self):
        """
        Return the labels of all data sets.

        Returns
        -------
        data_labels : list
            List of the labels of all data sets.
        """
        print(
            "This function is depreciated and will be removed, please use `data_labels` instead."
        )
        return self.data_labels

    def plot(self, data_labels, plot_title=None, plot_name="plot.png", subplot_assignment=None):
        """
        Plot the data sets.

        Parameters
        ----------
        data_labels : list or str
            List of data labels of the data sets that are plotted or
            in case only one data set is plotted a string.
        plot_title : list or str (optional)
            Title of the plots or subplots.
        plot_name : str (optional)
            The file name of the plot.
        subplot_assignment : list or None (optional)
            Assignment of the data sets to individual subplots.

        Returns
        -------
        fig : matplotlib.pyplot.figure or plotly.graph_objects.Figure
            Figure object of the plot.
        """
        if isinstance(data_labels, str):
            data_labels = [data_labels]
        if subplot_assignment is None:
            subplot_assignment = [0] * len(data_labels)
        (
            data_sets,
            x_ticks,
            y_ticks,
            x_tick_labels,
            y_tick_labels,
            sec_axis,
        ) = self._prepare_to_plot(data_labels, subplot_assignment)

        # We don't have to distinguish between single/multiple plots, if data_sets is a nested list
        # we create n subplots, otherwise the number of subplots is 1:
        n_subplots = 1
        if isinstance(data_sets[0], (tuple, list)):
            data_sets = data_sets[: self.subplot_nrows * self.subplot_ncols]
            for _ in range(
                len(data_sets),
                self.subplot_nrows
                // self._subplot_gf_y
                * self.subplot_ncols
                // self._subplot_gf_x,
            ):
                data_sets.append([])
            n_subplots = len(data_sets)
        else:
            data_sets = [data_sets]

        # We adjust the input parameters to fit the number of subplots and allow input suggestions
        # from the child classes which are overwritten by the class properties:
        if self.custom_xticks is not None:
            x_ticks = self.custom_xticks
        if self.custom_xticklabels is not None:
            x_tick_labels = self.custom_xticklabels
        if self.custom_yticks is not None:
            y_ticks = self.custom_yticks
        if self.custom_yticklabels is not None:
            y_tick_labels = self.custom_yticklabels
        plot_title, x_label, y_label, show_legend, show_colorbar = _check_plot_param_single_values(
            (plot_title, self.x_label, self.y_label, self.show_legend, self.show_colorbar),
            n_subplots,
        )
        if self.subplot_share_colorbar:
            show_colorbar = [False] * (n_subplots - 1) + [True]
        (
            x_range,
            y_range,
            x_ticks,
            y_ticks,
            x_tick_labels,
            y_tick_labels,
            sec_axis,
        ) = _check_plot_param_list_values(
            (
                self.x_range,
                self.y_range,
                x_ticks,
                y_ticks,
                x_tick_labels,
                y_tick_labels,
                sec_axis,
            ),
            n_subplots,
        )

        # We crate the figure and fill the subplots with data:
        axes = self._create_figure(n_subplots, plot_title)
        legend_handles = []
        legend_labels = []

        subplot_data_label_mapping = {}
        for dl, sp_a in zip(data_labels, subplot_assignment):
            subplot_data_label_mapping.setdefault(sp_a, []).append(dl)

        for ax_idx, ax in enumerate(axes):
            if ax_idx == len(data_sets):
                data_sets.append([])
            plot_extras = [[], []]
            for extra_idx, extra_attr in enumerate(
                ["_plot_extras_lines_background", "_plot_extras_lines_foreground"]
            ):
                extra_items = getattr(self, extra_attr, {})
                plot_extras[extra_idx] += extra_items.get("all", [])
                plot_extras[extra_idx] += extra_items.get(ax_idx, [])
                for data_label in subplot_data_label_mapping.get(ax_idx, []):
                    plot_extras[extra_idx] += extra_items.get(data_label, [])

            legend_handles0, legend_labels0 = self._plot2d(
                ax,
                plot_extras[0] + data_sets[ax_idx] + plot_extras[1],
                x_label[ax_idx],
                y_label[ax_idx],
                x_range[ax_idx],
                y_range[ax_idx],
                x_ticks[ax_idx],
                y_ticks[ax_idx],
                x_tick_labels[ax_idx],
                y_tick_labels[ax_idx],
                sec_axis[ax_idx],
                show_colorbar[ax_idx],
            )
            legend_handles.append(legend_handles0)
            legend_labels.append(legend_labels0)

        # We need to have another loop in case of a shared legend for the matplotlib backend...
        if self.subplot_share_legend:
            # We would like to have all legend handles/labels in the subplots:
            data_sets = [
                [item for sublist in data_sets for item in sublist] for _ in range(n_subplots)
            ]
            legend_handles = [
                [item for sublist in legend_handles for item in sublist] for _ in range(n_subplots)
            ]
            legend_labels = [
                [item for sublist in legend_labels for item in sublist] for _ in range(n_subplots)
            ]
        for ax_idx, ax in enumerate(axes):
            if show_legend[ax_idx]:
                self._create_legend(
                    ax, data_sets[ax_idx], (legend_handles[ax_idx], legend_labels[ax_idx])
                )

        if n_subplots > 1:
            self._set_subplot_parameters()
        self._finalize_plot(plot_name)
        return self._figure

    def _print_extra_properties(self):
        return ""

    @abc.abstractmethod
    def _prepare_to_plot(self, data_labels, subplot_assignment):
        """
        Create data sets that are plotted.

        TODO: increase description here.
        """
        data_sets = []
        x_ticks = None
        y_ticks = None
        x_tick_labels = None
        y_tick_labels = None
        sec_axis = None
        return data_sets, x_ticks, y_ticks, x_tick_labels, y_tick_labels, sec_axis

    def _apply_data_set_customizations(self, data_set):
        def swap_attr(value, attr_type):
            if isinstance(value, int):
                attr_list = getattr(self, "_default_" + attr_type)
                cust_attr_list = getattr(self, "custom_" + attr_type)
                if cust_attr_list is not None:
                    attr_list = cust_attr_list
                if value < len(attr_list):
                    value = attr_list[value]
                else:
                    value = attr_list[-1]
            return value

        for keyw in ["color", "facecolor"]:
            if keyw in data_set:
                data_set[keyw] = swap_attr(data_set[keyw], "colors")
        if "linestyle" in data_set:
            data_set["linestyle"] = swap_attr(data_set["linestyle"], "linestyles")
        if "linewidth" in data_set:
            data_set["linewidth"] = swap_attr(data_set["linewidth"], "linewidths")
        if "marker" in data_set:
            data_set["marker"] = swap_attr(data_set["marker"], "markers")
        if "hatch" in data_set:
            data_set["hatch"] = swap_attr(data_set["hatch"], "hatch")
        if "alpha" in data_set:
            data_set["alpha"] = swap_attr(data_set["alpha"], "alpha")

    def _return_data_set(self, data_label, dict_tree=[], deepcopy=True):
        if data_label in self._data:
            data_set = self._data[data_label]
            for dict_key in dict_tree:
                data_set = data_set[dict_key]
            if deepcopy:
                return copy.deepcopy(data_set)
            else:
                return data_set
        else:
            raise ValueError(f"data_label '{data_label}' not found.")

    def _check_data_label(self, data_label):
        if data_label in self._data:
            raise ValueError(f"data_label {data_label} already used.")

    def _auto_set_axis_properties(self, x_label=None, y_label=None):
        if (
            x_label is not None
            and getattr(self, "_auto_set_x_label", False)
            and self.x_label is None
        ):
            self.x_label = x_label
        if (
            y_label is not None
            and getattr(self, "_auto_set_y_label", False)
            and self.y_label is None
        ):
            self.y_label = y_label
