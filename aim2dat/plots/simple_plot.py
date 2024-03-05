"""Test plots for the base plot class (will be removed at a later stage)."""

from aim2dat.plots.base_plot import _BasePlot
from aim2dat.plots.base_mixin import _HLineMixin, _VLineMixin


class SimplePlot(_BasePlot, _HLineMixin, _VLineMixin):
    """
    Direct and flexible interface to the backend plot libraries.
    """

    _object_title = "Simple Plot"

    def import_scatter_data_set(
        self,
        data_label,
        x_values,
        y_values,
        y_values_2=None,
        plot_label=None,
        color=None,
        face_color=None,
        alpha=None,
        line_style=None,
        line_width=None,
        marker=None,
        marker_face_color=None,
        marker_edge_width=None,
        use_fill=False,
        use_fill_between=False,
    ):
        """
        Import data set for a scatter plot.

        Parameters
        ----------
        data_label : str
            Internal label of the data set.
        x_values : list
            List of x-values of the points.
        y_values : list
            List of y-values of the points.
        y_values_2 : list
            Second list of y-values to be used with ``use_fill_between``.
        plot_label : str (optional)
            Label of the data set shown in the legend.
        color : str (optional)
            Color of the data set.
        face_color : str (optional)
            Face color of the data set.
        alpha : float (optional)
            Transparency of the data set.
        line_style : str (optional)
            Line style of the data set.
        line_width : int or float
            Line width of the data set.
        marker : str (optional)
            Marker shape.
        marker_face_color : str (optional)
            Marker face color.
        marker_edge_width : int (optional)
            Marker size.
        use_fill : bool (optional)
            Whether to fill the area between the x-axis and the data points.
        use_fill_between : bool (optional)
            Whether to fill the area between the deta sets.
        """
        self._check_data_label(data_label)
        data_set = {
            "x_values": x_values,
            "y_values": y_values,
            "type": "scatter",
            "use_fill": use_fill,
            "use_fill_between": use_fill_between,
        }
        # TODO: validation of inputs, probably best in _BasePlot or additional mixin class?
        if y_values_2 is not None:
            data_set["y_values_2"] = y_values_2
        if plot_label is not None:
            data_set["label"] = plot_label
        if color is not None:
            data_set["color"] = color
        if face_color is not None:
            data_set["facecolor"] = face_color
        if alpha is not None:
            data_set["alpha"] = alpha
        if line_style is not None:
            data_set["linestyle"] = line_style
        if line_width is not None:
            data_set["linewidth"] = line_width
        if marker is not None:
            data_set["marker"] = marker
        if marker_face_color is not None:
            data_set["markerfacecolor"] = marker_face_color
        if marker_edge_width is not None:
            data_set["markeredgewidth"] = marker_edge_width
        self._data[data_label] = data_set

    def import_bar_data_set(
        self,
        data_label,
        x_values,
        y_values,
        heights,
        plot_label=None,
        color=None,
        alpha=None,
        bottom=0.0,
        width=0.8,
        align="center",
    ):
        """
        Import data set for a bar plot.

        Parameters
        ----------
        data_label : str
            Internal label of the data set.
        x_values : list
            List of x-values of the points.
        y_values : list
            List of y-values of the points.
        heights : list
            List of bar heights.
        plot_label : str (optional)
            Label of the data set shown in the legend.
        color : str (optional)
            Color of the data set.
        alpha : float (optional)
            Transparency of the data set.
        bottom : float
            Position of the bottom of the bars.
        width : float
            Width of the bars.
        align : str
            Alignment of the bar.
        """
        self._check_data_label(data_label)
        data_set = {
            "x_values": x_values,
            "y_values": y_values,
            "heights": heights,
            "bottom": bottom,
            "width": width,
            "align": align,
            "type": "bar",
        }
        if plot_label is not None:
            data_set["label"] = plot_label
        if plot_label is not None:
            data_set["label"] = plot_label
        if color is not None:
            data_set["color"] = color
        if alpha is not None:
            data_set["alpha"] = alpha
        self._data[data_label] = data_set

    def _prepare_to_plot(self, data_labels, subplot_assignment):
        data_sets = [[] for idx0 in range(max(subplot_assignment) + 1)]
        for data_label, subp_a in zip(data_labels, subplot_assignment):
            data_sets[subp_a].append(self._return_data_set(data_label))
        return data_sets, None, None, None, None, None
