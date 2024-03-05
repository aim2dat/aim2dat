"""Methods for plots using Plolty."""

# Third party library imports
from plotly.subplots import make_subplots
import plotly.graph_objs as go

# import plotly.tools as pl_tls

LINESTYLES = {
    "solid": "solid",
    "dashed": "dash",
    "dotted": "dot",
    "dashdot": "dashdot",
    "-": "solid",
    "--": "dash",
    ":": "dot",
    "-.": "dashdot",
}
COLORS = {
    "C0": "#1f77b4",
    "C1": "#ff7f0e",
    "C2": "#2ca02c",
    "C3": "#d62728",
    "C4": "#9467bd",
    "C5": "#8c564b",
    "C6": "#e377c2",
    "C7": "#7f7f7f",
    "C8": "#bcbd22",
    "C9": "#17becf",
}
MARKER_SYMBOLS = {
    "*": "star",
    "o": "circle",
    "<": "triangle-left",
    ">": "triangle-right",
    "s": "square",
    "v": "triangle-down",
    "h": "hexagon",
    "H": "hexagon2",
    "D": "diamond",
    "x": "x-thin",
    "X": "x",
}
PLOTLY_ARGS = ["customdata", "hovertemplate"]
RATIO_F = 100.0
RATIO_MARKER = 5.0
SPACING_F = 0.2


def _transform_color(data_set):
    color = data_set.get("color")
    if color is None:
        color = data_set.get("facecolor")
    if color is not None and color in COLORS:
        color = COLORS[color]
    return color


def _create_figure(obj, n_subplots, plot_title):
    layout_kwargs = {}
    if obj.ratio is not None:
        layout_kwargs["width"] = obj.ratio[0] * RATIO_F
        layout_kwargs["height"] = obj.ratio[1] * RATIO_F
    if n_subplots == 1:
        layout_kwargs["title"] = plot_title[0]
        obj._figure = go.Figure()
        subplots = [(None, None)]
    else:
        subplots_kwargs = {
            "rows": obj.subplot_nrows,
            "cols": obj.subplot_ncols,
            "print_grid": False,
            "shared_xaxes": obj.subplot_sharex,
            "shared_yaxes": obj.subplot_sharey,
        }

        if obj.subplot_wspace is not None:
            subplots_kwargs["horizontal_spacing"] = obj.subplot_wspace * SPACING_F
        if obj.subplot_hspace is not None:
            subplots_kwargs["vertical_spacing"] = obj.subplot_hspace * SPACING_F
        subplots = []
        row_idx = 1
        col_idx = 1
        if obj._subplot_gridspec_values is not None:
            n_rows = max(gridspec_val[1] for gridspec_val in obj._subplot_gridspec_values)
            n_cols = max(gridspec_val[3] for gridspec_val in obj._subplot_gridspec_values)
            specs = [[None for idx1 in range(n_cols)] for idx0 in range(n_rows)]
            for gridspec_val in obj._subplot_gridspec_values:
                while gridspec_val[0] >= len(specs):
                    specs.append([])
                while gridspec_val[2] >= len(specs[gridspec_val[0]]):
                    specs[gridspec_val[0]].append(None)
                specs[gridspec_val[0]][gridspec_val[2]] = {
                    "rowspan": gridspec_val[1] - gridspec_val[0],
                    "colspan": gridspec_val[3] - gridspec_val[2],
                }
                subplots.append((gridspec_val[0] + 1, gridspec_val[2] + 1))
            subplots_kwargs["specs"] = specs
        else:
            for subplot_idx in range(1, n_subplots + 1):
                subplots.append((row_idx, col_idx))
                col_idx += 1
                if col_idx > obj.subplot_ncols:
                    row_idx += 1
                    col_idx = 1
        obj._figure = make_subplots(
            **subplots_kwargs,
            subplot_titles=plot_title,
        )
    if obj.equal_aspect_ratio:
        for subp_idx in range(len(subplots)):
            yaxis = getattr(obj._figure.layout, "yaxis" + str(subp_idx + 1))
            yaxis.scaleanchor = "x" + str(subp_idx + 1)
            yaxis.scaleratio = 1.0
    obj._figure.update_layout(**layout_kwargs)
    return subplots


def _plot2d(
    obj,
    subplot,
    data_sets,
    x_label,
    y_label,
    x_range,
    y_range,
    x_ticks,
    y_ticks,
    x_tick_labels,
    y_tick_labels,
    sec_axis,
    show_colorbar,
):
    xaxes_args = {"row": subplot[0], "col": subplot[1], "title_text": x_label}
    yaxes_args = {"row": subplot[0], "col": subplot[1], "title_text": y_label}
    if x_range is not None:
        xaxes_args["range"] = x_range
    if y_range is not None:
        yaxes_args["range"] = y_range
    if x_tick_labels is not None:
        xaxes_args["tickmode"] = "array"
        xaxes_args["tickvals"] = x_ticks
        xaxes_args["ticktext"] = x_tick_labels
    if y_tick_labels is not None:
        yaxes_args["tickmode"] = "array"
        yaxes_args["tickvals"] = y_ticks
        yaxes_args["ticktext"] = y_tick_labels
    if sec_axis is not None:
        for ax in sec_axis:
            obj._create_secondary_axis(**ax)
    obj._figure.update_xaxes(**xaxes_args)
    obj._figure.update_yaxes(**yaxes_args)
    for data_set in data_sets:
        data_set0 = data_set.copy()
        obj._apply_data_set_customizations(data_set0)
        plot_type = "scatter"
        if "type" in data_set0:
            plot_type = data_set0.pop("type")
        plot_function = getattr(obj, "_add_" + plot_type)
        plot_function(subplot, data_set0)
    return [], []


def _finalize_plot(obj, plot_name):
    if obj._show_plot:
        obj._figure.show()
    if obj._store_plot and plot_name:
        if obj._store_path != "":
            obj._store_path += "/"
        obj._figure.write_image(obj._store_path + plot_name)


def _set_subplot_parameters(obj):
    pass


def _add_scatter(obj, subplot, data_set):
    data = {}
    data["x"] = data_set.pop("x_values")
    data["y"] = data_set.pop("y_values")
    data["mode"] = ""
    if "linestyle" in data_set and data_set["linestyle"] != "none":
        data["mode"] += "lines"
        data["line"] = {
            "dash": LINESTYLES[data_set["linestyle"]],
            "color": _transform_color(data_set),
            "width": data_set.get("width"),
        }
    elif "linestyle" not in data_set:
        data["mode"] += "lines"
        data["line"] = {
            "dash": "solid",
            "color": _transform_color(data_set),
            "width": data_set.get("width"),
        }
    if "linestyle" in data_set and data_set["linestyle"] == "none":
        data["mode"] += "markers"
        data["marker"] = {"color": _transform_color(data_set)}
        if "markeredgewidth" in data_set:
            data["marker"]["size"] = data_set["markeredgewidth"] * RATIO_MARKER
        if "marker" in data_set:
            data["marker_symbol"] = MARKER_SYMBOLS[data_set["marker"]]
    if "use_fill" in data_set and data_set["use_fill"]:
        data["fill"] = "tozeroy"
    if "use_fill_between" in data_set and data_set["use_fill_between"]:
        data["fill"] = "tonexty"
    if "legendgrouptitle_text" in data_set:
        data["legendgrouptitle_text"] = data_set["legendgrouptitle_text"]
    if "label" in data_set:
        data["name"] = data_set["label"]
    else:
        data["showlegend"] = False
    if "legendgroup" in data_set:
        data["legendgroup"] = data_set["legendgroup"]
    for arg in PLOTLY_ARGS:
        if arg in data_set:
            data[arg] = data_set[arg]
    obj._figure.add_trace(go.Scatter(**data), row=subplot[0], col=subplot[1])


def _add_heatmap(obj, subplot, data_set):
    data = {
        "x": data_set.get("x_values"),
        "y": data_set.get("y_values"),
        "z": data_set.get("z_values"),
        "colorscale": data_set.get("cmap", "Viridis"),
        "showscale": data_set.get("show_cmap", False),
    }
    obj._figure.add_trace(go.Heatmap(**data), row=subplot[0], col=subplot[1])


def _add_contour(obj, subplot, data_set):
    print("Warning: Contour plot is not yet implemented for this backend.")


def _add_bar(obj, subplot, bar):
    if "bottom" in bar:
        obj._figure.update_layout(barmode="stack")
    data = {
        "x": bar["x_values"],
        "y": bar["heights"],
        "marker_color": _transform_color(bar),
        "width": bar.get("width", 0.8),
    }
    if "label" in bar:
        data["name"] = bar["label"]
    else:
        data["showlegend"] = False
    obj._figure.add_trace(go.Bar(**data), row=subplot[0], col=subplot[1])


def _add_vline(obj, subplot, vline):
    processed_vline = {}
    processed_vline["type"] = "line"
    processed_vline["line"] = {"color": _transform_color(vline)}
    if "scaled" in vline and vline["scaled"]:
        processed_vline["yref"] = "y domain"
    if "linestyle" in vline:
        processed_vline["line"]["dash"] = LINESTYLES[vline["linestyle"]]
    if "ymin" in vline:
        processed_vline["y0"] = vline["ymin"]
    else:
        processed_vline["y0"] = 0.0
    if "ymax" in vline:
        processed_vline["y1"] = vline["ymax"]
    else:
        processed_vline["y1"] = 1.0
    processed_vline["x0"] = vline["x"]
    processed_vline["x1"] = vline["x"]
    obj._figure.add_shape(**processed_vline, row=subplot[0], col=subplot[1])


def _add_hline(obj, subplot, hline):
    processed_hline = {}
    processed_hline["type"] = "line"
    processed_hline["line"] = {"color": _transform_color(hline)}
    if "scaled" in hline and hline["scaled"]:
        processed_hline["xref"] = "x domain"
    if "linestyle" in hline:
        processed_hline["line"]["dash"] = LINESTYLES[hline["linestyle"]]
    if "xmin" in hline:
        processed_hline["x0"] = hline["xmin"]
    else:
        processed_hline["x0"] = 0.0
    if "xmax" in hline:
        processed_hline["x1"] = hline["xmax"]
    else:
        processed_hline["x1"] = 1.0
    processed_hline["y0"] = hline["y"]
    processed_hline["y1"] = hline["y"]
    obj._figure.add_shape(**processed_hline, row=subplot[0], col=subplot[1])


def _add_text(obj, subplot, text_data):
    data = {
        "x": [text_data["x"]],
        "y": [text_data["y"]],
        "text": [text_data["label"]],
        "mode": "text",
        "textposition": "middle center",
    }
    obj._figure.add_trace(go.Scatter(**data), row=subplot[0], col=subplot[1])


def _plot_heatmap(obj):
    return None


def _create_secondary_axis(obj, ticks, tick_labels, coord="x", loc="top"):
    for tick_pos, tick_label in zip(ticks, tick_labels):
        if coord == "x":
            y_pos = 0.0
            yshift = -25.0
            if loc == "top":
                y_pos = 1.0
                yshift = 25.0
            obj._figure.add_annotation(
                xref="x",
                yref="paper",
                yshift=yshift,
                x=tick_pos,
                y=y_pos,
                text=tick_label,
                showarrow=False,
            )
        else:
            x_pos = 0.0
            xshift = -25.0
            if loc == "top":
                x_pos = 1.0
                xshift = 25.0
            obj._figure.add_annotation(
                xref="paper",
                xshift=xshift,
                yref="y",
                x=x_pos,
                y=tick_pos,
                text=tick_label,
                showarrow=False,
            )


def _create_legend(obj, axes, data_sets, handles_labels):
    pass


def _create_colorbar():
    pass
