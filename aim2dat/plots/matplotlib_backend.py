"""Methods for plots using Matplotlib."""

# Standard library imports
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, NullLocator, LogLocator, SymmetricalLogLocator
import matplotlib.lines as plt_lines
import matplotlib.patches as plt_patches
import matplotlib.colors as plt_colors

MATPLOTLIB_ARGS = [
    "label",
    "marker",
    "markerfacecolor",
    "markeredgewidth",
    "linestyle",
    "linewidth",
    "color",
    "facecolor",
    "hatch",
    "alpha",
]


def _create_figure(obj, n_subplots, plot_title):
    plt.style.use(obj._style_sheet)
    obj._figure = plt.figure(figsize=obj.ratio)
    nrows = 1
    ncols = 1
    if n_subplots > 1:
        nrows = int(obj.subplot_nrows / obj._subplot_gf_y)
        ncols = int(obj.subplot_ncols / obj._subplot_gf_x)
    shared_x_axis = None
    shared_y_axis = None
    axes = []
    # ToDo adapt loop to logic in plotly backend
    gridspec = obj._figure.add_gridspec(obj.subplot_nrows, obj.subplot_ncols)
    if obj._subplot_gridspec_values is not None and n_subplots > 1:
        gridspec_values = obj.subplot_gridspec
    else:
        gridspec_values = [
            (
                i // obj.subplot_ncols,
                i // obj.subplot_ncols + 1,
                i % obj.subplot_ncols,
                i % obj.subplot_ncols + 1,
            )
            for i in range(n_subplots)
        ]

    for subplot_idx, gridspec_value in enumerate(gridspec_values):
        ax = obj._figure.add_subplot(
            gridspec[
                gridspec_value[0] : gridspec_value[1],
                gridspec_value[2] : gridspec_value[3],
            ],
            sharex=shared_x_axis,
            sharey=shared_y_axis,
        )
        if obj.subplot_sharex:
            if (subplot_idx + ncols) % ncols == 0:
                shared_x_axis = ax
            if not obj._subplot_center_last_row and subplot_idx < n_subplots - ncols:
                plt.setp(ax.get_xticklabels(), visible=False)
        if obj.subplot_sharey:
            if (subplot_idx + nrows) % nrows == 0:
                shared_y_axis = ax
            if subplot_idx % ncols != 0:
                plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_title(plot_title[subplot_idx])
        if obj.equal_aspect_ratio:
            ax.set_aspect("equal")
        axes.append(ax)
    return axes


def _set_subplot_parameters(obj):
    # Put sup-title and labels:
    if obj.subplot_sup_title is not None:
        obj._figure.suptitle(obj.subplot_sup_title)
    if obj.subplot_sup_x_label is not None:
        obj._figure.supxlabel(obj.subplot_sup_x_label)
    if obj.subplot_sup_y_label is not None:
        obj._figure.supylabel(obj.subplot_sup_y_label)
    # Adjust spacings:
    adjust_kwargs = {}
    if obj.subplot_wspace is not None:
        adjust_kwargs["wspace"] = obj.subplot_wspace
    if obj.subplot_hspace is not None:
        adjust_kwargs["hspace"] = obj.subplot_hspace
    adjust_kwargs.update(obj.subplot_adjust)
    obj._figure.subplots_adjust(**adjust_kwargs)
    if obj.subplot_tight_layout:
        obj._figure.tight_layout()
    if obj.subplot_align_ylabels:
        obj._figure.align_ylabels()


def _plot2d(
    obj,
    axes,
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
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.xaxis.set_minor_locator(AutoMinorLocator())
    axes.yaxis.set_minor_locator(AutoMinorLocator())

    if x_range is not None:
        axes.set_xlim(x_range)
    if y_range is not None:
        axes.set_ylim(y_range)
    if x_ticks is not None:
        axes.set_xticks(x_ticks)
    if y_ticks is not None:
        axes.set_yticks(y_ticks)
    if x_tick_labels is not None:
        axes.set_xticklabels(x_tick_labels)
        axes.xaxis.set_minor_locator(NullLocator())
    if y_tick_labels is not None:
        axes.set_yticklabels(y_tick_labels)
        axes.yaxis.set_minor_locator(NullLocator())
    if sec_axis is not None:
        for ax in sec_axis:
            obj._create_secondary_axis(axes, **ax)
    for data_set in data_sets:
        # We copy in case another plot needs the data sets:
        data_set0 = data_set.copy()
        obj._apply_data_set_customizations(data_set0)
        plot_type = "scatter"
        if "type" in data_set0:
            plot_type = data_set0.pop("type")
        plot_function = getattr(obj, "_add_" + plot_type)
        plot_function(axes, data_set0)
    # if show_legend:
    #     obj._create_legend(axes, data_sets)
    if show_colorbar and not obj.subplot_share_colorbar:
        obj._create_colorbar(axes)
    return axes.get_legend_handles_labels()


def _plot_ternary(obj):
    return None


def _finalize_plot(obj, plot_name):
    for ax in obj._figure.axes:
        if not (ax.lines or ax.collections or ax.containers):
            ax.set_axis_off()

    if obj.subplot_share_colorbar:
        _create_colorbar(obj, obj._figure.axes)

    if obj._show_plot:
        plt.draw()
        plt.show()

    if obj._store_plot and plot_name:
        if obj._store_path != "":
            obj._store_path += "/"
        file_format = plot_name.split(".")[-1]
        obj._figure.savefig(obj._store_path + plot_name, format=file_format)
    plt.close(obj._figure)


def _create_legend(obj, axes, data_sets, handles_labels):
    handles, labels = handles_labels
    new_handles = []
    new_labels = []
    group_handles = []
    group_labels = []
    used_labels = []
    for data_set in data_sets:
        if "group_by" in data_set:
            if data_set["group_by"] == "color" and "label" in data_set:
                label_idx = labels.index(data_set["label"])
                del labels[label_idx]
                handle = handles.pop(label_idx)

                try:
                    color = handle.get_color()
                except AttributeError:
                    color = handle.get_edgecolor()
                if data_set["label"] not in group_labels:
                    if isinstance(handle, plt_patches.Polygon):
                        new_handle = plt_lines.Line2D(
                            [],
                            [],
                            color="black",
                            label=handle.get_label(),
                            linestyle=handle.get_linestyle(),
                        )
                    else:
                        new_handle = type(handle)([], [])
                        new_handle.update_from(handle)
                        new_handle.set_color("black")
                    group_handles.append(new_handle)
                    group_labels.append(data_set["label"])
                if (
                    "legendgrouptitle_text" in data_set
                    and data_set["legendgrouptitle_text"] not in new_labels
                ):
                    new_handles.append(plt_patches.Patch(color=color, alpha=1.0))
                    new_labels.append(data_set["legendgrouptitle_text"])
            if data_set["group_by"].startswith("line"):
                print("Not yet implemented...")
        elif "label" in data_set:
            if data_set["label"] not in used_labels:
                label_idx = labels.index(data_set["label"])
                used_labels.append(data_set["label"])
                new_handles.append(handles[label_idx])
                new_labels.append(labels[label_idx])
    if len(new_handles + group_handles) > 0:
        if obj.legend_sort_entries:
            zipped = list(zip(new_labels, new_handles))
            zipped.sort(key=lambda point: point[0])
            new_labels, new_handles = zip(*zipped)
        elif hasattr(obj, "_legend_order"):
            new_labels, new_handles = _order_legend(new_labels, new_handles, obj._legend_order[0])
            group_labels, group_handles = _order_legend(
                group_labels, group_handles, obj._legend_order[1]
            )
        axes.legend(
            ncol=obj.legend_ncol,
            handles=list(new_handles) + group_handles,
            labels=list(new_labels) + group_labels,
            loc=obj._legend_loc,
            bbox_to_anchor=obj._legend_bbox_to_anchor,
        )


def _order_legend(labels, handles, order):
    if order is None:
        return labels, handles
    else:
        ordered_labels = []
        ordered_handles = []
        for label in order:
            label_idx = labels.index(label)
            ordered_labels.append(labels[label_idx])
            ordered_handles.append(handles[label_idx])
        return ordered_labels, ordered_handles


def _create_colorbar(obj, axes):
    if hasattr(obj, "_cmap"):
        obj._figure.colorbar(obj._cmap, ax=axes)


def _create_secondary_axis(obj, axes, ticks, tick_labels, coord="x", loc="top"):
    if coord == "x":
        secax = axes.secondary_xaxis(loc)
        secax.set_xticks(ticks)
        secax.set_xticklabels(tick_labels)
    else:
        secax = axes.secondary_yaxis(loc)
        secax.set_yticks(ticks)
        secax.set_yticklabels(tick_labels)


def _add_scatter(obj, axes, data_set):
    add_args = {}
    x_values = data_set.pop("x_values")
    y_values = data_set.pop("y_values")
    use_fill = False
    use_fill_between = False
    for arg in MATPLOTLIB_ARGS:
        if arg in data_set:
            add_args[arg] = data_set[arg]
    if "use_fill" in data_set:
        use_fill = data_set.pop("use_fill")
    if "use_fill_between" in data_set:
        use_fill_between = data_set.pop("use_fill_between")
    if use_fill:
        axes.fill(x_values, y_values, **add_args)
    elif use_fill_between:
        y_values_2 = data_set.pop("y_values_2")
        axes.fill_between(x_values, y_values, y_values_2, **add_args)
    else:
        axes.plot(x_values, y_values, **add_args)


def _add_heatmap(obj, axes, data_set):
    # show_cmap = data_set.pop("show_cmap", False)
    norm_type = data_set.pop("norm_type", None)
    if norm_type is not None:
        norm_f = getattr(plt_colors, norm_type)
        data_set["norm"] = norm_f(**data_set.pop("norm_args"))
    obj._cmap = axes.pcolor(
        data_set.pop("x_values"),
        data_set.pop("y_values"),
        data_set.pop("z_values"),
        **data_set,
        shading="auto",
    )


def _add_contour(obj, axes, data_set):
    show_cmap = data_set.pop("show_cmap", False)
    norm_type = data_set.pop("norm_type", None)
    if norm_type is not None:
        norm_f = getattr(plt_colors, norm_type)
        data_set["norm"] = norm_f(**data_set.pop("norm_args"))
    if data_set.pop("log_scale", False):
        data_set["locator"] = LogLocator(
            linthresh=data_set.pop("linthresh", 0.01), base=data_set.pop("base", 10)
        )
    elif data_set.pop("symlog_scale", False):
        data_set["locator"] = SymmetricalLogLocator(
            linthresh=data_set.pop("linthresh", 0.01), base=data_set.pop("base", 10)
        )
    if data_set.pop("filled", False):
        cmap = axes.contourf(
            data_set.pop("x_values"),
            data_set.pop("y_values"),
            data_set.pop("z_values"),
            **data_set,
        )
    else:
        cmap = axes.contour(
            data_set.pop("x_values"),
            data_set.pop("y_values"),
            data_set.pop("z_values"),
            **data_set,
        )
    if show_cmap:
        obj._figure.colorbar(cmap, ax=axes)


def _add_bar(obj, axes, bar):
    color = (
        (bar.get("color"), bar.get("alpha"))
        if "alpha" in bar and "color" in bar
        else bar.get("color", None)
    )
    alpha = bar.get("alpha", None) if not color else None
    axes.bar(
        x=bar["x_values"],
        height=bar["heights"],
        width=bar.get("width", 0.8),
        bottom=bar.get("bottom", 0),
        align=bar.get("align", "center"),
        yerr=bar.get("y_error", None),
        capsize=bar.get("capsize", None),
        ecolor=bar.get("ecolor", None),
        color=color,
        alpha=alpha,
        hatch=bar.get("hatch", None),
        edgecolor=bar.get("edgecolor", "black"),
        linewidth=bar.get("linewidth", 0.5),
        label=bar.get("label", None),
    )


def _add_vline(obj, axes, vline):
    scaled = False
    if "scaled" in vline:
        scaled = vline.pop("scaled")
    if scaled:
        axes.axvline(**vline)
    else:
        axes.vlines(**vline)


def _add_hline(obj, axes, hline):
    scaled = False
    if "scaled" in hline:
        scaled = hline.pop("scaled")
    if scaled:
        axes.axhline(**hline)
    else:
        axes.hlines(**hline)


def _add_text(obj, axes, text_data):
    x = text_data.pop("x")
    y = text_data.pop("y")
    label = text_data.pop("label")
    if "size" not in text_data:
        text_data["size"] = 20.0
    axes.text(x, y, label, **text_data)
