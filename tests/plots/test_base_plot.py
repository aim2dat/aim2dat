"""Test _BasePlot implementation."""

# Third party library imports
import pytest

# Internal library imports
from aim2dat.plots import SimplePlot


@pytest.mark.parametrize(
    "prop,value,ref_value,error_type,error_m",
    [
        ("backend", "plotly", "plotly", None, None),
        ("backend", "test", None, ValueError, "Backend 'test' is not supported."),
        ("ratio", [10, 5], (10, 5), None, None),
        (
            "ratio",
            [10, 5, 4],
            None,
            TypeError,
            "`ratio` must be a list/tuple consisting of two numbers.",
        ),
        (
            "ratio",
            ["", 5],
            None,
            TypeError,
            "`ratio` must be a list/tuple consisting of two numbers.",
        ),
        ("ratio", 1, None, TypeError, "`ratio` must be a list/tuple consisting of two numbers."),
        ("equal_aspect_ratio", True, True, None, None),
        ("equal_aspect_ratio", False, False, None, None),
        (
            "equal_aspect_ratio",
            "",
            None,
            TypeError,
            "`equal_aspect_ratio` needs to be of type bool.",
        ),
        ("store_path", None, "./", None, None),
        ("store_plot", True, True, None, None),
        ("store_plot", False, False, None, None),
        ("store_plot", "", None, TypeError, "`store_plot` needs to be of type bool."),
        ("show_plot", True, True, None, None),
        ("show_plot", False, False, None, None),
        ("show_plot", "", None, TypeError, "`show_plot` needs to be of type bool."),
        ("show_legend", True, (True,), None, None),
        ("show_legend", False, (False,), None, None),
        ("show_legend", [True, False], (True, False), None, None),
        (
            "show_legend",
            "",
            None,
            TypeError,
            "`show_legend` needs to be of type bool or a list/tuple of type bool.",
        ),
        ("show_grid", True, True, None, None),
        ("show_grid", False, False, None, None),
        ("show_grid", "", None, TypeError, "`show_grid` needs to be of type bool."),
        ("x_label", "Test", "Test", None, None),
        (
            "x_label",
            [1, "Test"],
            None,
            TypeError,
            "`x_label` must be of type str or a list/tuple consisting of str values.",
        ),
        ("y_label", ["Test 1", "Test 2"], ("Test 1", "Test 2"), None, None),
        (
            "y_label",
            1,
            None,
            TypeError,
            "`y_label` must be of type str or a list/tuple consisting of str values.",
        ),
        ("x_range", [0.0, 10.0], (0.0, 10.0), None, None),
        ("x_range", [0.0, None], (0.0, None), None, None),
        ("y_range", None, None, None, None),
        ("y_range", [None, [0.0, 1.0]], (None, (0.0, 1.0)), None, None),
        (
            "y_range",
            [[0.0, 1.0, 1.0], [0.0, 1.0]],
            None,
            TypeError,
            "`y_range` must be a nested list/tuple or a list/tuple of two numbers.",
        ),
        ("y_range", "", None, TypeError, "`y_range` must be `None` or of type list/tuple."),
        ("custom_colors", ["C0", "C1", "#aaaaaa"], ("C0", "C1", "#aaaaaa"), None, None),
        ("custom_colors", "", None, TypeError, "`custom_colors` must be a list/tuple of colors."),
        (
            "custom_colors",
            ["C0", "123"],
            None,
            ValueError,
            "The color '123' has the wrong format.",
        ),
    ],
)
def test_properties_validation(prop, value, ref_value, error_type, error_m):
    """Test setting and validation of properties."""
    splot = SimplePlot()
    if error_type is None:
        setattr(splot, prop, value)
        assert getattr(splot, prop) == ref_value, f"{prop} validation gives wrong value."
    else:
        with pytest.raises(error_type) as error:
            setattr(splot, prop, value)
        assert str(error.value) == error_m


def test_data_labels_and_print():
    """Test data_labels property and __str__ function."""
    splot = SimplePlot()
    assert splot.__str__() == (
        "----------------------------------------------------------------------\n"
        "---------------------------- Simple Plot -----------------------------\n"
        "----------------------------------------------------------------------\n"
        " Data labels: not set.\n"
        "\n"
        "----------------------------------------------------------------------"
    )
    splot.import_scatter_data_set("test dataset", [0.0, 1.0, 2.0, 3.0], [-1.0, 2.0, 3.0, 0.0])
    splot.import_scatter_data_set(
        "test dataset 2", [0.0, 1.0, 2.0, 3.0], [4.0, 1.0, 3.0, 2.0], plot_label="Test 2"
    )
    assert splot.data_labels == ("test dataset", "test dataset 2"), "Wrong data_labels."
    assert splot.__str__() == (
        "----------------------------------------------------------------------\n"
        "---------------------------- Simple Plot -----------------------------\n"
        "----------------------------------------------------------------------\n"
        " Data labels: - test dataset\n"
        "              - test dataset 2\n"
        "\n"
        "----------------------------------------------------------------------"
    )


def test_gridspec():
    """Test creation and reset of gridspec values."""
    splot = SimplePlot()
    splot.create_default_gridspec(5, 2, 9, heights=2, widths=3, center_last_row=True)
    assert splot.subplot_nrows == 10, "nrows is wrong."
    assert splot.subplot_ncols == 12, "ncols is wrong."
    assert splot._subplot_gf_x == 6, "gf_x is wrong."
    assert splot._subplot_gf_y == 2, "gf_y is wrong."
    assert splot._subplot_gridspec_values == (
        (0, 2, 0, 6),
        (0, 2, 6, 12),
        (2, 4, 0, 6),
        (2, 4, 6, 12),
        (4, 6, 0, 6),
        (4, 6, 6, 12),
        (6, 8, 0, 6),
        (6, 8, 6, 12),
        (8, 10, 3, 9),
    ), "gridspec_values are wrong."
    splot.reset_gridspec()
    assert splot.subplot_nrows == 5, "nrows is wrong."
    assert splot.subplot_ncols == 2, "ncols is wrong."
    assert splot._subplot_gf_x == 1, "gf_x is wrong."
    assert splot._subplot_gf_y == 1, "gf_y is wrong."
    assert splot._subplot_gridspec_values is None, "gridspec_values are wrong."
    splot.create_default_gridspec(2, 2, 3, heights=2, widths=3, center_last_row=False)
    assert splot.subplot_nrows == 4, "nrows is wrong."
    assert splot.subplot_ncols == 6, "ncols is wrong."
    assert splot._subplot_gf_x == 3, "gf_x is wrong."
    assert splot._subplot_gf_y == 2, "gf_y is wrong."
    assert splot._subplot_gridspec_values == (
        (0, 2, 0, 3),
        (0, 2, 3, 6),
        (2, 4, 0, 3),
    ), "gridspec_values are wrong."
