"""Test mixin classes for plots."""

# Standard library imports
import os

# Internal library imports
from aim2dat.io import read_yaml_file
from aim2dat.plots import SimplePlot


REF_PATH = os.path.dirname(__file__) + "/mixin/"


def test_line_mixin(create_plot_object, nested_dict_comparison, matplotlib_figure_comparison):
    """Test lines mixin classes for plots."""
    inp_dict = dict(read_yaml_file(REF_PATH + "lines_input.yaml"))
    ref = dict(read_yaml_file(REF_PATH + "lines_ref.yaml"))
    splot = create_plot_object(SimplePlot, "matplotlib", inp_dict)
    nested_dict_comparison(splot._plot_extras_lines_background, ref["extras_lines_background"])
    nested_dict_comparison(splot._plot_extras_lines_foreground, ref["extras_lines_foreground"])
    matplotlib_figure_comparison(splot.plot(**ref["plot_args"]), ref["with_lines"])
    splot.remove_additional_plot_elements()
    assert splot._plot_extras_lines_foreground == {}, "Removing lines does not work."
    assert splot._plot_extras_lines_background == {}, "Removing lines does not work."
    matplotlib_figure_comparison(splot.plot(**ref["plot_args"]), ref["without_lines"])
