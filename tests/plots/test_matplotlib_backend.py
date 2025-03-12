"""
Test matplotlib backend implementation.

TODO:
* Improve pytest fixture
* equal_aspect_ratio
* subplots
* etc.
"""

# Standard library imports
import os

# Internal library imports
from aim2dat.io import read_yaml_file
from aim2dat.plots import SimplePlot


REF_PATH = os.path.dirname(__file__) + "/matplotlib/"


def test_single_plot(create_plot_object, matplotlib_figure_comparison):
    """Test single plot."""
    inp_dict = dict(read_yaml_file(REF_PATH + "simple_plot_w_legend.yaml"))
    splot = create_plot_object(SimplePlot, "matplotlib", inp_dict)
    figure = splot.plot([inp["args"]["data_label"] for inp in inp_dict["import_data"]])
    matplotlib_figure_comparison(figure, inp_dict["ref"])
