"""
Test plotly backend implementation.

TODO: Improve pytest fixture
"""

# Standard library imports
import os

# Internal library imports
from aim2dat.io import read_yaml_file
from aim2dat.plots import SimplePlot


REF_PATH = os.path.dirname(__file__) + "/plotly/"


def test_single_plot(create_plot_object, plotly_figure_comparison):
    """Test single plot."""
    inp_dict = dict(read_yaml_file(REF_PATH + "simple_plot.yaml"))
    splot = create_plot_object(SimplePlot, "plotly", inp_dict)
    figure = splot.plot([inp["args"]["data_label"] for inp in inp_dict["import_data"]])
    plotly_figure_comparison(figure, inp_dict["ref"])
