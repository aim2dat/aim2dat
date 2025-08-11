"""
Module to plot spectroscopy data.
"""

# Internal imports
from aim2dat.plots.base_plot import _BasePlot
from aim2dat.plots.base_mixin import _VLineMixin, _PeakDetectionMixin, _SmearingMixin
from aim2dat.units import UnitConverter

# from aim2dat.structure_importer.online_databases import MPImporter


def Spectrum(*args, **kwargs):
    """Depreciated Spectrum class."""
    from warnings import warn

    warn(
        "This class will be removed, please use `SpectrumPlot` instead.",
        DeprecationWarning,
        2,
    )
    return SpectrumPlot(*args, **kwargs)


class SpectrumPlot(_BasePlot, _VLineMixin, _PeakDetectionMixin, _SmearingMixin):
    """
    Plot x-ray absorption spectra.

    Attributes
    ----------
    detect_peaks : bool
        Whether to detect peaks of the spectra.
    smooth_spectra : bool
        Whether to broaden the spectra.
    plot_original_spectra : bool
        Whether to plot the original spectra (in addition to the broadened spectra).
    plot_unit_x : str
        Unit of the x-values. Imported spectra are transformed to the corresponding unit if the
        unit of the imported data set is given.
    """

    def __init__(
        self,
        detect_peaks=False,
        smooth_spectra=False,
        plot_original_spectra=False,
        plot_unit_x="eV",
        **kwargs,
    ):
        """Initialize class."""
        # I would put an attribute for the peak detection
        self.detect_peaks = detect_peaks

        # Some attributes for smoothing the spectra:
        self.smooth_spectra = smooth_spectra
        self.plot_original_spectra = plot_original_spectra
        self.plot_unit_x = plot_unit_x

        _BasePlot.__init__(self, **kwargs)

    @property
    def plot_unit_x(self):
        """Set unit of x-axis."""
        return self._plot_unit_x

    @plot_unit_x.setter
    def plot_unit_x(self, value):
        if value.lower() not in UnitConverter.available_units:
            raise ValueError(f"{value} as unit not supported.")
        self._plot_unit_x = value.lower()

    def import_spectrum(self, data_label, x_values, y_values, unit_x):
        """
        Import spectrum.

        Parameters
        ----------
        data_label : str
            Internal label of the data set.
        x_values : list
            x-values of the spectrum.
        y_values : list
            y-values of the spectrum.
        unit_x : str
            Unit of the x-values.
        """
        self._check_data_label(data_label)

        if unit_x.lower() not in UnitConverter.available_units:
            raise ValueError(f"{unit_x} as unit not supported.")

        self._data[data_label] = {"x_values": x_values, "y_values": y_values, "unit_x": unit_x}

    def _prepare_to_plot(self, data_labels, subplot_assignment):
        quantity = UnitConverter._available_units[self.plot_unit_x].capitalize()
        unit = UnitConverter.plot_labels[self.plot_unit_x]
        self._auto_set_axis_properties(
            x_label=f"{quantity} in {unit}", y_label=r"Absorption Coeff. $\mu$"
        )

        data_sets = [[] for idx0 in range(max(subplot_assignment) + 1)]
        for data_label, subp_a in zip(data_labels, subplot_assignment):
            data_set = self._process_data_set(data_label, self.plot_unit_x)
            data_sets[subp_a] += data_set
        return data_sets, None, None, None, None, None

    def _process_data_set(self, data_label, plot_unit_x):
        """Process data set to be plotted."""
        data_sets_o = []
        data_set = self._return_data_set(data_label)
        x_values = data_set["x_values"]
        y_values = data_set["y_values"]

        # Check if we have to transform unit_x
        if data_set["unit_x"] != plot_unit_x:
            x_values = [
                UnitConverter.convert_units(x_value, data_set["unit_x"], plot_unit_x)
                for x_value in x_values
            ]

        # Check for peaks:
        if self.detect_peaks:
            self._find_peaks(x_values, y_values, data_label)
        # Check whether to plot the original unsmoothed data:
        if not self.smooth_spectra or self.plot_original_spectra:
            data_sets_o.append({"x_values": x_values, "y_values": y_values})

        # Check if data should be smoothed
        if self.smooth_spectra:
            x_values, y_values = self._apply_smearing(x_values, y_values)
            data_sets_o.append({"x_values": x_values, "y_values": y_values})

        return data_sets_o
