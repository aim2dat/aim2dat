"""Module containing units and unit conversions."""

# Third party library imports
import numpy as np
import ase.units as ase_un


class _BaseQuantity:
    _units = {}

    def __init__(self):
        pass

    def __getattr__(self, name):
        return self._units.get(name.lower(), None)

    @property
    def available_units(self):
        """
        List of all available units.
        """
        return list(self._units.keys())

    def get_unit(self, unit):
        """
        Return the value of the unit.

        Parameters
        ----------
        unit : str
            Physical unit.

        Returns
        -------
        value : float
            Value of the unit.
        """
        return self._units[unit.lower()]


class Length(_BaseQuantity):
    """
    Length units based on the ase library. Angstrom is set to ``1.0``.
    """

    _units = {
        "bohr": ase_un.Bohr,
        "nm": ase_un.nm,
        "ang": ase_un.Angstrom,
        "angstrom": ase_un.Angstrom,
        "m": ase_un.m,
        "mm": ase_un.m * 1e-3,
        "micro_m": ase_un.m * 1e-6,
        "micron": ase_un.m * 1e-6,
    }
    _plot_labels = {
        "bohr": "Bohr",
        "nm": "nm",
        "ang": r"$\mathrm{\AA}$",
        "angstrom": r"$\mathrm{\AA}$",
        "m": "m",
        "mm": "mm",
        "micro_m": r"$\mathrm{\mu}$m",
        "micron": r"$\mathrm{\mu}$m",
    }


class Energy(_BaseQuantity):
    """
    Energy units based on the ase library. eV is set to ``1.0``.
    """

    _units = {
        "rydberg": ase_un.Rydberg,
        "hartree": ase_un.Hartree,
        "joule": ase_un.J,
        "j": ase_un.J,
        "ev": ase_un.eV,
    }

    _plot_labels = {
        "rydberg": "Rydberg",
        "hartree": "Hartree",
        "joule": "Joule",
        "j": "Joule",
        "ev": "eV",
    }


class Frequency(_BaseQuantity):
    """
    Frequency units based on the ase library.
    """

    _units = {
        "hz": 1.0 / ase_un.second,
        "khz": 1e3 / ase_un.second,
        "mhz": 1e6 / ase_un.second,
        "ghz": 1e9 / ase_un.second,
        "thz": 1e12 / ase_un.second,
        "phz": 1e15 / ase_un.second,
    }

    _plot_labels = {
        "hz": "Hz",
        "khz": "kHz",
        "mhz": "MHz",
        "ghz": "GHz",
        "thz": "THz",
        "phz": "PHz",
    }


class Wavevector(_BaseQuantity):
    """
    Wavevector units based on the ase library. Angstrom-1 is set to ``1.0``.
    """

    _units = {
        "nm-1": 1.0 / ase_un.nm,
        "angstrom-1": 1 / ase_un.Angstrom,
        "m-1": 1.0 / ase_un.m,
        "cm-1": 1.0 / (ase_un.m * 1e-2),
        "mm-1": 1.0 / (ase_un.m * 1e-3),
        "micro_m-1": 1.0 / (ase_un.m * 1e-6),
    }

    _plot_labels = {
        "nm-1": r"nm$^{-1}$",
        "angstrom-1": r"$\mathrm{\AA}^{-1}$",
        "m-1": r"m$^{-1}$",
        "cm-1": r"cm$^{-1}$",
        "mm-1": r"mm$^{-1}$",
        "micro_m-1": r"$\mathrm{\mu}$m$^{-1}$",
    }


length = Length()
energy = Energy()
frequency = Frequency()
wavevector = Wavevector()


class _BaseUnitConverter:
    """
    Convert units related to spectroscopy.
    """

    _quantities = {
        "length": Length(),
        "energy": Energy(),
        "frequency": Frequency(),
        "wavevector": Wavevector(),
    }
    _available_units = {}
    for quantity, qu_class in _quantities.items():
        for unit in qu_class.available_units:
            _available_units[unit] = quantity
    available_units = list(_available_units.keys())
    plot_labels = {}
    for qu_class in _quantities.values():
        plot_labels.update(qu_class._plot_labels)

    @classmethod
    def _return_quantity(cls, unit):
        """
        Return the quantity of the unit.

        Parameters
        ----------
        unit : str
            Physical unit.

        Returns
        -------
        quantity : str
            Physical quantity.
        """
        for quantity_label, quantity in cls._quantities.items():
            if unit in quantity.available_units:
                return quantity_label
        return None

    @classmethod
    def _convert_units(cls, value, unit_1, unit_2):
        """
        Convert one unit into another.

        Parameters
        ----------
        value : float
            Input value.
        unit_1 : str
            Physical unit of the input value.
        unit_2 : str
            Physical unit to be converted into.

        Returns
        -------
        processed_data : float
            Output value.
        """
        unit_1 = unit_1.lower()
        unit_2 = unit_2.lower()
        for unit in [unit_1, unit_2]:
            if unit not in cls._available_units.keys():
                raise ValueError(f"'{unit}' is not supported for unit conversion.")

        # If both units are from the same quantity:
        if cls._available_units[unit_1] == cls._available_units[unit_2]:
            quantity_class = cls._quantities[cls._available_units[unit_1]]
            processed_data = cls._convert_unit(value, quantity_class, unit_1, unit_2)
        else:
            quantities = [cls._available_units[unit_1], cls._available_units[unit_2]]
            method_name = "_".join(sorted(quantities))
            conv_method = getattr(cls, "_convert_" + method_name)
            processed_data = conv_method(value=value, unit_1=unit_1, unit_2=unit_2)
        return processed_data

    @classmethod
    def _convert_energy_length(cls, value, unit_1, unit_2):
        length = cls._quantities["length"]
        energy = cls._quantities["energy"]
        # E = h_planck * c / lambda
        if unit_1 not in length.available_units:
            unit_1, unit_2 = unit_2, unit_1
        conv_factor = (energy.Joule / energy.get_unit(unit_2)) * (
            length.m / length.get_unit(unit_1)
        )
        # We should find here a different way without using private variables from ase...
        processed_data = (ase_un._c * ase_un._hplanck * conv_factor) / value
        return processed_data

    @classmethod
    def _convert_frequency_length(cls, value, unit_1, unit_2):
        length = cls._quantities["length"]
        frequency = cls._quantities["frequency"]

        # f = c / lambda
        if unit_1 not in frequency.available_units:
            unit_1, unit_2 = unit_2, unit_1
        conv_factor = (length.m / length.get_unit(unit_2)) * (
            frequency.Hz / frequency.get_unit(unit_1)
        )
        processed_data = (ase_un._c * conv_factor) / value
        return processed_data

    @classmethod
    def _convert_length_wavevector(cls, value, unit_1, unit_2):
        length = cls._quantities["length"]
        wavevector = cls._quantities["wavevector"]

        # k = 2 * pi / lambda
        if unit_1 not in length.available_units:
            unit_1, unit_2 = unit_2, unit_1
        conv_factor = 2 * np.pi / (length.get_unit(unit_1) * wavevector.get_unit(unit_2))
        return conv_factor / value

    @classmethod
    def _convert_energy_frequency(cls, value, unit_1, unit_2):
        energy = cls._quantities["energy"]
        function1 = cls._convert_energy_length
        function2 = cls._convert_frequency_length
        if unit_1 not in energy.available_units:
            function1, function2 = function2, function1
        return function2(function1(value, unit_1, "angstrom"), "angstrom", unit_2)

    @classmethod
    def _convert_energy_wavevector(cls, value, unit_1, unit_2):  #
        energy = cls._quantities["energy"]
        function1 = cls._convert_energy_length
        function2 = cls._convert_length_wavevector
        if unit_1 not in energy.available_units:
            function1, function2 = function2, function1
        return function2(function1(value, unit_1, "angstrom"), "angstrom", unit_2)

    @classmethod
    def _convert_frequency_wavevector(cls, value, unit_1, unit_2):
        frequency = cls._quantities["frequency"]
        function1 = cls._convert_frequency_length
        function2 = cls._convert_length_wavevector
        if unit_1 not in frequency.available_units:
            function1, function2 = function2, function1
        return function2(function1(value, unit_1, "angstrom"), "angstrom", unit_2)

    @staticmethod
    def _convert_unit(value, quantity, unit_1, unit_2):
        conv_factor = quantity.get_unit(unit_1) / quantity.get_unit(unit_2)
        return conv_factor * value


class UnitConverter(_BaseUnitConverter):
    """
    Convert units used in spectroscopy.
    """

    @classmethod
    def convert_units(cls, value, unit_1, unit_2):
        """
        Convert one unit into another.

        Parameters
        ----------
        value : float
            Input value.
        unit_1 : str
            Physical unit of the input value.
        unit_2 : str
            Physical unit to be converted into.

        Returns
        -------
        processed_data : float
            Output value.
        """
        return cls._convert_units(value, unit_1, unit_2)
