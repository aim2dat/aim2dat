"""
Module containing units and unit conversions. ``'eV'`` and ``'angstrom'`` are set to ``1.0``
and the unit of time is ``ansgrom/sqrt(u/eV)`` per default in the ``Quantity`` classes.
However, other units can be set as base:

>>> length = Length(base_unit="m")
>>> length.m
1.0
"""

# Standard library imports
import abc
from typing import Union, Tuple, List

# Third party library imports
import numpy as np

# Internal libraray imports
import aim2dat.utils.data as internal_data


class Constants:
    """Class to access fundamental constants."""

    def __init__(self, constants: Union[str, dict] = "CODATA_2022"):
        """initialize class."""
        if isinstance(constants, str):
            constants = internal_data.constants[constants]
        self._constants = constants

    def __getattr__(self, name: str):
        """
        Get value of constant.

        Parameters
        ----------
        name : str
            Name of the constant.

        Returns
        -------
        float
            Value of the constant.
        """
        return self._constants.get(name.lower(), None)

    def get_value(self, name: str) -> float:
        """
        Get value of constant.

        Parameters
        ----------
        name : str
            Name of the constant.

        Returns
        -------
        float
            Value of the constant.
        """
        return self.__getattr__(name)

    def get_unit(self, name: str) -> str:
        """
        Get unit of constant.

        Parameters
        ----------
        name : str
            Name of constant.

        Returns
        -------
        str
            Unit of the constant.
        """
        units = self._constants.get("units", {})
        return units.get(name.lower(), None)

    def get_value_unit(self, name: str) -> Tuple[float, str]:
        """
        Get value and unit of constant.

        Parameters
        ----------
        name : str
            Name of constant.

        Returns
        -------
        tuple
            Tuple containing value and unit of constant.
        """
        return self.get_value(name), self.get_unit(name)


class _BaseQuantity(abc.ABC):
    _plot_labels = {}

    def __init__(self, constants: Union[str, dict] = "CODATA_2022", base_unit: str = None):
        if isinstance(constants, str):
            constants = internal_data.constants[constants]
        self._derive_units(constants)
        if base_unit is not None:
            transf_val = self._units[base_unit]
            for k in self._units.keys():
                self._units[k] /= transf_val

    def __getitem__(self, name: str) -> float:
        return self._units.get(name.lower(), None)

    def __getattr__(self, name: str) -> float:
        return self[name]

    @property
    def available_units(self) -> List[str]:
        """
        List of all available units.
        """
        return list(self._units.keys())

    def get_unit(self, unit: str) -> float:
        """
        Return the value of the unit.

        Parameters
        ----------
        unit : str
            Physical unit.

        Returns
        -------
        float
            Value of the unit.
        """
        return self._units[unit.lower()]

    @abc.abstractmethod
    def _derive_units(self, constants: dict, base_unit: str):
        pass


class Length(_BaseQuantity):
    """
    Length units.
    """

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

    def _derive_units(self, constants: str):
        self._units = {
            "ang": 1.0,
            "angstrom": 1.0,
            "nm": 10.0,
            "micro_m": 1.0e4,
            "micron": 1.0e4,
            "mm": 1.0e7,
            "m": 1.0e10,
            "bohr": (4.0e10 * np.pi * constants["eps0"] * constants["hbar"] ** 2.0)
            / (constants["me"] * constants["e"] ** 2.0),
        }


class Energy(_BaseQuantity):
    """
    Energy units.
    """

    _plot_labels = {
        "rydberg": "Rydberg",
        "hartree": "Ha",
        "ha": "Ha",
        "joule": "Joule",
        "j": "Joule",
        "ev": "eV",
        "cal": "Cal",
    }

    def _derive_units(self, constants: str):
        self._units = {
            "ev": 1.0,
            "hartree": (constants["me"] * constants["e"] ** 3.0)
            / (16.0 * np.pi**2.0 * constants["eps0"] ** 2.0 * constants["hbar"] ** 2.0),
            "joule": 1.0 / constants["e"],
        }
        self._units["ha"] = self._units["hartree"]
        self._units["rydberg"] = self._units["hartree"] / 2.0
        self._units["j"] = self._units["joule"]
        self._units["cal"] = 4.184 * self._units["joule"]


class Force(_BaseQuantity):
    """Force units."""

    _plot_labels = {
        "ev_per_angstrom": r"eV $\mathrm{\AA}^{-1}$",
        "ev_per_ang": r"eV $\mathrm{\AA}^{-1}$",
        "hartree_per_bohr": r"Ha $\mathrm{Bohr}^{-1}$",
        "ha_per_bohr": r"Ha $\mathrm{Bohr}^{-1}$",
    }

    def _derive_units(self, constants: str):
        self._units = {
            "ev_per_angstrom": 1.0,
            "ev_per_ang": 1.0,
            "hartree_per_bohr": (constants["me"] ** 2.0 * constants["e"] ** 5.0)
            / (16.0 * 4.0e10 * np.pi**3.0 * constants["eps0"] ** 3.0 * constants["hbar"] ** 4.0),
        }
        self._units["ha_per_bohr"] = self._units["hartree_per_bohr"]


class Pressure(_BaseQuantity):
    """Pressure units."""

    _plot_labels = {
        "pa": "Pa",
        "pascal": "Pa",
        "bar": "bar",
        "atm": "atm",
    }

    def _derive_units(self, constants: str):
        self._units = {
            "pa": 1.0 / (constants["e"] * 1.0e30),
        }
        self._units["pascal"] = self._units["pa"]
        self._units["bar"] = self._units["pa"] * 1.0e5
        self._units["atm"] = self._units["pa"] * 1.01325e5


class Frequency(_BaseQuantity):
    """
    Frequency units.
    """

    _plot_labels = {
        "hz": "Hz",
        "khz": "kHz",
        "mhz": "MHz",
        "ghz": "GHz",
        "thz": "THz",
        "phz": "PHz",
    }

    def _derive_units(self, constants: str):
        self._units = {"hz": 1.0 / (1.0e10 * np.sqrt(constants["e"] / constants["am"]))}
        self._units["khz"] = 1.0e3 * self._units["hz"]
        self._units["mhz"] = 1.0e6 * self._units["hz"]
        self._units["ghz"] = 1.0e9 * self._units["hz"]
        self._units["thz"] = 1.0e12 * self._units["hz"]
        self._units["phz"] = 1.0e15 * self._units["hz"]


class Wavevector(_BaseQuantity):
    """
    Wavevector units.
    """

    _plot_labels = {
        "nm-1": r"nm$^{-1}$",
        "angstrom-1": r"$\mathrm{\AA}^{-1}$",
        "m-1": r"m$^{-1}$",
        "cm-1": r"cm$^{-1}$",
        "mm-1": r"mm$^{-1}$",
        "micro_m-1": r"$\mathrm{\mu}$m$^{-1}$",
    }

    def _derive_units(self, constants: str):
        self._units = {
            "angstrom-1": 1.0,
            "nm-1": 1.0e-1,
            "micro_m-1": 1.0e-4,
            "mm-1": 1.0e-7,
            "cm-1": 1.0e-8,
            "m-1": 1.0e-10,
        }


constants = Constants()
length = Length()
energy = Energy()
force = Force()
pressure = Pressure()
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
    def _return_quantity(cls, unit: str) -> Union[None, str]:
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
    def _convert_units(cls, value: Union[int, float], unit_1: str, unit_2: str) -> float:
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
    def _convert_energy_length(cls, value: Union[int, float], unit_1: str, unit_2: str) -> float:
        length = cls._quantities["length"]
        energy = cls._quantities["energy"]
        # E = h_planck * c / lambda
        if unit_1 not in length.available_units:
            unit_1, unit_2 = unit_2, unit_1
        conv_factor = (energy.Joule / energy.get_unit(unit_2)) * (
            length.m / length.get_unit(unit_1)
        )
        processed_data = (constants.c * constants.h * conv_factor) / value
        return processed_data

    @classmethod
    def _convert_frequency_length(
        cls, value: Union[int, float], unit_1: str, unit_2: str
    ) -> float:
        length = cls._quantities["length"]
        frequency = cls._quantities["frequency"]

        # f = c / lambda
        if unit_1 not in frequency.available_units:
            unit_1, unit_2 = unit_2, unit_1
        conv_factor = (length.m / length.get_unit(unit_2)) * (
            frequency.Hz / frequency.get_unit(unit_1)
        )
        processed_data = (constants.c * conv_factor) / value
        return processed_data

    @classmethod
    def _convert_length_wavevector(
        cls, value: Union[int, float], unit_1: str, unit_2: str
    ) -> float:
        length = cls._quantities["length"]
        wavevector = cls._quantities["wavevector"]

        # k = 2 * pi / lambda
        if unit_1 not in length.available_units:
            unit_1, unit_2 = unit_2, unit_1
        conv_factor = 2 * np.pi / (length.get_unit(unit_1) * wavevector.get_unit(unit_2))
        return conv_factor / value

    @classmethod
    def _convert_energy_frequency(
        cls, value: Union[int, float], unit_1: str, unit_2: str
    ) -> float:
        energy = cls._quantities["energy"]
        function1 = cls._convert_energy_length
        function2 = cls._convert_frequency_length
        if unit_1 not in energy.available_units:
            function1, function2 = function2, function1
        return function2(function1(value, unit_1, "angstrom"), "angstrom", unit_2)

    @classmethod
    def _convert_energy_wavevector(
        cls, value: Union[int, float], unit_1: str, unit_2: str
    ) -> float:
        energy = cls._quantities["energy"]
        function1 = cls._convert_energy_length
        function2 = cls._convert_length_wavevector
        if unit_1 not in energy.available_units:
            function1, function2 = function2, function1
        return function2(function1(value, unit_1, "angstrom"), "angstrom", unit_2)

    @classmethod
    def _convert_frequency_wavevector(
        cls, value: Union[int, float], unit_1: str, unit_2: str
    ) -> float:
        frequency = cls._quantities["frequency"]
        function1 = cls._convert_frequency_length
        function2 = cls._convert_length_wavevector
        if unit_1 not in frequency.available_units:
            function1, function2 = function2, function1
        return function2(function1(value, unit_1, "angstrom"), "angstrom", unit_2)

    @staticmethod
    def _convert_unit(
        value: Union[int, float], quantity: _BaseQuantity, unit_1: str, unit_2: str
    ) -> float:
        conv_factor = quantity.get_unit(unit_1) / quantity.get_unit(unit_2)
        return conv_factor * value


class UnitConverter(_BaseUnitConverter):
    """
    Convert units used in spectroscopy.
    """

    @classmethod
    def convert_units(cls, value: Union[int, float], unit_1: str, unit_2: str) -> float:
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
