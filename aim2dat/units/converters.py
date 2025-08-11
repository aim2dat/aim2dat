"""Module containing classes to convert one unit into another."""

# Standard library imports
import math
from typing import Union

# Internal library imports
from aim2dat.units.constants import Constants
from aim2dat.units.quantities import _BaseQuantity, Length, Energy, Frequency, Wavevector


class _BaseUnitConverter:
    """
    Convert units related to spectroscopy.
    """

    _constants = Constants()
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
        processed_data = (cls._constants.c * cls._constants.h * conv_factor) / value
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
        processed_data = (cls._constants.c * conv_factor) / value
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
        conv_factor = 2 * math.pi / (length.get_unit(unit_1) * wavevector.get_unit(unit_2))
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
