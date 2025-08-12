"""
Sub-package containing units and unit conversions. ``'eV'`` and ``'angstrom'`` are set to ``1.0``
and the unit of time is ``ansgrom/sqrt(u/eV)`` per default.
However, other units can be set as base:

>>> length = Length(base_unit="m")
>>> length.m
1.0
"""

# Internal library imports
from aim2dat.units.constants import Constants
from aim2dat.units.quantities import Length, Energy, Force, Pressure, Frequency, Wavevector
from aim2dat.units.converters import UnitConverter

__all__ = ["Constants", "UnitConverter"]


constants = Constants()
length = Length()
energy = Energy()
force = Force()
pressure = Pressure()
frequency = Frequency()
wavevector = Wavevector()
