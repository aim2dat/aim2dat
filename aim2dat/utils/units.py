"""
Deprecated units module.
"""

# Internal library imports
import aim2dat.units as new_units

from warnings import warn

warn(
    "This module will be removed, please use `aim2dat.units.Constants` instead.",
    DeprecationWarning,
    2,
)


class Constants(new_units.Constants):
    """Access fundamental constants."""

    pass


class Length(new_units.Length):
    """Access length quantity."""

    pass


class Energy(new_units.Energy):
    """Access energy quantity."""

    pass


class Force(new_units.Force):
    """Access force quantity."""

    pass


class Pressure(new_units.Pressure):
    """Access pressure quantity."""

    pass


class Frequency(new_units.Frequency):
    """Access frequency quantity."""

    pass


class Wavevector(new_units.Wavevector):
    """Access wavevector quantity."""

    pass


constants = Constants()
length = Length()
energy = Energy()
force = Force()
pressure = Pressure()
frequency = Frequency()
wavevector = Wavevector()


class UnitConverter(new_units.UnitConverter):
    """Deprecated UnitConverter class."""

    pass
