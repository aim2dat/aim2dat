"""
Deprecated units module.
"""

# Internal library imports
import aim2dat.units as new_units


def Constants(*args, **kwargs):
    """Access fundamental constants."""
    from warnings import warn

    warn(
        "This class will be removed, please use `aim2dat.units.Constants` instead.",
        DeprecationWarning,
        2,
    )

    return new_units.Constants(*args, **kwargs)


def Length(*args, **kwargs):
    """Access length quantity."""
    from warnings import warn

    warn(
        "This class will be removed, please use `aim2dat.units.Length` instead.",
        DeprecationWarning,
        2,
    )
    return new_units.Length(*args, **kwargs)


def Energy(*args, **kwargs):
    """Access energy quantity."""
    from warnings import warn

    warn(
        "This class will be removed, please use `aim2dat.units.Energy` instead.",
        DeprecationWarning,
        2,
    )
    return new_units.Energy(*args, **kwargs)


def Force(*args, **kwargs):
    """Access force quantity."""
    from warnings import warn

    warn(
        "This class will be removed, please use `aim2dat.units.Force` instead.",
        DeprecationWarning,
        2,
    )
    return new_units.Force(*args, **kwargs)


def Pressure(*args, **kwargs):
    """Access pressure quantity."""
    from warnings import warn

    warn(
        "This class will be removed, please use `aim2dat.units.Pressure` instead.",
        DeprecationWarning,
        2,
    )
    return new_units.Pressure(*args, **kwargs)


def Frequency(*args, **kwargs):
    """Access frequency quantity."""
    from warnings import warn

    warn(
        "This class will be removed, please use `aim2dat.units.Frequency` instead.",
        DeprecationWarning,
        2,
    )
    return new_units.Frequency(*args, **kwargs)


def Wavevector(*args, **kwargs):
    """Access wavevector quantity."""
    from warnings import warn

    warn(
        "This class will be removed, please use `aim2dat.units.Wavevector` instead.",
        DeprecationWarning,
        2,
    )
    return new_units.Wavevector(*args, **kwargs)


constants = Constants()
length = Length()
energy = Energy()
force = Force()
pressure = Pressure()
frequency = Frequency()
wavevector = Wavevector()


def UnitConverter(*args, **kwargs):
    """Access unit converter."""
    from warnings import warn

    warn(
        "This class will be removed, please use `aim2dat.units.UnitConverter` instead.",
        DeprecationWarning,
        2,
    )
    return new_units.UnitConverter(*args, **kwargs)
