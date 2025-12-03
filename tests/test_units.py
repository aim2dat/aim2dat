"""Test units module."""

# Third party library imports
import pytest

# Internal library imports
from aim2dat.units import UnitConverter, Constants, Length


def test_constants():
    """Test constants class."""
    constants = Constants(
        constants={"a": 10.0, "b": 20.0, "units": {"a": "test_unit", "b": "test_unit2"}}
    )
    assert constants.A == 10.0
    assert constants.get_value("a") == 10.0
    assert constants.get_unit("B") == "test_unit2"
    assert constants.get_value_unit("a") == (10.0, "test_unit")


def test_units():
    """Test units class."""
    length = Length(constants={"eps0": 1.0, "hbar": 1.0, "me": 1.0, "e": 1.0}, base_unit="nm")
    assert abs(length.nm - 1.0) < 1.0e-12
    assert abs(length.bohr - 12566370614.359173) < 1.0e-12


@pytest.mark.parametrize(
    "inp_unit, inp_value, out_unit, out_value",
    [
        ("m", 1.0, "mm", 1000.0),
        ("angstrom", 23.0, "ev", 539.061732318262),
        ("hartree", 10.0, "angstrom", 45.563352473376504),
        ("hartree", -0.001834, "kj_per_mol", -4.815166344697022),
        ("eV", 1.0, "micro_m", 1.2398419843320025),
        ("micro_m", 1.2398419843320025, "eV", 1.0),
        ("kHz", 2.0, "m", 149896.229),
        ("m", 0.057652395769230765, "GHz", 5.2),
        ("m", 13.2, "m-1", 0.47599888690754444),
        ("m-1", 0.47599888690754444, "m", 13.2),
        ("eV", 1.0, "THz", 241.79892420849183),
        ("THz", 241.79892420849183, "ev", 1.0),
        ("eV", 1.0, "micro_m-1", 5.067730716156396),
        ("micro_m-1", 5.067730716156396, "eV", 1.0),
        ("THz", 1.0, "micro_m-1", 0.020958450219516814),
        ("micro_m-1", 0.020958450219516814, "THz", 1.0),
    ],
)
def test_unit_converter(inp_unit, inp_value, out_unit, out_value):
    """Test conversion of units."""
    assert abs(UnitConverter.convert_units(inp_value, inp_unit, out_unit) - out_value) < 1.0e-12


def test_unit_converter_unit_list():
    """Test available units list."""
    assert UnitConverter.available_units == [
        "ang",
        "angstrom",
        "nm",
        "micro_m",
        "micron",
        "mm",
        "m",
        "bohr",
        "ev",
        "hartree",
        "joule",
        "ha",
        "rydberg",
        "j",
        "cal",
        "kj_per_mol",
        "hz",
        "khz",
        "mhz",
        "ghz",
        "thz",
        "phz",
        "angstrom-1",
        "nm-1",
        "micro_m-1",
        "mm-1",
        "cm-1",
        "m-1",
    ]


def test_unit_converter_invalid_unit():
    """Test invalid input for the UnitConverter."""
    with pytest.raises(ValueError) as error:
        UnitConverter.convert_units(1.0, "wrong_unit", "nm")
    assert str(error.value) == "'wrong_unit' is not supported for unit conversion."
