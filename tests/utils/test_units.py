"""Test units module."""

# Third party library imports
import pytest

# Internal library imports
from aim2dat.utils.units import UnitConverter


@pytest.mark.parametrize(
    "inp_unit, inp_value, out_unit, out_value",
    [
        ("m", 1.0, "mm", 1000.0),
        ("angstrom", 23.0, "ev", 539.0617278104661),
        ("hartree", 10.0, "angstrom", 45.56335251919246),
        ("eV", 1.0, "micro_m", 1.239841973964072),
        ("micro_m", 1.239841973964072, "eV", 1.0),
        ("kHz", 2.0, "m", 149896.229),
        ("m", 0.057652395769230765, "GHz", 5.2),
        ("m", 13.2, "m-1", 0.47599888690754444),
        ("m-1", 0.47599888690754444, "m", 13.2),
        ("eV", 1.0, "THz", 241.79892623048698),
        ("THz", 241.79892623048698, "ev", 1.0),
        ("eV", 1.0, "micro_m-1", 5.06773075853428),
        ("micro_m-1", 5.06773075853428, "eV", 1.0),
        ("THz", 1.0, "micro_m-1", 0.020958450219516814),
        ("micro_m-1", 0.020958450219516814, "THz", 1.0),
    ],
)
def test_unit_converter(inp_unit, inp_value, out_unit, out_value):
    """Test conversion of units."""
    assert out_value == UnitConverter.convert_units(inp_value, inp_unit, out_unit)


def test_unit_converter_unit_list():
    """Test available units list."""
    assert UnitConverter.available_units == [
        "bohr",
        "nm",
        "ang",
        "angstrom",
        "m",
        "mm",
        "micro_m",
        "micron",
        "rydberg",
        "hartree",
        "joule",
        "j",
        "ev",
        "hz",
        "khz",
        "mhz",
        "ghz",
        "thz",
        "phz",
        "nm-1",
        "angstrom-1",
        "m-1",
        "cm-1",
        "mm-1",
        "micro_m-1",
    ]


def test_unit_converter_invalid_unit():
    """Test invalid input for the UnitConverter."""
    with pytest.raises(ValueError) as error:
        UnitConverter.convert_units(1.0, "wrong_unit", "nm")
    assert str(error.value) == "'wrong_unit' is not supported for unit conversion."
