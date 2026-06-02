"""Tests for the Gaussian cube file parser."""

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import Structure
from aim2dat.io import read_gaussian_cube_file, write_gaussian_cube_file


def test_errors_and_warnings():
    """Test errors and warnings."""
    with pytest.raises(ValueError) as error:
        read_gaussian_cube_file("test", unit="test")
    assert (
        str(error.value)
        == "Invalid unit 'test'. "
        + "Supported values are ang, angstrom, nm, micro_m, micron, mm, m, bohr."
    )

    strct = Structure.from_str("H2O")
    with pytest.raises(ValueError) as error:
        write_gaussian_cube_file("test", strct)
    assert str(error.value) == "Missing 'cube' attribute, cannot write cube file."
    strct.set_attribute("cube", None)
    with pytest.raises(ValueError) as error:
        write_gaussian_cube_file("test", strct)
    assert str(error.value) == "Missing 'cube' attribute, cannot write cube file."
