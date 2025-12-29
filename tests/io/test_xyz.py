"""Functions to test the xyz parser."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.io import read_xyz_file
from aim2dat.strct import Structure

cwd = os.path.dirname(__file__) + "/"
PATH = cwd + "xyz/"
STRUCTURES_PATH = cwd + "../strct/structures/"


def test_read_errors():
    """Test errors of the `read_xyz_file` function."""
    with pytest.raises(ValueError) as error:
        read_xyz_file(PATH + "error_lattice.xyz")
    assert str(error.value) == "'Lattice' needs to have 9 numbers separated by space."
    with pytest.raises(ValueError) as error:
        read_xyz_file(PATH + "error_pbc.xyz")
    assert str(error.value) == "'pbc' needs to have 3 booleans separated by space."
    with pytest.raises(ValueError) as error:
        read_xyz_file(PATH + "error_properties.xyz")
    assert (
        str(error.value) == "'Properties' needs to have a multiple of 3 entries separated by ':'."
    )


def test_write_warnings(tmpdir):
    """Test warnings of the `write_xyz_file` function."""
    strct = Structure.from_str("H2O")

    strct.set_site_attribute("test", [1, [0, 1], 2])
    with pytest.warns(UserWarning, match="Cannot add 'test' due to length mismatch."):
        strct.to_file("test.xyz")

    strct.set_site_attribute("test", [{0: 1}, 1, 2])
    with pytest.warns(
        UserWarning,
        match="Cannot add 'test' since the values cannot be cast into str, int or float.",
    ):
        strct.to_file("test.xyz")

    strct.set_site_attribute("test", ["f", "t f", "t"])
    with pytest.warns(
        UserWarning, match="Cannot add 'test' since the values contain white spaces."
    ):
        strct.to_file("test.xyz")

    strct.set_site_attribute("test", [1, {0: 1}, 2])
    with pytest.warns(
        UserWarning, match="Cannot add 'test' since the values have different types."
    ):
        strct.to_file("test.xyz")

    strct.site_attributes = None
    strct.set_attribute("test_ignored", "i + 0")
    strct.set_attribute("test", "i + 0")
    with pytest.warns(
        UserWarning, match="Cannot add attribute 'test' since the value contains white spaces."
    ):
        strct.to_file("test.xyz", include_attributes=["test"])
    with pytest.warns(
        UserWarning,
        match="Cannot add attribute 'test_ignored' since the value contains white spaces.",
    ):
        strct.to_file("test.xyz", exclude_attributes=["test"])
