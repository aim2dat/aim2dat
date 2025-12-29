"""Functions to test the xsf parser."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import Structure, StructureCollection

cwd = os.path.dirname(__file__) + "/"
PATH = cwd + "xsf/"
STRUCTURES_PATH = cwd + "../strct/structures/"


def test_forces(tmpdir, structure_comparison):
    """Test reading/writing forces to xsf file."""
    strct = Structure.from_str("H2O")
    forces_wrong_type = [[0.3, "test", 0.03], [0.0, 1.0, 0.0], [1, 40.0, 0.0]]
    strct.set_site_attribute("forces", forces_wrong_type)
    with pytest.warns(UserWarning, match="Cannot parse force of site 0."):
        strct.to_file(str(tmpdir) + "H2O_test_wrong_forces.xsf", backend="internal")
    strct_read = Structure.from_file(str(tmpdir) + "H2O_test_wrong_forces.xsf", backend="internal")
    assert strct_read.site_attributes == {}

    strct._site_attributes = {}
    strct.set_site_attribute("test", [0, 1, 2])
    with pytest.warns(
        UserWarning,
        match="The current implementation of the 'xsf' file parser "
        + "only supports 'forces' as `site_attributes`.",
    ):
        strct.to_file(
            str(tmpdir) + "H2O_test_other_site_attributes.xsf",
            backend="internal",
            include_site_attributes=["test"],
        )
    strct_read = Structure.from_file(
        str(tmpdir) + "H2O_test_other_site_attributes.xsf", backend="internal"
    )
    assert strct_read.site_attributes == {}

    forces = [[0.3, 0.3, 0.03], [0.0, 1.0, 0.0], [1, 40.0, 0.0]]
    strct.set_site_attribute("forces", forces)
    strct.to_file(str(tmpdir) + "H2O_test.xsf", backend="internal")
    strct_read = Structure.from_file(str(tmpdir) + "H2O_test.xsf", backend="internal")
    structure_comparison(strct, strct_read)


def test_fixed_trajectory(tmpdir, structure_comparison):
    """Test reading/writing trajectory with fixed cell."""
    strct = Structure.from_file(STRUCTURES_PATH + "NaCl_225_prim.yaml")
    strct_c = StructureCollection()
    for i in range(3):
        strct0 = strct.copy()
        strct0.set_positions([[p + 0.2 * i for p in pos] for pos in strct0.get_positions()])
        strct_c.append_structure(strct0, f"strct_{i}")
    strct_c.to_file(str(tmpdir) + "test_fixed_traj.xsf", backend="internal")
    strct_c_read = StructureCollection.from_file(
        str(tmpdir) + "test_fixed_traj.xsf", backend="internal"
    )
    assert len(strct_c) == len(strct_c_read)
    for strct0, strct1 in zip(strct_c, strct_c_read):
        structure_comparison(strct0, strct1)


def test_var_trajectory(tmpdir, structure_comparison):
    """Test reading/writing trajectory with variable cell."""
    strct = Structure.from_file(STRUCTURES_PATH + "NaCl_225_prim.yaml")
    strct_c = StructureCollection()
    for i in range(3):
        strct0 = strct.copy()
        strct0.cell = [[v + 0.3 * i for v in vec] for vec in strct0.cell]
        strct_c.append_structure(strct0, f"strct_{i}")
    strct_c.to_file(str(tmpdir) + "test_var_cell_traj.xsf", backend="internal")
    strct_c_read = StructureCollection.from_file(
        str(tmpdir) + "test_var_cell_traj.xsf", backend="internal"
    )
    assert len(strct_c) == len(strct_c_read)
    for strct0, strct1 in zip(strct_c, strct_c_read):
        structure_comparison(strct0, strct1)
