"""Functions to test the cif parser."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.io import read_cif_file, read_yaml_file
from aim2dat.strct import Structure

cwd = os.path.dirname(__file__) + "/"
PATH = cwd + "cif/"
STRUCTURES_PATH = cwd + "../strct/structures/"


def test_errors_and_warnings(structure_comparison):
    """Test errors and warnings."""
    with pytest.raises(ValueError) as error:
        read_cif_file(PATH + "error_chem_f.cif", extract_structures=True)
    assert str(error.value) == "Chemical formula doesn't match with number of sites."

    with pytest.raises(ValueError) as error:
        read_cif_file(PATH + "error_determine_el.cif", extract_structures=True)
    assert str(error.value) == "Could not determine element of 'abc'."

    with pytest.raises(ValueError) as error:
        read_cif_file(PATH + "error_loop.cif", extract_structures=True)
    assert str(error.value) == "Number of values differ for loop finishing on line 7."

    with pytest.warns(UserWarning, match="Two data bloocks have the same title: 'GaAs_216_conv'."):
        read_cif_file(PATH + "warning_same_title.cif", extract_structures=True)
    with pytest.warns(UserWarning, match="Data block 'structures' is overwritten."):
        read_cif_file(PATH + "warning_structures.cif", extract_structures=True)
    with pytest.warns(
        UserWarning,
        match="Could not determine symmetry operations directly, using space group details.",
    ):
        read_cif_file(PATH + "warning_space_group.cif", extract_structures=True)
    with pytest.warns(
        UserWarning,
        match=r"The sites \{8, 9, 10\} are omitted as they are duplicate of other sites.",
    ):
        outp_dict = read_cif_file(PATH + "warning_duplicate_sites.cif", extract_structures=True)
        strct = Structure(**outp_dict["structures"][0])
        ref_strct = read_yaml_file(STRUCTURES_PATH + f"{strct.label}.yaml")
        ref_strct["label"] = strct.label
        structure_comparison(strct, ref_strct)


def test_loops():
    """Test different loop constructions."""
    outp_dict = read_cif_file(PATH + "loop_test.cif")
    assert outp_dict == {
        "Test": {
            "loops": [
                {
                    "citation_id": ["primary", "primary", "primary"],
                    "citation_journal_full": ["\nTest Journal\n", "Test", "Test2\n"],
                    "citation_year": [1974, 2000, 2001],
                    "citation_journal_volume": [30, 31, 45],
                    "citation_page_first": [1481, 14, 24],
                    "citation_page_last": [1484, 20, 450],
                    "citation_journal_id_astm": ["ACBCAR", "ACBC", "DDD"],
                },
                {
                    "platon_squeeze_void_nr": [1],
                    "platon_squeeze_void_average_x": [-0.01],
                    "platon_squeeze_void_average_y": [-0.025],
                    "platon_squeeze_void_average_z": [-0.5],
                    "platon_squeeze_void_volume": [10],
                    "platon_squeeze_void_count_electrons": [20],
                    "platon_squeeze_void_content": [""],
                },
                {
                    "symmetry_equiv_pos_site_id": [1, 2],
                    "symmetry_equiv_pos_as_xyz": ["x,y,z", "1/2+x,1/2-y,1/2-z"],
                },
            ]
        }
    }


def test_extract_structures(structure_comparison):
    """Test the parsing of crystal structures."""
    outp_dict = read_cif_file(PATH + "crystals.cif", extract_structures=True)
    structures = [Structure(**strct) for strct in outp_dict["structures"]]
    for strct in structures:
        ref_strct = read_yaml_file(STRUCTURES_PATH + f"{strct.label}.yaml")
        ref_strct["label"] = strct.label
        structure_comparison(strct, ref_strct)


def test_extract_structures_with_site_attributes(structure_comparison):
    """Test parsing of site attributes and strct parameters."""
    outp_dict = read_cif_file(
        PATH + "crystals_with_site_attributes.cif",
        extract_structures=True,
        strct_check_chem_formula=False,
        strct_get_sym_op_from_sg=False,
    )
    structure = Structure(**outp_dict["structures"][0])
    ref_strct = read_yaml_file(STRUCTURES_PATH + f"{structure.label}.yaml")
    ref_strct["label"] = structure.label
    structure_comparison(structure, ref_strct)
    assert structure.site_attributes == {
        "atom_site_occupancy": (0.5, 0.5, 1.0, 0.5, 0.2, 0.4, 0.2, 0.1)
    }
