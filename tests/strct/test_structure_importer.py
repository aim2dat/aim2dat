"""Test the StructureCollection class."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import StructureImporter
from aim2dat.io import read_yaml_file

STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
REF_PATH = os.path.dirname(__file__) + "/structure_importer/"


def test_print_and_constraints():
    """Test different constraints."""
    # Empty print
    strct_import = StructureImporter()
    assert strct_import.__str__() == (
        "----------------------------------------------------------------------\n"
        "------------------------ Structure Collection ------------------------\n"
        "----------------------------------------------------------------------\n"
        "\n"
        "----------------------------------------------------------------------\n"
        "\n"
        "                     Chemical element constraints                     \n"
        "   Neglecting elemental structures: False\n"
        "\n"
        "                     Chemical formula constraints                     \n"
        "   Not set.\n"
        "\n"
        "                        Attribute constraints                         \n"
        "   Not set.\n"
        "\n"
        "----------------------------------------------------------------------"
    )

    # Constraint errors:
    with pytest.raises(ValueError) as error:
        strct_import.set_concentration_constraint("H", min_conc=-0.8, max_conc=0.6)
    assert str(error.value) == "`min_conc` and `max_conc` need to be inbetween 0.0 and 1.0."
    with pytest.raises(ValueError) as error:
        strct_import.set_concentration_constraint("H", min_conc=0.8, max_conc=0.6)
    assert str(error.value) == "`max_conc` needs to be larger than `min_conc`."
    with pytest.raises(TypeError) as error:
        strct_import.add_chem_formula_constraint(10.0)
    assert str(error.value) == "`chem_formula` needs to be string, dict or list."
    with pytest.raises(ValueError) as error:
        strct_import.set_attribute_constraint("band_gap", min_value=0.8, max_value=0.2)
    assert str(error.value) == "`max_value` needs to be equal or larger than `min_value`."

    # Check concentration constraints:
    strct_import.set_concentration_constraint("Cs", max_conc=0.9)
    strct_import.set_concentration_constraint("Te", min_conc=0.6, max_conc=0.7)
    assert strct_import.concentration_constraints == {"Cs": [0.0, 0.9], "Te": [0.6, 0.7]}
    strct_import.import_from_oqmd("Cs-Te")

    assert len(strct_import.structures) == 3
    strct_import.remove_constraints()
    assert not strct_import.neglect_elemental_structures
    assert strct_import.concentration_constraints == {}
    assert strct_import.chem_formula_constraints == []
    assert strct_import.attribute_constraints == {}

    strct_import.neglect_elemental_structures = True
    strct_import.import_from_oqmd("Cs")
    assert len(strct_import.structures) == 3

    # Check chem formula constraints:
    strct_import.add_chem_formula_constraint("H6P2O8")
    strct_import.import_from_oqmd("Cs2Te")
    assert strct_import.chem_formula_constraints == [
        {"formula": {"H": 3.0, "P": 1.0, "O": 4.0}, "is_reduced": True}
    ]
    strct_import.add_chem_formula_constraint("Cs8Te4", reduced_formula=False)
    assert strct_import.chem_formula_constraints == [
        {"formula": {"H": 3.0, "P": 1.0, "O": 4.0}, "is_reduced": True},
        {"formula": {"Cs": 8.0, "Te": 4.0}, "is_reduced": False},
    ]
    strct_import.import_from_oqmd("Cs2Te")
    assert len(strct_import.structures) == 6
    strct_import.remove_constraints()
    strct_import.import_from_oqmd("Cs")
    assert len(strct_import.structures) == 41
    strct_import.add_chem_formula_constraint("Cs-Te")
    strct_import.import_from_oqmd("Cs-Sb")
    assert len(strct_import.structures) == 41

    # Check attribute constraints
    strct_import.remove_constraints()
    strct_import.set_attribute_constraint("band_gap", 0.15, 0.5)
    strct_import.import_from_oqmd("Cs3Sb")
    assert len(strct_import.structures) == 42

    strct_import.add_chem_formula_constraint("H6P2O8")
    strct_import.add_chem_formula_constraint("Cs8Te4", reduced_formula=False)
    strct_import.set_concentration_constraint("Te", min_conc=0.6, max_conc=0.7)
    assert strct_import._import_details == {"oqmd": [42, ["Cs", "Sb", "Te"]]}
    assert strct_import.__str__() == (
        "----------------------------------------------------------------------\n"
        "------------------------ Structure Collection ------------------------\n"
        "----------------------------------------------------------------------\n"
        "\n"
        "                         Imported from: oqmd                          \n"
        "   - Number of structures: 42\n"
        "   - Elements: Cs-Sb-Te\n"
        "\n"
        "----------------------------------------------------------------------\n"
        "\n"
        "                     Chemical element constraints                     \n"
        "   Neglecting elemental structures: False\n"
        "   Te: - min: 0.6\n"
        "       - max: 0.7\n"
        "\n"
        "                     Chemical formula constraints                     \n"
        "   - H3PO4 (reduced)\n"
        "   - Cs8Te4\n"
        "\n"
        "                        Attribute constraints                         \n"
        "   band_gap: - min: 0.15\n"
        "             - max: 0.5\n"
        "\n"
        "----------------------------------------------------------------------"
    )


def test_mp_openapi_interface(structure_comparison):
    """Test the materials project interface."""
    ref_structures = read_yaml_file(REF_PATH + "mp_Cs-Te_openapi.yaml")
    strct_import = StructureImporter()
    strct_import.set_concentration_constraint("Cs", 0.3, 0.8)
    strct_import.set_concentration_constraint("Te", 0.3, 0.8)
    strct_collect = strct_import.import_from_mp(
        "Cs-Te",
        os.environ["MP_OPENAPI_KEY"],
    )
    assert strct_import._import_details == {"mp_openapi": [4, ["Cs", "Te"]]}
    for structure in strct_collect:
        ref_structure = None
        for ref_strct in ref_structures:
            if ref_strct["source_id"] == structure["attributes"]["source_id"]:
                ref_structure = ref_strct
                break
        if ref_structure is None:
            raise ValueError(f"Structure {structure['attributes']['source_id']} not found.")
        structure_comparison(structure, ref_structure)
        for attr in ["source_id", "source", "space_group", "theoretical"]:
            assert structure["attributes"][attr] == ref_structure[attr], f"'{attr}' doesn't match."
        for attr in ["formation_energy", "stability", "band_gap"]:
            assert (
                abs(structure["attributes"][attr]["value"] - ref_structure[attr]["value"]) < 1e-5
            ), f"Values of '{attr}' don't match."
            assert (
                structure["attributes"][attr]["unit"] == ref_structure[attr]["unit"]
            ), f"Units of '{attr}' don't match."


@pytest.mark.parametrize(
    "mp_id,property_data,import_details",
    [
        (
            "mp-573763",
            ["el_band_structure", "el_dos", "ph_band_structure", "ph_dos"],
            {"mp_openapi": [1, ["Cs", "Te"]]},
        ),
        ("mp-35925", ["xas_spectra"], {"mp_openapi": [1, ["Ni", "O"]]}),
    ],
)
def test_append_from_mp_by_id(
    structure_comparison, nested_dict_comparison, mp_id, property_data, import_details
):
    """Test mp interface for append_from_mp_by_id function."""
    ref_structure = read_yaml_file(REF_PATH + f"mp_{mp_id}.yaml")

    strct_import = StructureImporter()
    structure = strct_import.append_from_mp_by_id(
        mp_id, os.environ["MP_OPENAPI_KEY"], property_data=property_data
    )
    assert strct_import._import_details == import_details
    structure_comparison(structure, ref_structure)
    for attr in ["source_id", "source", "space_group", "theoretical"]:
        assert structure["attributes"][attr] == ref_structure[attr], f"'{attr}' doesn't match."
    for attr in ["formation_energy", "stability", "band_gap"]:
        assert (
            abs(structure["attributes"][attr]["value"] - ref_structure[attr]["value"]) < 1e-5
        ), f"Values of '{attr}' don't match."
        assert (
            structure["attributes"][attr]["unit"] == ref_structure[attr]["unit"]
        ), f"Units of '{attr}' don't match."
    nested_dict_comparison(structure["extras"], ref_structure["extras"])


def test_oqmd_interface(structure_comparison):
    """Test the open quantum materials database interface."""
    ref_structures = read_yaml_file(REF_PATH + "oqmd_Cs2Te.yaml")
    strct_import = StructureImporter()
    strct_collect = strct_import.import_from_oqmd("Cs2Te")
    assert strct_import._import_details == {"oqmd": [10, ["Cs", "Te"]]}
    for structure in strct_collect:
        ref_structure = None
        for ref_strct in ref_structures:
            if ref_strct["source_id"] == structure["attributes"]["source_id"]:
                ref_structure = ref_strct
                break
        if ref_structure is None:
            raise ValueError(f"Structure {structure['attributes']['source_id']} not found.")
        structure_comparison(structure, ref_structure)
        for attr in ["source_id", "source", "icsd_ids", "space_group", "functional"]:
            assert structure["attributes"][attr] == ref_structure[attr], f"'{attr}' doesn't match."
        for attr in ["formation_energy", "stability", "band_gap"]:
            assert (
                abs(structure["attributes"][attr]["value"] - ref_structure[attr]["value"]) < 1e-5
            ), f"Values of '{attr}' don't match."
            assert (
                structure["attributes"][attr]["unit"] == ref_structure[attr]["unit"]
            ), f"Units of '{attr}' don't match."


def test_optimade_interface(structure_comparison):
    """Test the optimade interface."""
    ref_structures = read_yaml_file(REF_PATH + "optimade_SiO2.yaml")
    strct_import = StructureImporter()
    strct_collect = strct_import.import_from_optimade("SiO2", "mcloud.mc3d-pbe-v1")
    assert strct_import._import_details == {"optimade-mcloud.mc3d-pbe-v1": [20, ["O", "Si"]]}
    for structure in strct_collect:
        ref_structure = None
        for ref_strct in ref_structures:
            if ref_strct["attributes"]["source_id"] == structure["attributes"]["source_id"]:
                ref_structure = ref_strct
                break
        if ref_structure is None:
            raise ValueError(f"Structure {structure['attributes']['source_id']} not found.")
        structure_comparison(structure, ref_structure)
        for attr in ["source_id", "source"]:
            assert (
                structure["attributes"][attr] == ref_structure["attributes"][attr]
            ), f"'{attr}' doesn't match."


@pytest.mark.parametrize("system", [("Cs-Te"), ("Cs-K-Sb")])
def test_pyxtal_interface_formula_series(system):
    """Test the creation of a formula series."""
    data = read_yaml_file(REF_PATH + system + "_formula_series.yaml")
    parameters = data["parameters"]
    reference = data["reference"]
    strct_import = StructureImporter()
    strct_import.neglect_elemental_structures = parameters.pop("neglect_elemental_structures")
    formula_series = strct_import._create_formula_series(**parameters)

    for (conc, formula), (ref_conc, ref_formula) in zip(formula_series, reference):
        assert conc == tuple(ref_conc) and formula == ref_formula


def test_pyxtal_interface_generate_crystals():
    """Test the import of structures in a concentration range."""
    args = {
        "formulas": "Cs-Sb",
        "dimensions": 3,
        "volume_factor": 1.0,
        "molecular": False,
        "tol_tuples": None,
        "max_structures": 25,
        "max_structures_per_cs": 10,
        "max_structures_per_sg": 20,
        "excl_space_groups": list(range(0, 17)) + list(range(220, 231)),
        "bin_size": 0.2,
    }
    elements = ["Cs", "Sb"]
    c_range = [0.2, 1.0]
    strct_import = StructureImporter()
    strct_import.neglect_elemental_structures = True
    strct_import.set_concentration_constraint("Cs", c_range[0], c_range[1])
    strct_collect = strct_import.generate_random_crystals(**args)
    assert strct_import._import_details == {"PyXtaL": [25, ["Cs", "Sb"]]}

    # Check entry specifications:
    for structure in strct_collect:
        concentration = sum(1 for el in structure["elements"] if el == elements[0]) / len(
            structure["elements"]
        )
        assert structure["attributes"]["space_group"] not in args["excl_space_groups"]
        assert c_range[0] <= concentration
        assert concentration <= c_range[1]


@pytest.mark.parametrize("system", ["tobmof-11", "coremof2019"])
def test_mofxdb_interface(nested_dict_comparison, structure_comparison, system):
    """Test the MOFXDB interface."""
    data = read_yaml_file(REF_PATH + "mofxdb_" + system + ".yaml")
    strct_import = StructureImporter()
    structures = strct_import.import_from_mofxdb(**data["parameters"])
    assert len(structures) == len(data["structures"]), "Number of queried MOFs is wrong."
    for strct, ref_strct in zip(structures, data["structures"]):
        structure_comparison(strct, ref_strct, compare_site_attrs=False)
        nested_dict_comparison(strct.attributes, ref_strct["attributes"])
        nested_dict_comparison(strct.extras, ref_strct["extras"])
