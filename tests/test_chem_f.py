"""Test chem formula utils module."""

# Third party library imports
import pytest

# Internal library imports
import aim2dat.chem_f as chem_f


@pytest.mark.parametrize(
    "formula_dict, formula_str, formula_lat_str, formula_list, formula_red",
    [
        (
            {"H": 2.0, "O": 1.0},
            "H2O",
            r"$\mathrm{H}_\mathrm{2}\mathrm{O}$",
            ["O", "H", "H"],
            {"H": 2, "O": 1},
        ),
        ({"Fe": 30}, "Fe30", r"$\mathrm{Fe}_\mathrm{30}$", ["Fe"] * 30, {"Fe": 1}),
        (
            {"Li": 2, "Fe": 4, "O": 2},
            "Li2Fe4O2",
            r"$\mathrm{Li}_\mathrm{2}\mathrm{Fe}_\mathrm{4}\mathrm{O}_\mathrm{2}$",
            ["Li"] * 2 + ["Fe"] * 4 + ["O"] * 2,
            {"Li": 1, "Fe": 2, "O": 1},
        ),
    ],
)
def test_transformations(formula_dict, formula_str, formula_lat_str, formula_list, formula_red):
    """Test transformation between different types."""
    assert formula_dict == chem_f.transform_str_to_dict(formula_str)
    assert formula_str == chem_f.transform_dict_to_str(formula_dict)
    assert formula_lat_str == chem_f.transform_dict_to_latexstr(formula_dict)
    assert formula_dict == chem_f.transform_list_to_dict(formula_list)
    assert formula_red == chem_f.reduce_formula(formula_dict)


@pytest.mark.parametrize(
    "formula_str, formula_dict",
    [
        ("HOH", {"H": 2.0, "O": 1.0}),
        ("H.5(CO)CH3{OH[CH]4}3.5", {"C": 16.0, "O": 4.5, "H": 21.0}),
        ("(OH)2CH34", {"O": 2.0, "H": 36.0, "C": 1.0}),
    ],
)
def test_str_transformations(formula_str, formula_dict):
    """Test specific formating of string representation."""
    assert formula_dict == chem_f.transform_str_to_dict(formula_str)


@pytest.mark.parametrize(
    "formula_dict, tol, formula_red",
    [
        ({"Li": 2, "C": 2.3, "O": 2.4}, 1.0e-3, {"Li": 20, "C": 23, "O": 24}),
        ({"Li": 0.3334, "C": 0.6667}, 1.0e-3, {"Li": 1, "C": 2}),
        ({"Li": 0.3334, "C": 0.6667}, 1.0e-4, {"Li": 3334, "C": 6667}),
        ({"Sc": 0.001}, 1.0e-4, {"Sc": 1}),
    ],
)
def test_reduction_w_float_nrs(formula_dict, tol, formula_red):
    """Test reduction of chemical formulas with float numbers."""
    assert formula_red == chem_f.reduce_formula(formula_dict, tolerance=tol)


@pytest.mark.parametrize(
    "formula1, formula2, reduced, reference",
    [
        ({"Li": 2, "C": 2.3, "O": 2.4}, {"Li": 1.0, "C": 1.15, "O": 1.2}, True, True),
        ({"Li": 2, "C": 2.3, "O": 2.4}, {"Li": 1.0, "C": 1.15, "O": 1.2}, False, False),
        ({"Li": 2, "C": 2.3, "O": 2.4}, {"Li": 1.0, "C": 1.15, "O": 1}, True, False),
        ({"H": 2, "O": 1.0}, {"H": 2, "O": 1.0}, False, True),
        ({"H": 2, "O": 1}, {"H": 3, "O": 1.5}, True, True),
    ],
)
def test_comparison(formula1, formula2, reduced, reference):
    """Test the comparison of two formulas."""
    assert reference == chem_f.compare_formulas(formula1, formula2, reduced)
