"""Test element properties utils module."""

# Third party library imports
import pytest

# Internal library imports
import aim2dat.elements as el_properties


@pytest.mark.parametrize(
    "element,radius_type,value",
    [
        ("Cs", "covalent", 2.44),
        ("C", "vdw", 1.77),
        ("Sc", "vdw", 2.58),
        ("Si", "chen_manz", 1.38),
        ("Ne", "chen_manz", None),
    ],
)
def test_get_atomic_radius(element, radius_type, value):
    """Test atomic radius function."""
    assert el_properties.get_atomic_radius(element, radius_type=radius_type) == value


def test_get_atomic_radius_invalid_input():
    """Test invalid input for get_atomic_radius."""
    with pytest.raises(ValueError) as error:
        el_properties.get_atomic_radius("Cs", radius_type="invalid")
    assert str(error.value) == "Radius type 'invalid' not supported."


@pytest.mark.parametrize("element,scale,value", [("Cs", "pauling", 0.79), ("C", "allen", 2.544)])
def test_get_electronegativity(element, scale, value):
    """Test electronegativity function."""
    assert el_properties.get_electronegativity(element, scale=scale) == value


def test_get_electronegativity_invalid_input():
    """Test invalid input for get_electronegativity."""
    with pytest.raises(ValueError) as error:
        el_properties.get_electronegativity("Cs", scale="invalid")
    assert str(error.value) == "Scale 'invalid' not supported."


@pytest.mark.parametrize(
    "test_input, test_result", [(6, 6), ("Oxygen", 8), ("Pb", 82), ("sulfur", 16), ("na", 11)]
)
def test_get_atomic_number(test_input, test_result):
    """Test the ``get_atomic_number``-function."""
    assert test_result == el_properties.get_atomic_number(test_input)


def test_get_atomic_number_invalid_input():
    """Test invalid input for get_atomtic_number."""
    with pytest.raises(TypeError) as error:
        el_properties.get_atomic_number([2.4])
    assert str(error.value) == "Element '[2.4]' needs to have the type str or int."
    with pytest.raises(ValueError) as error:
        el_properties.get_atomic_number("Wr")
    assert str(error.value) == "Element 'Wr' could not be found."


@pytest.mark.parametrize(
    "test_input, test_result",
    [("C", "C"), ("Oxygen", "O"), (82, "Pb"), ("sulfur", "S"), ("na", "Na"), (118, "Og")],
)
def test_get_element_symbol(test_input, test_result):
    """Test get_element_symbol-function."""
    assert test_result == el_properties.get_element_symbol(test_input)


@pytest.mark.parametrize("element,value", [("Cs", 132.90545196), ("C", 12.011)])
def test_get_atomic_mass(element, value):
    """Test invalid input for get_atomic_mass."""
    assert value == el_properties.get_atomic_mass(element)


@pytest.mark.parametrize("element,value", [("Cs", 1), ("C", 4), (100, 14)])
def test_get_val_electrons(element, value):
    """Test invalid input for get_atomic_mass."""
    assert value == el_properties.get_val_electrons(element)
