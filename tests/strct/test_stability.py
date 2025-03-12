"""Test stability module."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.io import read_yaml_file
from aim2dat.strct.stability import (
    _find_most_stable_elemental_phases,
    _calculate_stabilities,
)
from aim2dat.utils.units import UnitConverter


REF_PATH = os.path.dirname(__file__) + "/stabilities/"


class Entry:
    """Class to mimick Structure to test stability calculation."""

    def __init__(self, label, chem_formula, attributes):
        """initialize class."""
        self.label = label
        self.chem_formula = chem_formula
        self.attributes = attributes
        self.elements = []
        for el, val in chem_formula.items():
            self.elements += [el] * val

    def set_attribute(self, key, value):
        """Set attribute."""
        self.attributes[key] = value


@pytest.mark.parametrize("system", ["Cs-Te", "Cs-Te_excl_keys"])
def test_stability_calculation(system):
    """Test calculation of formation energy and stability."""
    system_info = dict(read_yaml_file(REF_PATH + system + "_system.yaml"))
    structures = [
        Entry(f"label_{idx}", entry.pop("chem_formula"), entry)
        for idx, entry in enumerate(system_info["entries"])
    ]
    _calculate_stabilities(structures, None, exclude_keys=system_info.get("excl_keys", []))

    for entry, ref_val in zip(system_info["entries"], system_info["ref_values"]):
        assert (
            abs(entry["formation_energy"] - ref_val["formation_energy"]) < 1e-5
        ), "Formation energy differs."
        assert abs(entry["stability"] - ref_val["stability"]) < 1e-5, "Stability differs."


@pytest.mark.parametrize("unit", [None, "eV", "Hartree", "Rydberg"])
def test_unit_conversion(unit):
    """Test the use of different units."""
    system_info = dict(read_yaml_file(REF_PATH + "different_units_system.yaml"))
    structures = [
        Entry(f"label_{idx}", entry.pop("chem_formula"), entry)
        for idx, entry in enumerate(system_info["entries"])
    ]
    _calculate_stabilities(structures, unit)
    if unit is None:
        unit = "eV"
    for entry, ref in zip(structures, system_info["ref_values"]):
        for attr_key in ("formation_energy", "stability"):
            ref_val = UnitConverter.convert_units(ref[attr_key], "eV", unit)
            assert (
                entry.attributes[attr_key]["unit"] == unit + "/atom"
            ), f"Wrong unit for {attr_key}."
            assert abs(entry.attributes[attr_key]["value"] - ref_val) < 1e-5, f"{attr_key} differ."


def test_missing_unit():
    """Test missing unit."""
    system_info = dict(read_yaml_file(REF_PATH + "missing_unit_system.yaml"))
    structures = [
        Entry(f"label_{idx}", entry.pop("chem_formula"), entry)
        for idx, entry in enumerate(system_info["entries"])
    ]
    _calculate_stabilities(structures, "eV")
    for entry, ref in zip(structures, system_info["ref_values"]):
        for attr_key in ("formation_energy", "stability"):
            assert entry.attributes[attr_key]["unit"] == "eV/atom", f"Wrong unit for {attr_key}."
            assert (
                abs(entry.attributes[attr_key]["value"] - ref[attr_key]) < 1e-5
            ), f"{attr_key} differ."


def test_exceptions():
    """Test missing information in entries."""
    system_info = dict(read_yaml_file(REF_PATH + "exceptions_system.yaml"))
    structures = [
        Entry(f"label_{idx}", entry.pop("chem_formula"), entry)
        for idx, entry in enumerate(system_info["entries"])
    ]
    elemental_phases, _ = _find_most_stable_elemental_phases(structures, None, [])
    for element, total_energy in system_info["elemental_phases"].items():
        assert (
            abs(elemental_phases[element] - total_energy) < 1e-5
        ), f"Elemental phase {element} wrong."
    _calculate_stabilities(structures)
    for entry, ref_values in zip(system_info["entries"], system_info["ref_values"]):
        if "formation_energy" in ref_values:
            diff_f_energy = abs(entry["formation_energy"] - ref_values["formation_energy"])
            assert diff_f_energy < 1e-5, "Formation energy differs."
        else:
            assert "formation_energy" not in entry
        if "stability" in ref_values:
            diff_stability = abs(entry["stability"] - ref_values["stability"])
            assert diff_stability < 1e-5, "Stability differs."
        else:
            assert "stability" not in entry


def test_elemental_phases_only():
    """Test system only consisting of elemental phases."""
    system_info = dict(read_yaml_file(REF_PATH + "elemental_system.yaml"))
    structures = [
        Entry(f"label_{idx}", entry.pop("chem_formula"), entry)
        for idx, entry in enumerate(system_info["entries"])
    ]
    _calculate_stabilities(structures)
    for entry, ref_val in zip(system_info["entries"], system_info["ref_values"]):
        if ref_val is None:
            assert "formation_energy" not in entry and "stability" not in entry
        else:
            assert (
                abs(entry["formation_energy"] - ref_val) < 1e-5
            ), "Formation energy for elemental phases wrong."
            assert (
                abs(entry["stability"] - ref_val) < 1e-5
            ), "Stability for elemental phases wrong."
