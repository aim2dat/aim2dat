"""
Functions to calculate the formation energy and stability.
"""

# Internal library imports
from aim2dat.chem_f import transform_dict_to_str
from aim2dat.fct.hull import get_convex_hull
from aim2dat.units import UnitConverter


def _extract_value(structure, attr, unit):
    value = structure.attributes[attr]
    unit0 = None
    if isinstance(value, dict):
        unit0 = value["unit"].split("/")[0]
        value = value["value"]
    if unit0 is None and unit is not None:
        chem_f_str = transform_dict_to_str(structure.chem_formula)
        print(
            f"No unit given for {structure.label} - {chem_f_str} with attr. '{attr}', ",
            f"assuming {unit} as energy unit.",
        )
        unit0 = unit
    elif unit is None and unit0 is not None:
        print(f"Unit {unit0} is used as energy unit.")
        unit = unit0
    if unit0 is None and unit is None:
        return value, unit
    else:
        return UnitConverter.convert_units(value, unit0, unit), unit


def _set_structure_attr(structure, key, value, unit):
    if unit is None:
        structure.set_attribute(key, value)
    else:
        structure.set_attribute(key, {"value": value, "unit": unit + "/atom"})


def _find_most_stable_elemental_phases(structures, output_unit, excl_indices):
    """
    Find most stable elemental phases of a list of entries based on the total energy per atom.
    """
    unit = output_unit
    elemental_phases = {}
    for idx, structure in enumerate(structures):
        if idx in excl_indices:
            continue
        if len(set(structure.elements)) == 1:
            el_symbol = structure.elements[0]
            energy_p_atom = None
            if "total_energy_per_atom" in structure.attributes:
                energy_p_atom, unit = _extract_value(structure, "total_energy_per_atom", unit)
            elif "total_energy" in structure.attributes:
                total_energy, unit = _extract_value(structure, "total_energy", unit)
                energy_p_atom = total_energy / len(structure.elements)
            if energy_p_atom is not None:
                if elemental_phases.get(el_symbol) is None:
                    elemental_phases[el_symbol] = energy_p_atom
                elif elemental_phases[el_symbol] > energy_p_atom:
                    elemental_phases[el_symbol] = energy_p_atom
    return elemental_phases, unit


def _calculate_formation_energy(structure, elemental_phases, unit):
    """
    Calculate formation energies of a list of entries based on the total energy an the total
    energy per atom.
    """
    chem_formula = structure.chem_formula
    if any(elemental_phases.get(element) is None for element in chem_formula.keys()):
        print(
            f"Elemental phase missing, cannot process entry {structure.label} - "
            f"{transform_dict_to_str(chem_formula)}."
        )
        return None, unit
    if "total_energy" in structure.attributes:
        form_energy, unit = _extract_value(structure, "total_energy", unit)
    elif "total_energy_per_atom" in structure.attributes:
        total_energy_per_atom, unit = _extract_value(structure, "total_energy_per_atom", unit)
        form_energy = total_energy_per_atom * sum(chem_formula.values())
    else:
        print(
            f"Total energy missing, cannot process entry {structure.label} - "
            f"{transform_dict_to_str(chem_formula)}."
        )
        return None, unit
    for element, quantity in chem_formula.items():
        form_energy -= elemental_phases[element] * quantity
    form_energy /= sum(chem_formula.values())

    _set_structure_attr(structure, "formation_energy", form_energy, unit)
    return form_energy, unit


def _calculate_stabilities(structures, output_unit=None, exclude_keys=[]):
    """
    Calculate the stability of a list of entries.
    """
    excl_indices = [structures.index(key) if isinstance(key, str) else key for key in exclude_keys]
    elemental_phases, unit = _find_most_stable_elemental_phases(
        structures, output_unit, excl_indices
    )
    if len(elemental_phases) == 0:
        print("No elemental reference phases found.")
        return [], []

    formation_energies = []
    stabilities = []
    conc_el = list(elemental_phases.keys())[0]
    ch_points = []
    concentrations = []
    for idx, structure in enumerate(structures):
        form_energy, unit = _calculate_formation_energy(structure, elemental_phases, unit)
        conc = 0.0
        if conc_el in structure.chem_formula:
            conc = structure.chem_formula[conc_el] / sum(structure.chem_formula.values())
        if form_energy is not None and idx not in excl_indices:
            ch_points.append((conc, form_energy))
        concentrations.append(conc)
        formation_energies.append(form_energy)

    convex_hull = [[0]]
    if len(elemental_phases) > 1:
        convex_hull = get_convex_hull(ch_points, upper_hull=False)

    for structure, conc, form_energy in zip(structures, concentrations, formation_energies):
        if form_energy is None:
            stabilities.append(None)
        else:
            ch_entry = 0.0
            # TODO this part only works for binary systems.
            for ch_idx in range(len(convex_hull[0]) - 1):
                if convex_hull[0][ch_idx] <= conc <= convex_hull[0][ch_idx + 1]:
                    x_val1 = convex_hull[0][ch_idx]
                    x_val2 = convex_hull[0][ch_idx + 1]
                    y_val1 = convex_hull[1][ch_idx]
                    y_val2 = convex_hull[1][ch_idx + 1]
                    ch_entry = (conc - x_val1) * (y_val2 - y_val1) / (x_val2 - x_val1) + y_val1
                    break
            _set_structure_attr(structure, "stability", form_energy - ch_entry, unit)
            stabilities.append(form_energy - ch_entry)
    return formation_energies, stabilities
