"""Constraint mixin class for the StructureImporter class."""

# Standard library imports
from __future__ import annotations

# Internal library imports
from aim2dat.chem_f import transform_str_to_dict, transform_dict_to_str, reduce_formula


class ConstraintError(Exception):
    """Constraint error."""

    pass


class ConstraintsMixin:
    """Mixin to implement structural constraints."""

    def remove_constraints(self):
        """
        Remove all constraints.
        """
        self._formula_constraints = []
        self._conc_constraints = {}
        self._neglect_el_structures = False
        self._attr_constraints = {}

    @property
    def chem_formula_constraints(self):
        """
        Constraints on the chemical formula.
        """
        constraints = []
        if hasattr(self, "_formula_constraints"):
            constraints = self._formula_constraints
        return constraints

    @property
    def concentration_constraints(self):
        """
        Elemental concentration constraints.
        """
        constraints = {}
        if hasattr(self, "_conc_constraints"):
            constraints = self._conc_constraints
        return constraints

    @property
    def attribute_constraints(self):
        """
        Attribute constraints.
        """
        constraints = {}
        if hasattr(self, "_attr_constraints"):
            constraints = self._attr_constraints
        return constraints

    @property
    def neglect_elemental_structures(self):
        """
        Whether to neglect elemental phases.
        """
        neg_el_structures = False
        if hasattr(self, "_neglect_el_structures"):
            neg_el_structures = self._neglect_el_structures
        return neg_el_structures

    @neglect_elemental_structures.setter
    def neglect_elemental_structures(self, value):
        self._neglect_el_structures = value

    def add_chem_formula_constraint(self, chem_formula, reduced_formula=True):
        """
        Add a chemical formula as a constraint.

        The formula can be given as a string, dictionary or list of strings or dictionaries.

        Parameters
        ----------
        chem_formula : list, dict or str
            Chemical formula given as list, dict or str.
        reduced_formula : bool (optional)
            If set to ``True`` the reduced formulas are compared. The default value is ``True``.
        """
        if isinstance(chem_formula, (str, dict)):
            chem_formula = [chem_formula]
        if not isinstance(chem_formula, list):
            raise TypeError("`chem_formula` needs to be string, dict or list.")

        for formula in chem_formula:
            formula_add = {"is_reduced": reduced_formula}
            formula_dict = transform_str_to_dict(formula)
            unspecified_quantity = "-"
            if any(quantity == unspecified_quantity for quantity in formula_dict.values()):
                formula_add["element_set"] = []
                for element in formula_dict.keys():
                    formula_add["element_set"].append(element)
            else:
                if reduced_formula:
                    formula_dict = reduce_formula(formula_dict)
                formula_add["formula"] = formula_dict
            if not hasattr(self, "_formula_constraints"):
                self._formula_constraints = []
            if formula_add not in self._formula_constraints:
                self._formula_constraints.append(formula_add)

    def set_concentration_constraint(self, element, min_conc=0.0, max_conc=1.0):
        """
        Set a constraint on the concentration of an element in the structure.

        The minimum and maximum values have to be set between 0.0 and 1.0.

        Parameters
        ----------
        element : str
            Element to be constraint.
        min_conc : float
            Minimum concentration. In case of no limit the variable can be set to ``0.0``.
        max_conc : float
            Maximum concentration. In case of no limit the variable can be set to ``1.0``.
        """
        for conc in [min_conc, max_conc]:
            if conc < 0.0 or conc > 1.0:
                raise ValueError("`min_conc` and `max_conc` need to be inbetween 0.0 and 1.0.")
        if max_conc < min_conc:
            raise ValueError("`max_conc` needs to be larger than `min_conc`.")
        if not hasattr(self, "_conc_constraints"):
            self._conc_constraints = {}
        self._conc_constraints[element] = [min_conc, max_conc]

    def set_attribute_constraint(self, attribute, min_value=None, max_value=None):
        """
        Set a constraint on attributes.

        Parameters
        ----------
        attribute : str
            Attribute to be constraint.
        min_value : float
            Minimum value of the attribute. In case of no limit the variable can be set to ``0.0``.
        max_value : float
            Maximum value of the attribute. In case of no limit the variable can be set to ``1.0``.
        """
        if min_value is not None and max_value is not None and max_value < min_value:
            raise ValueError("`max_value` needs to be equal or larger than `min_value`.")
        if not hasattr(self, "_attr_constraints"):
            self._attr_constraints = {}
        self._attr_constraints[attribute] = [min_value, max_value]

    def _check_chem_formula_constraints(self, structure, print_message, raise_error):
        def _validate_formula_constraint(structure, constr, constr_formulas):
            const_fulfilled = True
            constr_formula_str = transform_dict_to_str(constr["formula"])
            chem_f = structure["chem_formula"]
            if constr["is_reduced"]:
                chem_f = reduce_formula(chem_f)
            constr_formulas.append(constr_formula_str)

            if len(chem_f) != len(constr["formula"]):
                const_fulfilled = False
            elif not all(el in chem_f for el in constr["formula"]):
                const_fulfilled = False
            elif not all(chem_f[el] == constr["formula"][el] for el in constr["formula"].keys()):
                const_fulfilled = False
            return const_fulfilled

        def _validate_el_set_constraint(structure, constr, constr_formulas):
            const_fulfilled = True
            constr_formulas.append("-".join(constr["element_set"]))
            for el in structure["chem_formula"].keys():
                if el not in constr["element_set"]:
                    const_fulfilled = False
                    break
            return const_fulfilled

        const_fulfilled = True
        if hasattr(self, "_formula_constraints") and self._formula_constraints is not None:
            constr_formulas = []
            for constr in self._formula_constraints:
                const_fulfilled = True
                if "formula" in constr:
                    if _validate_formula_constraint(structure, constr, constr_formulas):
                        break
                    else:
                        const_fulfilled = False
                else:
                    if _validate_el_set_constraint(structure, constr, constr_formulas):
                        break
                    else:
                        const_fulfilled = False
            if not const_fulfilled:
                formula_str = transform_dict_to_str(structure["chem_formula"])
                constr_reason = (
                    str(structure["label"])
                    + " - Chem. formula constraint: "
                    + formula_str
                    + " doesn't match with "
                    + ", ".join(constr_formulas)
                    + "."
                )
                if raise_error:
                    raise ConstraintError(constr_reason)
                elif print_message:
                    print(constr_reason)
        return const_fulfilled

    def _apply_constraint_checks(self, structure, raise_error):
        if not self._check_concentration_constraints(structure, True, raise_error):
            return False
        if not self._check_chem_formula_constraints(structure, True, raise_error):
            return False
        if not self._check_attribute_constraints(structure, True, raise_error):
            return False
        return True

    def _check_concentration_constraints(self, structure, print_message, raise_error):
        chem_formula = structure["chem_formula"]
        const_fulfilled = True
        if hasattr(self, "_neglect_el_structures") and self._neglect_el_structures:
            if len(chem_formula) == 1:
                formula_str = transform_dict_to_str(chem_formula)
                const_fulfilled = False
                constr_reason = (
                    str(structure["label"])
                    + " - Concentration constraint: Elemental structures neglected, "
                    + formula_str
                    + "."
                )
                if raise_error:
                    raise ConstraintError(constr_reason)
                elif print_message:
                    print(constr_reason)

        if (
            const_fulfilled
            and hasattr(self, "_conc_constraints")
            and self._conc_constraints is not None
        ):
            for element, constr in self._conc_constraints.items():
                if element in chem_formula:
                    conc = float(chem_formula[element]) / sum(chem_formula.values())
                    if conc <= constr[0]:
                        const_fulfilled = False
                        constr_reason = (
                            str(structure["label"])
                            + " - Concentration constraint: "
                            + str(round(conc, 5))
                            + " lower than "
                            + str(constr[0])
                            + " for "
                            + element
                            + "."
                        )
                        if raise_error:
                            raise ConstraintError(constr_reason)
                        else:
                            print(constr_reason)
                        break
                    elif conc >= constr[1]:
                        const_fulfilled = False
                        constr_reason = (
                            str(structure["label"])
                            + " - Concentration constraint: "
                            + str(round(conc, 5))
                            + " greater than "
                            + str(constr[1])
                            + " for "
                            + element
                            + "."
                        )
                        if raise_error:
                            raise ConstraintError(constr_reason)
                        else:
                            print(constr_reason)
                        break
        return const_fulfilled

    def _check_attribute_constraints(self, structure, print_message, raise_error):
        const_fulfilled = True
        if hasattr(self, "_attr_constraints") and self._attr_constraints is not None:
            for attr, constr in self._attr_constraints.items():
                if attr not in structure["attributes"]:
                    const_fulfilled = False
                    constr_reason = (
                        f"{structure['label']} - Attribute constraint: attribute {attr} not found."
                    )
                    break

                attr_value = structure["attributes"][attr]
                if isinstance(attr_value, dict):
                    attr_value = attr_value["value"]
                if constr[0] is not None and attr_value < constr[0]:
                    const_fulfilled = False
                    constr_reason = (
                        str(structure["label"])
                        + " - Attribute constraint: "
                        + str(round(attr_value, 5))
                        + " lower than "
                        + str(constr[0])
                        + " for "
                        + str(attr)
                        + "."
                    )
                    break
                elif constr[1] is not None and attr_value > constr[1]:
                    const_fulfilled = False
                    constr_reason = (
                        str(structure["label"])
                        + " - Attribute constraint: "
                        + str(round(attr_value, 5))
                        + " greater than "
                        + str(constr[1])
                        + " for "
                        + attr
                        + "."
                    )
                    break
            if not const_fulfilled:
                if raise_error:
                    raise ConstraintError(constr_reason)
                elif print_message:
                    print(constr_reason)

        return const_fulfilled
