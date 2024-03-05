"""Mixin classes for the Structure and StructureOperations classes."""

# Standard library imports
from __future__ import annotations
from typing import List, Tuple, Union
import abc
from typing import TYPE_CHECKING

# Internal library imports
import aim2dat.utils.chem_formula as utils_cf
from aim2dat.strct.strct_point_groups import determine_point_group
from aim2dat.strct.strct_space_groups import determine_space_group
from aim2dat.strct.strct_misc import (
    calculate_distance,
    calculate_angle,
    calculate_dihedral_angle,
)
from aim2dat.strct.strct_coordination import calculate_coordination
from aim2dat.strct.strct_super_cell import calculate_voronoi_tessellation
from aim2dat.strct.strct_prdf import calculate_ffingerprint
from aim2dat.strct.strct_manipulation import (
    delete_atoms,
    scale_unit_cell,
    substitute_elements,
)


if TYPE_CHECKING:
    from aim2dat.strct import Structure
    from aim2dat.strct import StructureCollection


def analysis_method(func):
    """Mark function as calculation function."""

    def check_arguments(*args, **kwargs):
        if args[0].__class__.__name__ == "Structure" and len(args) > 1:
            args = [args[0], None] + list(args)[1:]
        return func(*args, **kwargs)

    check_arguments._is_analysis_method = True
    return check_arguments


def manipulates_structure(func):
    """Mark structure manipulating functions."""

    def check_arguments(*args, **kwargs):
        if args[0].__class__.__name__ == "Structure" and len(args) > 1:
            args = [args[0], None] + list(args)[1:]
        return func(*args, **kwargs)

    check_arguments._manipulates_structure = True
    return check_arguments


class AnalysisMixin:
    """Mixin class to perform structural analysis tasks."""

    @classmethod
    @property
    def analysis_methods(cls) -> list:
        """list: Return calculation methods."""
        analysis_methods = []
        for name, method in AnalysisMixin.__dict__.items():
            if getattr(method, "_is_analysis_method", False):
                analysis_methods.append(name)
        return analysis_methods

    @analysis_method
    def determine_point_group(
        self,
        key: Union[str, int, tuple, list] = None,
        threshold_distance: float = 0.1,
        threshold_angle: float = 1.0,
        threshold_inertia: float = 0.1,
    ) -> dict:
        """
        Determine the point group of a molecule.

        Parameters
        ----------
        key : str, int, tuple or list
            Only used in the ``StructureOperations`` class. Specifies the key or list/tuple of
            keys of the underlying ``StructureCollection`` object.
        threshold_distance : float (optional)
            Tolerance parameter for distances.
        threshold_angle : float (optional)
            Tolerance parameter for angles.
        threshold_inertia : float (optional)
            Tolerance parameter for inertia.

        Returns
        -------
        dict
            Dictionary containing the point group and symmetry elements of the structure.
        """
        kwargs = {
            "threshold_distance": threshold_distance,
            "threshold_angle": threshold_angle,
            "threshold_inertia": threshold_inertia,
        }
        return self._perform_strct_analysis(key, determine_point_group, kwargs)

    @analysis_method
    def determine_space_group(
        self,
        key: Union[str, int, tuple, list] = None,
        symprec: float = 0.005,
        angle_tolerance: float = -1.0,
        hall_number: int = 0,
        return_sym_operations: bool = False,
        return_primitive_structure: bool = False,
        return_standardized_structure: bool = False,
        no_idealize: bool = False,
    ) -> dict:
        """
        Determine the space group of the structure using spglib as backend.

        Parameters
        ----------
        key : str, int, tuple or list
            Only used in the ``StructureOperations`` class. Specifies the key or list/tuple of
            keys of the underlying ``StructureCollection`` object.
        symprec : float (optional)
            Tolerance parameter for spglib
        angle_tolerance : float (optional)
            Tolerance parameter for spglib.
        hall_number : int (optional)
            The argument to constrain the space-group-type search only for the Hall symbol
            corresponding to it.
        return_sym_operations : bool (optional)
            Additionally, return all symmetry elements.
        return_primitive_structure : bool (optional)
            Whether to return the primitive standardized structure.
        return_standardized_structure : bool (optional)
            Whether to the non-primitive standardized structure.
        no_idealize : bool (optional)
            Whether to idealize unit cell vectors and angles.

        Returns
        -------
        dict
            Dictionary containing the internal space group number and labels.
        """
        kwargs = {
            "symprec": symprec,
            "angle_tolerance": angle_tolerance,
            "hall_number": hall_number,
            "return_sym_operations": return_sym_operations,
            "return_primitive_structure": return_primitive_structure,
            "return_standardized_structure": return_standardized_structure,
            "no_idealize": no_idealize,
        }
        return self._perform_strct_analysis(key, determine_space_group, kwargs)

    @analysis_method
    def calculate_distance(
        self,
        key: Union[str, int, tuple, list] = None,
        site_index1: Union[int, List[int]] = 0,
        site_index2: Union[int, List[int]] = 1,
        backfold_positions: bool = True,
        use_supercell: bool = False,
        r_max: float = 7.5,
    ) -> Union[float, list]:
        """
        Calculate distance between two atoms.

        Parameters
        ----------
        key : str, int, tuple or list
            Only used in the ``StructureOperations`` class. Specifies the key or list/tuple of
            keys of the underlying ``StructureCollection`` object.
        site_index1 : int
            Index of the site.
        site_index2 : int
            Index of the site.
        backfold_positions : bool
            Whether to backfold the atomic sites and return the smallest distance.
        use_supercell : bool
            User supercell to calculate all distances between the two atomic sites up to the
            radius ``r_max``.
        r_max : float
            Cut-off value for the maximum distance between two atoms in angstrom.

        Returns
        -------
        float
            Distance between the two atoms or a list of distances (if ``use_super_cell`` is
            set to ``True``).
        """
        kwargs = {
            "site_index1": site_index1,
            "site_index2": site_index2,
            "backfold_positions": backfold_positions,
            "use_supercell": use_supercell,
            "r_max": r_max,
        }
        return self._perform_strct_analysis(key, calculate_distance, kwargs)

    @analysis_method
    def calculate_angle(
        self,
        key: Union[str, int, tuple, list] = None,
        site_index1: int = 0,
        site_index2: int = 1,
        site_index3: int = 2,
        backfold_positions: bool = True,
    ) -> float:
        """
        Calculate angle between three atoms.

        Parameters
        ----------
        key : str, int, tuple or list
            Only used in the ``StructureOperations`` class. Specifies the key or list/tuple of
            keys of the underlying ``StructureCollection`` object.
        site_index1 : int
            Index of the site.
        site_index2 : int
            Index of the site.
        site_index3 : int
            Index of the site.
        backfold_positions : bool
            Whether to backfold the atomic sites and return the smallest distance.

        Returns
        -------
        float
            Angle calculated via the vectors from atom 2 to atom 1 and atom 3.
        """
        kwargs = {
            "site_index1": site_index1,
            "site_index2": site_index2,
            "site_index3": site_index3,
            "backfold_positions": backfold_positions,
        }
        return self._perform_strct_analysis(key, calculate_angle, kwargs)

    @analysis_method
    def calculate_dihedral_angle(
        self,
        key: Union[str, int, tuple, list] = None,
        site_index1: int = 0,
        site_index2: int = 1,
        site_index3: int = 2,
        site_index4: int = 3,
        backfold_positions: bool = True,
    ) -> float:
        """
        Calculate dihedral angle between four atoms.

        Parameters
        ----------
        key : str, int, tuple or list
            Only used in the ``StructureOperations`` class. Specifies the key or list/tuple of
            keys of the underlying ``StructureCollection`` object.
        site_index1 : int
            Index of the site.
        site_index2 : int
            Index of the site.
        site_index3 : int
            Index of the site.
        site_index4 : int
            Index of the site.
        backfold_positions : bool
            Whether to backfold the atomic sites and return the smallest distance.

        Returns
        --------
        float
            Dihedral angle.
        """
        kwargs = {
            "site_index1": site_index1,
            "site_index2": site_index2,
            "site_index3": site_index3,
            "site_index4": site_index4,
            "backfold_positions": backfold_positions,
        }
        return self._perform_strct_analysis(key, calculate_dihedral_angle, kwargs)

    @analysis_method
    def calculate_voronoi_tessellation(
        self, key: Union[str, int, tuple, list] = None, r_max: float = 10.0
    ) -> List[List[dict]]:
        """
        Calculate voronoi polyhedron for each atomic site.

        Parameters
        ----------
        key : str, int, tuple or list
            Only used in the ``StructureOperations`` class. Specifies the key or list/tuple of
            keys of the underlying ``StructureCollection`` object.
        r_max : float (optional)
            Cut-off value for the maximum distance between two atoms in angstrom.

        Returns
        -------
        list
            List of voronoi details for each atomic site.
        """
        return self._perform_strct_analysis(key, calculate_voronoi_tessellation, {"r_max": r_max})

    @analysis_method
    def calculate_coordination(
        self,
        key: Union[str, int, tuple, list] = None,
        r_max: float = 10.0,
        method: str = "minimum_distance",
        min_dist_delta: float = 0.1,
        n_nearest_neighbours: int = 5,
        econ_tolerance: float = 0.5,
        econ_conv_threshold: float = 0.001,
        voronoi_weight_type: float = "rel_solid_angle",
        voronoi_weight_threshold: float = 0.5,
        okeeffe_weight_threshold: float = 0.5,
    ) -> dict:
        """
        Calculate coordination environment of each atomic site.

        Parameters
        ----------
        key : str, int, tuple or list
            Only used in the ``StructureOperations`` class. Specifies the key or list/tuple of
            keys of the underlying ``StructureCollection`` object.
        r_max : float (optional)
            Cut-off value for the maximum distance between two atoms in angstrom.
        method : str (optional)
            Method used to calculate the coordination environment. The default value is
            ``'minimum_distance'``.
        min_dist_delta : float (optional)
            Tolerance parameter that defines the relative distance from the nearest neighbour atom
            for the ``'minimum_distance'`` method.
        n_nearest_neighbours : int (optional)
            Number of neighbours that are considered coordinated for the ``'n_neighbours'``
            method.
        econ_tolerance : float (optional)
            Tolerance parameter for the econ method.
        econ_conv_threshold : float (optional)
            Convergence threshold for the econ method.
        voronoi_weight_type : str (optional)
            Weight type of the Voronoi facets. Supported options are ``'covalent_atomic_radius'``,
            ``'area'`` and ``'solid_angle'``. The prefix ``'rel_'`` specifies that the relative
            weights with respect to the maximum value of the polyhedron are calculated.
        voronoi_weight_threshold : float (optional)
            Weight threshold to consider a neighbouring atom coordinated.
        okeeffe_weight_threshold : float (optional)
            Threshold parameter to distinguish indirect and direct neighbour atoms for the
            ``'okeeffe'``.

            This parameter is depreciated and will be removed in a future version.
            The original results can be obtained by using the ``voronoi_weight_threshold``
            parameter and setting ``voronoi_weight_type`` to ``'rel_solid_angle'``.

        Returns
        -------
        dict
            Dictionary containing the coordination information of the structure.
        """
        kwargs = {
            "r_max": r_max,
            "method": method,
            "min_dist_delta": min_dist_delta,
            "n_nearest_neighbours": n_nearest_neighbours,
            "econ_tolerance": econ_tolerance,
            "econ_conv_threshold": econ_conv_threshold,
            "voronoi_weight_type": voronoi_weight_type,
            "voronoi_weight_threshold": voronoi_weight_threshold,
            "okeeffe_weight_threshold": okeeffe_weight_threshold,
        }
        return self._perform_strct_analysis(key, calculate_coordination, kwargs)

    @analysis_method
    def calculate_ffingerprint(
        self,
        key: Union[str, int, tuple, list] = None,
        r_max: float = 20.0,
        delta_bin: float = 0.005,
        sigma: float = 0.05,
        use_legacy_smearing: bool = False,
        distinguish_kinds: bool = False,
    ) -> Tuple[dict, dict]:
        """
        Calculate f-fingerprint function for each element-pair and atomic site.

        The calculation is based on equation (3) in :doi:`10.1063/1.3079326`.

        Parameters
        ----------
        key : str, int, tuple or list
            Only used in the ``StructureOperations`` class. Specifies the key or list/tuple of
            keys of the underlying ``StructureCollection`` object.
        r_max : float (optional)
            Cut-off value for the maximum distance between two atoms in angstrom.
        delta_bin : float (optional)
            Bin size to descritize the function in angstrom.
        sigma : float (optional)
            Smearing parameter for the Gaussian function.
        use_legacy_smearing : bool
            Use the depreciated smearing method.
        distinguish_kinds: bool (optional)
            Whether different kinds should be distinguished e.g. Ni0 and Ni1 would be considered as
            different elements if ``True``.

        Returns
        -------
        element_fingerprints : dict
            Dictionary containing all fingerprint functions of the structure summed over all atoms
            of the same element.
        atomic_fingerprints : dict
            Dictionary containing all individual fingerprint functions for each atomic site.
        """
        kwargs = {
            "r_max": r_max,
            "delta_bin": delta_bin,
            "sigma": sigma,
            "distinguish_kinds": distinguish_kinds,
            "use_legacy_smearing": use_legacy_smearing,
        }
        return self._perform_strct_analysis(key, calculate_ffingerprint, kwargs)

    @abc.abstractmethod
    def _perform_strct_analysis(self, key, method, kwargs):
        pass


class ManipulationMixin:
    """Mixin class to perform structural manipulation tasks."""

    @classmethod
    @property
    def manipulation_methods(cls) -> list:
        """list: Return manipulation methods."""
        manipulation_methods = []
        for name, method in ManipulationMixin.__dict__.items():
            if getattr(method, "_manipulates_structure", False):
                manipulation_methods.append(name)
        return manipulation_methods

    @manipulates_structure
    def delete_atoms(
        self,
        key: Union[str, int, tuple, list] = None,
        elements: Union[str, List[str]] = [],
        site_indices: Union[int, List[int]] = [],
        change_label: bool = False,
    ) -> Union["Structure", "StructureCollection"]:
        """
        Delete atoms by element, list of elements, site index  or list of site indices.

        Parameters
        ----------
        key : str, int, tuple or list
            Only used in the ``StructureOperations`` class. Specifies the key or list/tuple of
            keys of the underlying ``StructureCollection`` object.
        elements :  str, list or tuple
            Element or tuple or list of  the elements to be deleted.
        site_indices : list or tuple
            Site index or tuple or list of site indices to be deleted.

        Returns
        -------
        aim2dat.strct.Structure
            Structure with deleted atoms.
        """
        kwargs = {
            "elements": elements,
            "site_indices": site_indices,
            "change_label": change_label,
        }
        return self._perform_strct_manipulation(key, delete_atoms, kwargs)

    @manipulates_structure
    def scale_unit_cell(
        self,
        key: Union[str, int, tuple, list] = None,
        scaling_factor: float = 1.0,
        change_label: bool = False,
    ) -> Union["Structure", "StructureCollection"]:
        """
        Scale unit cell of the structure.

        Parameters
        ----------
        key : str, int, tuple or list
            Only used in the ``StructureOperations`` class. Specifies the key or list/tuple of
            keys of the underlying ``StructureCollection`` object.
        scaling_factor : float
            Scaling factor.

        Returns
        -------
        aim2dat.strct.Structure
            Structure with scaled unit cell.
        """
        kwargs = {"scaling_factor": scaling_factor, "change_label": change_label}
        return self._perform_strct_manipulation(key, scale_unit_cell, kwargs)

    @manipulates_structure
    def substitute_elements(
        self,
        key: Union[str, int, tuple, list] = None,
        elements: Union[List[Tuple[str]], List[Tuple[int]]] = [],
        radius_type: Union[str, None] = "covalent",
        change_label: bool = False,
    ) -> Union["Structure", "StructureCollection"]:
        """
        Substitute all atoms of one or several elements.

        Parameters
        ----------
        key : str, int, tuple or list
            Only used in the ``StructureOperations`` class. Specifies the key or list/tuple of
            keys of the underlying ``StructureCollection`` object.
        elements : list or tuple
            Tuple or list of tuples of the elements that are substituted.
        radius_type : str or None (optional)
            Radius type used to calculate the scaling factor for the unit cell. If set to ``None``
            no scaling is applied. The default value is ``covalent``.

        Returns
        -------
        aim2dat.strct.Structure
            Structure with substituted elements.
        """
        kwargs = {
            "elements": elements,
            "radius_type": radius_type,
            "change_label": change_label,
        }
        return self._perform_strct_manipulation(key, substitute_elements, kwargs)

    @abc.abstractmethod
    def _perform_strct_manipulation(self, key, method_name, kwargs):
        pass


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
            formula_dict = utils_cf.transform_str_to_dict(formula)
            unspecified_quantity = "-"
            if any(quantity == unspecified_quantity for quantity in formula_dict.values()):
                formula_add["element_set"] = []
                for element in formula_dict.keys():
                    formula_add["element_set"].append(element)
            else:
                if reduced_formula:
                    formula_dict = utils_cf.reduce_formula(formula_dict)
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
            constr_formula_str = utils_cf.transform_dict_to_str(constr["formula"])
            chem_f = structure["chem_formula"]
            if constr["is_reduced"]:
                chem_f = utils_cf.reduce_formula(chem_f)
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
                formula_str = utils_cf.transform_dict_to_str(structure["chem_formula"])
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
                formula_str = utils_cf.transform_dict_to_str(chem_formula)
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
