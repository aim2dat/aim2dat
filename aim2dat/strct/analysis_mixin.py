"""Anylsis mixin class for the Structure and StructureOperations classes."""

# Standard library imports
from __future__ import annotations
from typing import List, Tuple, Union
import abc
from collections.abc import Callable
from functools import wraps

# Internal library imports
from aim2dat.strct.analysis.symmetry import determine_point_group, determine_space_group
from aim2dat.strct.analysis.geometry import (
    calc_distance,
    calc_angle,
    calc_dihedral_angle,
)
from aim2dat.strct.analysis.voronoi_tessellation import calc_voronoi_tessellation
from aim2dat.strct.analysis.coordination import calc_coordination
from aim2dat.strct.analysis.rdf import calc_ffingerprint


def analysis_method(func):
    """Mark internal structure analysis functions."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._is_analysis_method = True
    return wrapper


class classproperty:
    """Custom, temporary decorator to depreciate class properties."""

    def __init__(self, func):
        """Initiate class."""
        self.fget = func

    def __get__(self, instance, owner):
        """Get method."""
        from warnings import warn

        warn(
            "This function will be removed soon, please use the `list_*_methods` instead.",
            DeprecationWarning,
            2,
        )
        return self.fget(owner)


class AnalysisMixin:
    """Mixin class to perform structural analysis tasks."""

    @classmethod
    def list_analysis_methods(cls) -> list:
        """
        Get a list with the function names of all available analysis methods.

        Returns
        -------
        list:
            Return a list of all available analysis methods.
        """
        analysis_methods = []
        for name, method in AnalysisMixin.__dict__.items():
            if getattr(method, "_is_analysis_method", False):
                analysis_methods.append(name)
        return analysis_methods

    @classproperty
    def analysis_methods(cls) -> list:
        """
        list: Return calculation methods. This property is depreciated and will be removed soon.
        """
        return cls.list_analysis_methods()

    @analysis_method
    def calc_point_group(
        self,
        threshold_distance: float = 0.1,
        threshold_angle: float = 1.0,
        threshold_inertia: float = 0.1,
    ) -> dict:
        """
        Determine the point group of a molecule.

        Parameters
        ----------
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
        return self._perform_strct_analysis(
            determine_point_group, kwargs, mapping={"point_group": ("point_group",)}
        )

    @analysis_method
    def calc_space_group(
        self,
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
        return self._perform_strct_analysis(
            determine_space_group, kwargs, mapping={"space_group": ("space_group", "number")}
        )

    @analysis_method
    def calc_distance(
        self,
        site_index1: Union[int, List[int]] = 0,
        site_index2: Union[int, List[int]] = 1,
        backfold_positions: bool = True,
        use_supercell: bool = False,
        r_max: float = 7.5,
        return_pos: bool = False,
    ) -> Union[float, list]:
        """
        Calculate distance between two atoms.

        Parameters
        ----------
        site_index1 : int, list or None
            Index of the site.
        site_index2 : int, list or None
            Index of the site.
        backfold_positions : bool
            Whether to backfold the atomic sites and return the smallest distance.
        use_supercell : bool
            User supercell to calculate all distances between the two atomic sites up to the
            radius ``r_max``.
        r_max : float
            Cut-off value for the maximum distance between two atoms in angstrom.
        return_pos : bool
            Whether to return the positions. Useful if ``use_supercell`` is set to ``True`` or when
            trying to determine the closest periodic image.

        Returns
        -------
        float, dict or None
            Distance between the two atoms or a list of distances (if ``use_super_cell`` is
            set to ``True``). If one of the indices is a list, a dictionary with all index pairs
            as keys and distances as values is returned. If ``return_pos`` is set to ``True``, the
            positions are returned as well. In case ``use_super_cell`` is set to ``True`` and the
            distance between the two sites exceeds ``r_max``, ``None`` is returned.
        """
        kwargs = {
            "site_index1": site_index1,
            "site_index2": site_index2,
            "backfold_positions": backfold_positions,
            "use_supercell": use_supercell,
            "r_max": r_max,
            "return_pos": return_pos,
        }
        return self._perform_strct_analysis(calc_distance, kwargs)

    @analysis_method
    def calc_angle(
        self,
        site_index1: Union[int, List[int]] = 0,
        site_index2: Union[int, List[int]] = 1,
        site_index3: Union[int, List[int]] = 2,
        backfold_positions: bool = True,
    ) -> float:
        """
        Calculate angle between three atoms.

        Parameters
        ----------
        site_index1 : int, list or None
            Index of the site.
        site_index2 : int, list or None
            Index of the site.
        site_index3 : int, list or None
            Index of the site.
        backfold_positions : bool
            Whether to backfold the atomic sites and return the smallest distance.

        Returns
        -------
        float or dict
            Angle calculated via the vectors from atom 2 to atom 1 and atom 3. If one of the
            indices is a list, a dictionary with all index pairs as keys and angles as values is
            returned.
        """
        kwargs = {
            "site_index1": site_index1,
            "site_index2": site_index2,
            "site_index3": site_index3,
            "backfold_positions": backfold_positions,
        }
        return self._perform_strct_analysis(calc_angle, kwargs)

    @analysis_method
    def calc_dihedral_angle(
        self,
        site_index1: Union[int, List[int]] = 0,
        site_index2: Union[int, List[int]] = 1,
        site_index3: Union[int, List[int]] = 2,
        site_index4: Union[int, List[int]] = 3,
        backfold_positions: bool = True,
    ) -> float:
        """
        Calculate dihedral angle between four atoms.

        Parameters
        ----------
        site_index1 : int, list or None
            Index of the site.
        site_index2 : int, list or None
            Index of the site.
        site_index3 : int, list or None
            Index of the site.
        site_index4 : int, list or None
            Index of the site.
        backfold_positions : bool
            Whether to backfold the atomic sites and return the smallest distance.

        Returns
        --------
        float or dict
            Dihedral angle. If one of the indices is a list, a dictionary with all
            index pairs as keys and angles as values is returned.
        """
        kwargs = {
            "site_index1": site_index1,
            "site_index2": site_index2,
            "site_index3": site_index3,
            "site_index4": site_index4,
            "backfold_positions": backfold_positions,
        }
        return self._perform_strct_analysis(calc_dihedral_angle, kwargs)

    @analysis_method
    def calc_voronoi_tessellation(
        self,
        r_max: float = 10.0,
    ) -> List[List[dict]]:
        """
        Calculate voronoi polyhedron for each atomic site.

        Parameters
        ----------
        r_max : float (optional)
            Cut-off value for the maximum distance between two atoms in angstrom.

        Returns
        -------
        list
            List of voronoi details for each atomic site.
        """
        return self._perform_strct_analysis(calc_voronoi_tessellation, {"r_max": r_max})

    @analysis_method
    def calc_coordination(
        self,
        indices: Union[int, list, tuple] = None,
        r_max: float = 10.0,
        method: str = "atomic_radius",
        min_dist_delta: float = 0.1,
        n_nearest_neighbours: int = 5,
        radius_type: str = "chen_manz",
        atomic_radius_delta: float = 0.0,
        econ_tolerance: float = 0.5,
        econ_conv_threshold: float = 0.001,
        voronoi_weight_type: float = "rel_solid_angle",
        voronoi_weight_threshold: float = 0.5,
        okeeffe_weight_threshold: float = 0.5,
        get_statistics: bool = True,
    ) -> Union[dict, list]:
        """
        Calculate coordination environment of each atomic site.

        Parameters
        ----------
        indices : list (optional)
            Site indices to include in the analysis.
        r_max : float (optional)
            Cut-off value for the maximum distance between two atoms in angstrom.
        method : str (optional)
            Method used to calculate the coordination environment.
        min_dist_delta : float (optional)
            Tolerance parameter that defines the relative distance from the nearest neighbour atom
            for the ``'minimum_distance'`` method.
        n_nearest_neighbours : int (optional)
            Number of neighbours that are considered coordinated for the ``'n_neighbours'``
            method.
        radius_type : str (optional)
            Type of the atomic radius used for the ``'atomic_radius'`` method (``'covalent'`` is
            used as fallback in the radius for an element is not defined).
        atomic_radius_delta : float (optional)
            Tolerance relative to the sum of the atomic radii for the ``'atomic_radius'`` method.
            If set to ``0.0`` the maximum threshold is defined by the sum of the atomic radii,
            positive (negative) values increase (decrease) the threshold.
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
        get_statistics : bool (optional)
            If set to ``False`` only the ``'sites'`` list is returned without average/min/max
            values for the distances and coordination numbers.

        Returns
        -------
        dict
            Dictionary containing the coordination information of the structure.
        """
        kwargs = {
            "indices": indices,
            "r_max": r_max,
            "method": method,
            "min_dist_delta": min_dist_delta,
            "n_nearest_neighbours": n_nearest_neighbours,
            "radius_type": radius_type,
            "atomic_radius_delta": atomic_radius_delta,
            "econ_tolerance": econ_tolerance,
            "econ_conv_threshold": econ_conv_threshold,
            "voronoi_weight_type": voronoi_weight_type,
            "voronoi_weight_threshold": voronoi_weight_threshold,
            "okeeffe_weight_threshold": okeeffe_weight_threshold,
            "get_statistics": get_statistics,
        }
        return self._perform_strct_analysis(calc_coordination, kwargs)

    @analysis_method
    def calc_ffingerprint(
        self,
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
        return self._perform_strct_analysis(calc_ffingerprint, kwargs)

    def perform_analysis(self, method: Callable, kwargs: dict = None):
        """
        Perform structure analaysis using an external method.

        Parameters
        ----------
        method : function
            Analysis function.
        kwargs : dict
            Arguments to be passed to the function.

        Returns
        ------
        output
            Output of the analysis.
        """
        kwargs = {} if kwargs is None else kwargs
        if not getattr(method, "_is_analysis_method", False):
            raise TypeError("Function is not a structure analysis method.")
        return self._perform_strct_analysis(method, kwargs)

    @abc.abstractmethod
    def _perform_strct_analysis(self, method, kwargs, mapping=None):
        pass
