"""Manipulation Mixin class for the Structure and StructureOperations classes."""

# Standard library imports
from __future__ import annotations
from typing import List, Tuple, Union
import abc
from typing import TYPE_CHECKING
from collections.abc import Callable
from functools import wraps

# Internal library imports
from aim2dat.strct.manipulation.cell import scale_unit_cell, create_supercell
from aim2dat.strct.manipulation.sites import delete_atoms, substitute_elements

if TYPE_CHECKING:
    from aim2dat.strct import Structure
    from aim2dat.strct import StructureCollection


def manipulates_structure(func):
    """Mark structure manipulating functions."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._manipulates_structure = True
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


class ManipulationMixin:
    """Mixin class to perform structural manipulation tasks."""

    @classmethod
    def list_manipulation_methods(cls) -> list:
        """
        Get a list with the function names of all available manipulation methods.

        Returns
        -------
        list:
            Return a list of all available manipulation methods.
        """
        manipulation_methods = []
        for name, method in ManipulationMixin.__dict__.items():
            if getattr(method, "_manipulates_structure", False):
                manipulation_methods.append(name)
        return manipulation_methods

    @classproperty
    def manipulation_methods(cls) -> list:
        """
        list: Return manipulation methods. This property is depreciated and will be removed soon.
        """
        return cls.list_manipulation_methods()

    @manipulates_structure
    def delete_atoms(
        self,
        elements: Union[str, List[str]] = [],
        site_indices: Union[int, List[int]] = [],
        change_label: bool = False,
    ) -> Union["Structure", "StructureCollection"]:
        """
        Delete atoms by element, list of elements, site index  or list of site indices.

        Parameters
        ----------
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
        return self._perform_strct_manipulation(delete_atoms, kwargs)

    @manipulates_structure
    def scale_unit_cell(
        self,
        scaling_factors: Union[float, List[float]] = None,
        pressure: float = None,
        bulk_modulus: float = None,
        random_factors: float = None,
        random_seed: int = None,
        change_label: bool = False,
    ) -> Union["Structure", "StructureCollection"]:
        """
        Scale the unit cell of the structure, supporting isotropic and anisotropic strain,
        and pressure-based strain.

        Parameters
        ----------
        scaling_factors : float, list of floats, or arry-like, optional
            Scaling factor(s) to scale the unit cell.
            - If a single float, isotropic scaling is applied.
            - If a list of 3 floats or a 1D array, anisotropic scaling is
              applied along the principal axes.
            - If a 3x3 nested list or a 2D array, it defines a full
              transformation matrix.
            Scaling factors are interpreted as 1 + strain. For example:
            - A 1% strain corresponds to a scaling factor of 1.01.
            - A -2% strain (compression) corresponds to a scaling factor of 0.98.
        pressure : float, optional
            Hydrostatic pressure to apply. Requires `bulk_modulus` to calculate scaling.
        bulk_modulus : float, optional
            Bulk modulus of the material. Required if `pressure` is provided. Ensure the units
            of `bulk_modulus` and `pressure` are consistent.
        random_factors : float, optional
            Extend to which the unitcell will be randomly scaled/distorted.
        random_seed : int, optional
            Specify the random seed to ensure reproducible results.
        change_label : bool, optional
            If True, appends a suffix to the structure's label to reflect
            the scaling applied. Defaults to True

        Returns
        -------
        Structure or StructureCollection
            The scaled structure or a collection of scaled structures.

        Raises
        ------
        ValueError
            If required parameters are missing or invalid, such as when `pressure` is given
            without `bulk_modulus`, or invalid `scaling_factors` inputs.

        Notes
        -----
        - The `pressure` and `bulk_modulus` inputs are mutually exclusive with direct
        `scaling_factors` input.
        - Scaling factors directly modify the unit cell dimensions and are applied such that
          fractional atomic positions remain unchanged.
        """
        kwargs = {
            "scaling_factors": scaling_factors,
            "pressure": pressure,
            "bulk_modulus": bulk_modulus,
            "random_factors": random_factors,
            "random_seed": random_seed,
            "change_label": change_label,
        }
        return self._perform_strct_manipulation(scale_unit_cell, kwargs)

    @manipulates_structure
    def create_supercell(
        self, size: Union[int, list, tuple] = 2, wrap: bool = True, change_label: bool = False
    ):
        """
        Create supercell.

        Parameters
        ----------
        size : int, list, tuple
            Super cell size, given as a list/tuple of three positive integer values or one integer
            value, applied to all directions.
        wrap : bool
            Wrap atomic positions back into the unit cell.
        change_label : bool

        Returns
        -------
        Structure or StructureCollection
            Structure or a collection of structures.

        Raises
        ------
        TypeError
            If ``size`` has the wrong type.
        ValueError
            If ``size`` is not a positive integer number or a tuple/list of three positive integer
            numbers.
        Warning
            If ``size`` is gives a multiple of 1 for a non-periodic direction.
        """
        kwargs = {
            "size": size,
            "wrap": wrap,
            "change_label": change_label,
        }
        return self._perform_strct_manipulation(create_supercell, kwargs)

    @manipulates_structure
    def substitute_elements(
        self,
        elements: Union[List[Tuple[str]], List[Tuple[int]]] = [],
        radius_type: Union[str, None] = "covalent",
        remove_kind: bool = False,
        change_label: bool = False,
    ) -> Union["Structure", "StructureCollection"]:
        """
        Substitute all atoms of one or several elements.

        Parameters
        ----------
        elements : list or tuple
            Tuple or list of tuples of the elements that are substituted.
        remove_kind : bool (optional)
            Sets the entries of the substituted sites in `kinds` to `None`.
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
            "remove_kind": remove_kind,
            "change_label": change_label,
        }
        return self._perform_strct_manipulation(substitute_elements, kwargs)

    def perform_manipulation(self, method: Callable, kwargs: dict = None):
        """
        Perform structure manipulation using an external method.

        Parameters
        ----------
        method : function
            Function which manipulates the structure(s).
        kwargs : dict
            Arguments to be passed to the function.

        Returns
        ------
        aim2dat.strct.Structure or
        aim2dat.strct.StructureCollection
            Manipulated structure(s).
        """
        kwargs = {} if kwargs is None else kwargs
        if not getattr(method, "_manipulates_structure", False):
            raise TypeError("Function is not a structure analysis method.")
        return self._perform_strct_manipulation(method, kwargs)

    @abc.abstractmethod
    def _perform_strct_manipulation(self, method_name, kwargs):
        pass
