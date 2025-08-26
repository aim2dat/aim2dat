"""Module implementing a class to generate surfaces."""

# Standard library imports
from typing import List, Tuple, Union

# Internal library imports
from aim2dat.strct.structure import Structure
from aim2dat.strct.structure_collection import StructureCollection
from aim2dat.ext_interfaces import _return_ext_interface_modules
from aim2dat.strct.surface_utils import _surface_create, _surface_create_slab

# from aim2dat.aiida_data.surface_data import SurfaceData


class SurfaceGeneration:
    """Generates a surfaces and surface slabs based on a bulk crystal structure."""

    def __init__(
        self,
        structure: Structure,
    ):
        """Initialize object."""
        self.structure = structure

    def __getitem__(self, key):
        """Return item by key."""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Set item by key."""
        setattr(self, key, value)

    def create_surface(
        self,
        miller_indices: Union[Tuple[int], List[int]] = (1, 0, 0),
        termination: int = 1,
        tolerance: float = 0.005,
        symprec: float = 0.005,
        angle_tolerance: float = -1.0,
        hall_number: int = 0,
    ) -> dict:
        """
        Create surface from a bulk crystal structure.

        Parameters
        ----------
        miller_indices : list or tuple (optional)
            Miller indices of the surface. The default value is ``(1, 0, 0)``.
        termination : int (optional)
            Determine termination of the surface.
        tolerance : float (optional)
            Numerical tolerance. The default value is ``0.005``.
        symprec : float (optional)
            Tolerance parameter for spglib. The default value is ``0.005``.
        angle_tolerance : float (optional)
            Tolerance parameter for spglib. The default value is ``-1.0``.
        hall_number : int (optional)
            The argument to constrain the space-group-type search only for the Hall symbol
            corresponding to it. The default number is ``0``.

        Returns
        -------
        dict
            Dictionary containing the surface data.
        """
        sg_details = self.structure.calc_space_group(
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            hall_number=hall_number,
            return_standardized_structure=True,
            no_idealize=False,
        )
        return _surface_create(sg_details, miller_indices, termination, tolerance)

    def to_aiida_surfacedata(
        self,
        miller_indices: Union[Tuple[int], List[int]] = (1, 0, 0),
        termination: int = 1,
        tolerance: float = 0.005,
        symprec: float = 0.005,
        angle_tolerance: float = -1.0,
        hall_number: int = 0,
    ):
        """
        Create surface from a bulk crystal structure.

        Parameters
        ----------
        miller_indices : list or tuple (optional)
            Miller indices of the surface. The default value is ``(1, 0, 0)``.
        termination : int (optional)
            Determine termination of the surface.
        tolerance : float (optional)
            Numerical tolerance. The default value is ``0.005``.
        symprec : float (optional)
            Tolerance parameter for spglib. The default value is ``0.005``.
        angle_tolerance : float (optional)
            Tolerance parameter for spglib. The default value is ``-1.0``.
        hall_number : int (optional)
            The argument to constrain the space-group-type search only for the Hall symbol
            corresponding to it. The default number is ``0``.

        Returns
        -------
        SurfaceData
            AiiDA surface data node.
        """
        backend_module = _return_ext_interface_modules("aiida")
        label = self.structure.attributes.get("label", "")
        surf_dict = self.create_surface(
            miller_indices, termination, tolerance, symprec, angle_tolerance, hall_number
        )
        return backend_module._create_surface_node(label, surf_dict, miller_indices, termination)

    def store_surfaces_in_aiida_db(
        self,
        miller_indices: Tuple[int] = (1, 0, 0),
        tolerance: float = 0.005,
        symprec: float = 0.005,
        angle_tolerance: float = -1.0,
        hall_number: int = 0,
        group_label: str = None,
        group_description: str = None,
    ):
        """
        Store surfaces into the AiiDA-database.

        Parameters
        ----------
        miller_indices : list or tuple (optional)
            Miller indices of the surface. The default value is ``(1, 0, 0)``.
        tolerance : float (optional)
            Numerical tolerance. The default value is ``0.005``.
        symprec : float (optional)
            Tolerance parameter for spglib. The default value is ``0.005``.
        angle_tolerance : float (optional)
            Tolerance parameter for spglib. The default value is ``-1.0``.
        hall_number : int (optional)
            The argument to constrain the space-group-type search only for the Hall symbol
            corresponding to it. The default number is ``0``.
        group_label : str (optional)
            Label of the AiiDA group.
        group_description : str (optional)
            Description of the AiiDA group.

        Returns
        -------
        list
            List containing dictionary of all surface nodes.
        """
        backend_module = _return_ext_interface_modules("aiida")
        ter = 1
        surfaces = []
        surface_dict = True
        bulk_label = "" if self.structure.label is None else self.structure.label
        while surface_dict is not None:
            surface_dict = self.create_surface(
                miller_indices, ter, tolerance, symprec, angle_tolerance, hall_number
            )
            if surface_dict is not None:
                surfaces.append(
                    {
                        "label": (
                            bulk_label
                            + "_"
                            + "".join(str(idx0) for idx0 in miller_indices)
                            + "_"
                            + str(ter)
                        ),
                        "surface_dict": surface_dict,
                        "miller_indices": miller_indices,
                        "termination": ter,
                    }
                )
                ter += 1
        return backend_module._store_surfaces(group_label, group_description, surfaces)

    def generate_surface_slabs(
        self,
        miller_indices: Union[Tuple[int], List[int]] = (1, 0, 0),
        nr_layers: int = 5,
        periodic: bool = False,
        vacuum: float = 10.0,
        vacuum_factor: float = 0.0,
        symmetrize: bool = True,
        tolerance: float = 0.01,
        symprec: float = 0.005,
        angle_tolerance: float = -1.0,
        hall_number: int = 0,
    ) -> Union[StructureCollection, None]:
        """
        Generate surface slabs with all terminations for a certain direction given by its
        miller indices.

        Parameters
        ----------
        miller_indices : list or tuple (optional)
            Miller indices of the surface. The default value is ``(1, 0, 0)``.
        nr_layers : int (optional)
            Number of repititions of the underlying periodic surface cell. The default
            value is ``5``.
        periodic : bool (optional)
            Whether to apply periodic boundary conditions in the direction normal to the
            surface plane. The default value is ``False``.
        vacuum : float (optional)
            Vacuum space added at the top and bottom of the slab. The default value
            is ``10.0``.
        vacuum_factor : float (optional)
            Alternatively to the ``vacuum``-parameter the amount of vacuum can be set as a
            multiple of the slab size. The method is only applied if the parameter is
            larger than zero. The default value is ``0.0``.
        symmetrize : bool (optional)
            Create slabs that have the same termination on both sides. The default value
            is ``True``.
        tolerance : float (optional)
            Numerical tolerance. The default value is ``0.005``.
        symprec : float (optional)
            Tolerance parameter for spglib. The default value is ``0.005``.
        angle_tolerance : float (optional)
            Tolerance parameter for spglib. The default value is ``-1.0``.
        hall_number : int (optional)
            The argument to constrain the space-group-type search only for the Hall symbol
            corresponding to it. The default number is ``0``.

        Returns
        -------
        StructureCollection
            Collection of the generated surface slabs.
        """
        surfaces_collection = StructureCollection()
        ter = 1
        surface = self.create_surface(
            miller_indices, ter, tolerance, symprec, angle_tolerance, hall_number
        )
        bulk_label = "" if self.structure.label is None else self.structure.label
        while surface is not None:
            surface_slab = _surface_create_slab(
                surface, nr_layers, periodic, vacuum, vacuum_factor, symmetrize
            )
            label = (
                bulk_label + "_" + "".join(str(idx0) for idx0 in miller_indices) + "_" + str(ter)
            )
            surfaces_collection.append(label, **surface_slab)
            ter += 1
            surface = self.create_surface(
                miller_indices, ter, tolerance, symprec, angle_tolerance, hall_number
            )

        if len(surfaces_collection) > 0:
            return surfaces_collection
