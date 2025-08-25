"""
AiiDA data classes for surfaces.
"""

# Standard library imports
import copy

# Third party library imports
import numpy as np
from aiida.orm import Data

# Internal library imports
from aim2dat.strct.validation import (
    _structure_validate_cell,
    _structure_validate_el_pos,
)


class SurfaceData(Data):
    """AiiDA data object to store surface data."""

    def __init__(self, aperiodic_dir=2, miller_indices=(1, 0, 0), termination=1, **kwargs):
        """Initialize object."""
        super().__init__(**kwargs)
        self.aperiodic_dir = aperiodic_dir
        self.miller_indices = miller_indices
        self.termination = termination

    @property
    def aperiodic_dir(self):
        """Non-periodic direction of the slab."""
        return self.base.attributes.get("aperiodic_dir")

    @aperiodic_dir.setter
    def aperiodic_dir(self, value):
        if not isinstance(value, int):
            raise TypeError("`miller_indices` needs to be of type int.")
        if value not in [0, 1, 2]:
            raise ValueError("`aperiodic_dir` needs to be a number between 0 and 2.")
        self.base.attributes.set("aperiodic_dir", value)

    @property
    def miller_indices(self):
        """Miller indices of the surface facet."""
        return copy.deepcopy(self.base.attributes.get("miller_indices"))

    @miller_indices.setter
    def miller_indices(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError("`miller_indices` needs to be of type tuple or list.")
        if len(value) != 3:
            raise ValueError("`miller_indices` needs to have a length of 3.")
        if not all(isinstance(val0, int) for val0 in value):
            raise ValueError("All entries of `miller_indices` need to be of type integer.")
        self.base.attributes.set("miller_indices", tuple(value))

    @property
    def termination(self):
        """Termination of the surface facet."""
        return self.base.attributes.get("termination")

    @termination.setter
    def termination(self, value):
        if not isinstance(value, int):
            raise TypeError("`termination` needs to be of type int.")
        if value < 1:
            raise ValueError("`termination` needs to be larger than 0.")
        self.base.attributes.set("termination", value)

    @property
    def repeating_structure(self):
        """Repeating structure of the surface."""
        return self._get_structure("repeating_structure")

    @property
    def top_terminating_structure(self):
        """Top terminating structure of the surface."""
        return self._get_structure("top_structure")

    @property
    def top_terminating_structure_nsym(self):
        """Top terminating non-symmetric structure of the surface."""
        return self._get_structure("top_structure_nsym")

    @property
    def bottom_terminating_structure(self):
        """Bottom terminating structure of the surface."""
        return self._get_structure("bottom_structure")

    @property
    def surface_area(self):
        """Surface area."""
        if self.repeating_structure is None:
            return None
        else:
            periodic_dirs = [idx0 for idx0 in range(3) if idx0 != self.aperiodic_dir]
            cell_v1 = np.array(self.repeating_structure["cell"][periodic_dirs[0]])
            cell_v2 = np.array(self.repeating_structure["cell"][periodic_dirs[1]])
            return np.linalg.norm(np.cross(cell_v1, cell_v2))

    def set_repeating_structure(
        self, elements, positions, cell, is_cartesian, translational_vector, kinds=None
    ):
        """
        Set repeating structure.

        Parameters
        ----------
        elements : list
            List of elements or atomic numbers.
        positions : list
            Nested list of the coordinates, either in cartesian or scaled coordinates.
        cell : list or np.array
            Nested 3x3 list of the cell vectors in angstrom.
        is_cartesian : bool
            Whether the coordinates are cartesian or scaled.
        translational_vector : list or np.array
            Translational shift between two layers of repeating units.
        """
        self._set_structure(
            "repeating_structure",
            elements,
            positions,
            cell,
            is_cartesian,
            add_kws={"translational_vector": translational_vector},
        )

    def set_top_terminating_structure(self, elements, positions, cell, is_cartesian, kinds=None):
        """
        Set top-terminating structure.

        Parameters
        ----------
        elements : list
            List of elements or atomic numbers.
        positions : list
            Nested list of the coordinates, either in cartesian or scaled coordinates.
        cell : list or np.array
            Nested 3x3 list of the cell vectors in angstrom.
        is_cartesian : bool
            Whether the coordinates are cartesian or scaled.
        """
        self._set_structure("top_structure", elements, positions, cell, is_cartesian)

    def set_top_terminating_structure_nsym(self, elements, positions, cell, is_cartesian):
        """
        Set top-terminating non-symmetric structure.

        Parameters
        ----------
        elements : list
            List of elements or atomic numbers.
        positions : list
            Nested list of the coordinates, either in cartesian or scaled coordinates.
        cell : list or np.array
            Nested 3x3 list of the cell vectors in angstrom.
        is_cartesian : bool
            Whether the coordinates are cartesian or scaled.
        """
        self._set_structure("top_structure_nsym", elements, positions, cell, is_cartesian)

    def set_bottom_terminating_structure(
        self, elements, positions, cell, is_cartesian, kinds=None
    ):
        """
        Set bottom-terminating non-symmetric structure.

        Parameters
        ----------
        elements : list
            List of elements or atomic numbers.
        positions : list
            Nested list of the coordinates, either in cartesian or scaled coordinates.
        cell : list or np.array
            Nested 3x3 list of the cell vectors in angstrom.
        is_cartesian : bool
            Whether the coordinates are cartesian or scaled.
        """
        self._set_structure("bottom_structure", elements, positions, cell, is_cartesian)

    def _set_structure(self, attr_label, elements, positions, cell, is_cartesian, add_kws={}):
        cell, inv_cell = _structure_validate_cell(cell)
        elements, positions_cart, _ = _structure_validate_el_pos(
            elements, positions, [True, True, False], cell, inv_cell, is_cartesian
        )
        strct_dict = {
            "elements": elements,
            "positions": positions,
            "cell": cell,
            "is_cartesian": True,
        }
        for keyw, value in add_kws.items():
            strct_dict[keyw] = value
        self.base.attributes.set(attr_label, strct_dict)

    def _get_structure(self, attr_label):
        structure = None
        if attr_label in self.base.attributes:
            structure = copy.deepcopy(self.base.attributes.get(attr_label))
        return structure
