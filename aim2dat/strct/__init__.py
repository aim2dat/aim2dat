"""Modules to analyze the chemical environment and the coordination of atoms."""

from aim2dat.strct.structure_collection import StructureCollection
from aim2dat.strct.strct import Structure
from aim2dat.strct.structure_operations import StructureOperations
from aim2dat.strct.surface import SurfaceGeneration
from aim2dat.strct.structure_importer import StructureImporter
from aim2dat.strct.strct_validation import SamePositionsError

__all__ = [
    "StructureCollection",
    "Structure",
    "StructureOperations",
    "SurfaceGeneration",
    "StructureImporter",
    "SamePositionsError",
]
