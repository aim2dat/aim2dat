"""Classes and methods to analyse and manipulate molecular or crystalline structures."""

from aim2dat.strct.structure import Structure
from aim2dat.strct.structure_collection import StructureCollection
from aim2dat.strct.structure_importer import StructureImporter
from aim2dat.strct.structure_operations import StructureOperations
from aim2dat.strct.surface import SurfaceGeneration
from aim2dat.strct.validation import SamePositionsError

__all__ = [
    "Structure",
    "StructureCollection",
    "StructureImporter",
    "StructureOperations",
    "SurfaceGeneration",
    "SamePositionsError",
]
