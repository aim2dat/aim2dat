===================
Structural Analysis
===================

.. toctree::
    :maxdepth: 1
    :hidden:

    stra-structure
    stra-multiple_structures
    stra-structure_importer
    stra-surfaces

.. figure:: Images/structure_analysis.svg
    :width: 500px
    :align: center
    :alt: structure_analysis.svg

    Flowchart showing the relations between the different classes of the :mod:`strct <aim2dat.strct>` subpackage.


The central object storing all structural information is the :class:`Structure <aim2dat.strct.Structure>` object (:doc:`more details <stra-structure>`).
Due to the initial high-throughput background of this library, one usually works with multiple structures and not a single one.
To avoid frequent implementations of loops by the user to call methods on all the structures, we provide the
:class:`StructureOperations <aim2dat.strct.StructureOperations>` and
:class:`StructureCollection <aim2dat.strct.StructureCollection>` classes to wrap analysis and manipulation methods
and as a datacontainer for multiple structures, respectively (:doc:`more details <stra-multiple_structures>`).

The :class:`StructureImporter <aim2dat.strct.StructureImporter>` class has the purpose to "obtain" input structures and as such it acts as an interface
to online databases but also offers  the capability to randomly generate crystals (:doc:`more details <stra-structure_importer>`).
Last but not least, the :class:`SurfaceGeneration <aim2dat.strct.SurfaceGeneration>` class aims to create surface related structures and data
(:doc:`more details <stra-surfaces>`).

.. note::
   All classes in the `structure_analysis` subpackage underly the principle of object-oriented programming.
   The :class:`StructureCollection <aim2dat.strct.StructureCollection>`, :class:`StructureImporter <aim2dat.strct.StructureImporter>` and
   :class:`StructureOperations <aim2dat.strct.StructureOperations>` classes wrap one or several :class:`Structure <aim2dat.strct.Structure>` instances.
   If the :class:`Structure <aim2dat.strct.Structure>` objects are changed outside of these classes it can affect their functionality.

   In order to avoid interferences one can use the :func:`copy <aim2dat.strct.Structure.copy>` of the :class:`Structure <aim2dat.strct.Structure>` object
   function to create an unrelated deepcopy of the object.
