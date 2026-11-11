===============
AiiDA Processes
===============

Input and output parameters for all AiiDA work chains and calculation jobs.

chargemol
=========

.. aiida:calcjob:: ChargemolCalculation
    :module: aim2dat.wfl.aiida.chargemol.calcjobs

CP2K
====

.. aiida:calcjob:: Cp2kCalculation
    :module: aim2dat.wfl.aiida.cp2k.calcjobs


.. aiida:calcjob:: CubecruncherCalculation
    :module: aim2dat.wfl.aiida.cubecruncher.calcjobs

.. aiida:workchain:: FindSCFParametersWorkChain
    :module: aim2dat.wfl.aiida.cp2k.find_scf_p_work_chain

.. aiida:workchain:: GeoOptWorkChain
    :module: aim2dat.wfl.aiida.cp2k.geo_opt_work_chain

.. aiida:workchain:: CellOptWorkChain
    :module: aim2dat.wfl.aiida.cp2k.cell_opt_work_chain

.. aiida:workchain:: BandStructureWorkChain
    :module: aim2dat.wfl.aiida.cp2k.band_structure_work_chain

.. aiida:workchain:: EigenvaluesWorkChain
    :module: aim2dat.wfl.aiida.cp2k.eigenvalues_work_chain

.. aiida:workchain:: PDOSWorkChain
    :module: aim2dat.wfl.aiida.cp2k.pdos_work_chain

.. aiida:workchain:: PartialChargesWorkChain
    :module: aim2dat.wfl.aiida.cp2k.partial_charges_work_chain

.. aiida:workchain:: CubeWorkChain
    :module: aim2dat.wfl.aiida.cp2k.cube_work_chain

.. aiida:workchain:: PlanarFieldsWorkChain
    :module: aim2dat.wfl.aiida.cp2k.planar_fields_work_chain

.. aiida:workchain:: ChargeDensityDifferenceWorkChain
    :module: aim2dat.wfl.aiida.cp2k.charge_density_difference_work_chain

.. aiida:workchain:: ElectronicPropertiesWorkChain
    :module: aim2dat.wfl.aiida.cp2k.combined_work_chains

.. aiida:workchain:: SurfaceOptWorkChain
    :module: aim2dat.wfl.aiida.cp2k.combined_work_chains

critic2
=======

.. aiida:calcjob:: Critic2Calculation
    :module: aim2dat.wfl.aiida.critic2.calcjobs

enumlib
=======

.. aiida:calcjob:: EnumlibCalculation
    :module: aim2dat.wfl.aiida.enumlib.calcjobs
