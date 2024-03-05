=====
Plots
=====

.. toctree::
    :maxdepth: 1
    :hidden:

    plots-simple_plot
    plots-properties_and_functions
    plots-subplots

Specific topics on how to adjust the plots are covered in the following sections:

* The base using principle is described :doc:`here <plots-simple_plot>` using the :class:`SimplePlot <aim2dat.plots.SimplePlot>`.
* Settings of the plot like e.g. the axis labels or ranges or the ratio of the plot can be adjusted using the corresponding properties of the plot classes described :doc:`here <plots-properties_and_functions>`.
* How to plot data sets in several subplots is described :doc:`here <plots-subplots>`.


List of all plot classes
========================

The following plot classes are implemented:

* :class:`BandStructurePlot <aim2dat.plots.BandStructurePlot>`: Class to plot electronic or phonon band structures. Additionally, the class implements functions to analyse and compare band structures.
* :class:`DOSPlot <aim2dat.plots.DOSPlot>`: Class to plot electronic or phonon density of states. The class is focused on analysing and plotting the different orbital contributions of a projected density of states.
* :class:`BandStructureDOSPlot <aim2dat.plots.BandStructureDOSPlot>`: Combines the previous two classes to plot band structures and density of states side by side.
* :class:`PartialChargesPlot <aim2dat.plots.PartialChargesPlot>`: Class to plot partial charges.
* :class:`PartialRDFPlot <aim2dat.plots.PartialRDFPlot>`: Class to plot radial distribution functions.
* :class:`PhasePlot <aim2dat.plots.PhasePlot>`: Class to plot formation energies, stabilities or other properties with regard to the chemical composition.
* :class:`PlanarFieldPlot <aim2dat.plots.PlanarFieldPlot>`: Class to plot planar fields intended to be used to analyse output data from the |critic2_page| code.
* :class:`SimplePlot <aim2dat.plots.SimplePlot>`: Class that implements a simple and flexible interface to all the plotting features.
* :class:`SpectrumPlot <aim2dat.plots.SpectrumPlot>`: Class to plot spectra.
* :class:`SurfacePlot <aim2dat.plots.SurfacePlot>`: Class to plot surface energies and other surface properties.

Related examples
================

* :doc:`Plotting the band structure and projected density of states (pDOS) from CP2K output-files <examples/plots-band_structure_and_pdos_cp2k>`
* :doc:`Plotting the band structure and projected density of states (pDOS) from FHI-aims output files <examples/plots-band_structure_and_pdos_fhi-aims>`
* :doc:`Plotting the band structure, projected density of states (pDOS) and thermal properties from phonopy output-files <examples/plots-band_structure_and_pdos_phonopy>`
* :doc:`Plotting the band structure and projected density of states (pDOS) from Quantum ESPRESSO output-files <examples/plots-band_structure_and_pdos_qe>`
* :doc:`Plotting the band structure and projected density of states (pDOS) from Materials Project <examples/plots-band_structure_materials_project>`
* :doc:`Plotting atomic partial charges from Critic2 output-files <examples/plots-partial_charges_critic2>`
* :doc:`Using the SimplePlot class as a flexible plotting framework <examples/plots-simple_plot>`
* :doc:`How to use the plots package to plot a x-ray absorption spectrum <examples/plots-spectroscopy>`
* :doc:`F-Fingerprint and partial radial distribution function (prdf) <examples/strct-partial_rdf>`

.. |critic2_page| raw:: html

   <a href="https://aoterodelaroza.github.io/critic2/" target="_blank">Critic2</a>
