========================
Scikit-learn integration
========================

This page gives an overview of classes and methods that can be used with the scikit-learn python package.

List of structure transformers
==============================

The structure transformer classes are based on scikit-learn's |scikit_estimator| class and can be combined with other transformer or estimator classes by defining pipelines (more details on scikit-learn's pipeline framework can be found |scikit_compose|).


.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Python class
     - Description
   * - :class:`StructureCompositionTransformer <aim2dat.ml.transformers.StructureCompositionTransformer>`
     - Transforms the structures into feature vectors consisting of the elemental concentrations.
   * - :class:`StructureDensityTransformer <aim2dat.ml.transformers.StructureDensityTransformer>`
     -  Transforms the perdiodic structures into feature vectors of the atomic density (nr atoms per cell volume) of each element.
   * - :class:`StructureCoordinationTransformer <aim2dat.ml.transformers.StructureCoordinationTransformer>`
     -  Transforms the structures into feature vectors consisting of the coordination number and distance to neighbouring atoms for each element pair.
   * - :class:`StructureChemOrderTransformer <aim2dat.ml.transformers.StructureChemOrderTransformer>`
     -  Transforms the structures into their Warren Cowley like order parameters introduced in :doi:`10.1103/PhysRevB.96.024104`.
   * - :class:`StructureFFPrintTransformer <aim2dat.ml.transformers.StructureFFPrintTransformer>`
     -  Transforms the periodic structures into their F-Fingerprint as defined in :doi:`10.1063/1.3079326`.
   * - :class:`StructurePRDFTransformer <aim2dat.ml.transformers.StructurePRDFTransformer>`
     -  Transforms structures the into their partial radial distribution functions (pRDF) as defined in :doi:`10.1103/PhysRevB.89.205118`.
   * - :class:`StructureMatrixTransformer <aim2dat.ml.transformers.StructureMatrixTransformer>`
     -  Transforms the structures into different types of interaction matrices as defined in :doi:`10.1002/qua.24917`. The implementation is based on the |dsribe_page| python package.
   * - :class:`StructureACSFTransformer <aim2dat.ml.transformers.StructureACSFTransformer>`
     -  Transforms the structures into their ACSF descriptor as defined in :doi:`10.1063/1.3553717`. The implementation is based on the |dsribe_page| python package.
   * - :class:`StructureSOAPTransformer <aim2dat.ml.transformers.StructureSOAPTransformer>`
     -  Transforms the structures into their SOAP descriptor as defined in :doi:`10.1103/PhysRevB.87.184115`. The implementation is based on the |dsribe_page| python package.
   * - :class:`StructureMBTRTransformer <aim2dat.ml.transformers.StructureMBTRTransformer>`
     -  Transforms the structures into their many-body tensor representation (MBTR) as defined in :doi:`10.1088/2632-2153/aca005`. The implementation is based on the |dsribe_page| python package.


List of custom metrics and kernels
==================================

The following metrics and kernels are built on-top of the :class:`StructureFFPrintTransformer <aim2dat.ml.transformers.StructureFFPrintTransformer>` and can be used for different ML models:

* The :meth:`ffprint_cosine <aim2dat.ml.metrics.ffprint_cosine>` function allows the metric based on the cosine distance originally introduced in :doi:`10.1063/1.3079326`  for the F-Fingerprint descriptor to be used with e.g. the k-neighbours model.
* The :meth:`krr_ffprint_cosine <aim2dat.ml.kernels.krr_ffprint_cosine>` function calculates a kernel directly derived from the F-Fingerprint cosine distance to be used with the kernel ridge regression model.
* The :meth:`krr_ffprint_laplace <aim2dat.ml.kernels.krr_ffprint_laplace>` kernel implements the laplacian kernel of the cosine distance for the F-Fingerprint descriptor to be used with the kernel ridge regression model.


.. |dsribe_page| raw:: html

   <a href="https://singroup.github.io" target="_blank">dscribe</a>

.. |scikit_estimator| raw:: html

   <a href="https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html" target="_blank">BaseEstimator</a>

.. |scikit_compose| raw:: html

   <a href="https://scikit-learn.org/stable/modules/compose.html" target="_blank">here</a>
