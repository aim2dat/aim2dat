=====================
aim2dat Documentation
=====================

The Automated Ab-Initio Materials Modeling and Data Analysis Toolkit (``aim2dat``) is an open-source
python library aimed to facilitate high-throughput studies based on density functional theory calculations.
As such this package includes functionality to assist and automatize every step of the overall procedure,
ranging data-mining and workflow management to post-processing routines and plotting features:

.. figure:: Images/high-throughput_general_workflow.svg
    :width: 800px
    :align: center
    :alt: high-throughput_general_workflow.svg

    The workflow of a high-throughput project including the different steps and
    the related sub-packages and of this library.

The main functionalities related to each of the steps are the following:

1. **Create initial structure pool:** The `strct` subpackage (:doc:`more details <stra-overview>`) implements several classes to
   handle larger quantities of crystals, molecules or related atomic structures. It interfaces with online
   databases and has several analysis methods implemented.
2. **Run high-throughput workflow:** The high-throughput capabilities are driven by the |aiida_page| python
   package. This library implements additional calculation jobs and work chains for |cp2k_page|, |critic2_page|, |chargemol_page| and |enumlib_page|.
   To enhance the modularity and easily retrieve calculated results the `WorkflowBuilder` and `MultipleWorkflowBuilder`
   classes take care of the workflow's provenance and checks which properties have been already calculated (:doc:`more details <htw-overview>`).
3. **Analyse output data:** The output data of the workflow can be readily plotted using the different plot classes
   (:doc:`more details <plots-simple_plot>`). Two backends are implemented: The |plotly_page| package is used to visualize plots interactively
   and the |matplotlib_page| package is used for high-quality and publication-ready plots. The `fct` sub-package implements methods to compare and analyse
   2-dimensional functions in a quantitative fashion (:doc:`more details <fa-overview>`). Structural properties can be then be analysed using the infrastructure
   of the `strct` subpackage (:doc:`more details <stra-overview>`).
4. **Train ML models:**
   To exploit the produced data, a direct interface to the |sklearn_page| package is given by the `StructureTransformer`
   classes that allow to extract features from crystalline or molecular structures and can be integrated in pipelines (:doc:`more details <ml-overview>`).


Feature List
============

* Managing and analysing sets of crystals and molecules.
* Ab-initio high-throughput calculations based on |aiida_page|.
* Plotting material's properties such as electronic band structures, projected density of states or phase diagrams.
* Interface to machine learning routines via sci-kit learn.
* Function analysis: discretizing and comparing 2-dimensional functions.
* Parsers for the DFT codes |cp2k_page|, |fhi-aims_page|, |qe_page| as well as |phonopy_page| and |critic2_page|.


Contributing
============

Contributions are very welcome and are directly handled via the code's `github repository <https://github.com/aim2dat/aim2dat>`_.
Bug reports, feature requests or general discussions can be accomplished by filing an `issue <https://github.com/aim2dat/aim2dat/issues>`_.
Extensions or changes to the code can also be directly suggested by opening a `pull request <https://github.com/aim2dat/aim2dat/pulls>`_.

A few guidelines suggested for code contributions:

* Please try to abide the |python_styleguide| and |google_styleguide| style guides for python.
* Style checks are implemented using the python packages |black_pypi| and |flake8_pypi|.
  Before submitting the pull request the checks can be performed via the commands ``black --diff .`` and ``flake8`` executed in the root folder of the library.
* Tests are performed using the python package |pytest_pypi|. Additional tests for new features are very welcome, ideally each pull request should maintain
  or increase the test coverage of the code.
* Please use doc-strings to describe the usage of the code. The doc-strings should be able to explain how the class/function is used, e.g. by showing code-snippets
  to highlight the most important features. We currently use NumPy-docstrings, examples can be found |sphinx_numpy|.
* Additional documentation, like a jupyter-notebook for the example section or a dedicated section explaining how to use the feature is very welcome as well.

The package |precommit_pypi| can be used to run style checks before every commit by executing the following commands the main folder of the repository:

.. code-block:: bash

    pip install pre-commit
    pre-commit install


.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Installation

    installation

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: User Guide

    user_guide
    stra-overview
    htw-overview
    plots-overview
    fa-overview
    ml-overview

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Examples

    examples

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Python API

    autoapi/aim2dat/index

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: AiiDA Processes

    api-aiida_processes

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Changelog

    changelog


.. |aiida_page| raw:: html

   <a href="https://www.aiida.net" target="_blank">AiiDA</a>

.. |black_pypi| raw:: html

   <a href="https://pypi.org/project/black/" target="_blank">black</a>

.. |chargemol_page| raw:: html

   <a href="https://sourceforge.net/projects/ddec/" target="_blank">chargemol</a>

.. |cp2k_page| raw:: html

   <a href="https://www.cp2k.org" target="_blank">CP2K</a>

.. |critic2_page| raw:: html

   <a href="https://aoterodelaroza.github.io/critic2/" target="_blank">critic2</a>

.. |enumlib_page| raw:: html

   <a href="https://github.com/msg-byu/enumlib" target="_blank">enumlib</a>

.. |fhi-aims_page| raw:: html

   <a href="https://fhi-aims.org/" target="_blank">FHI-aims</a>

.. |flake8_pypi| raw:: html

   <a href="https://pypi.org/project/flake8/" target="_blank">flake8</a>

.. |google_styleguide| raw:: html

   <a href="https://google.github.io/styleguide/pyguide.html" target="_blank">google</a>

.. |matplotlib_page| raw:: html

   <a href="https://matplotlib.org/" target="_blank">matplotlib</a>

.. |phonopy_page| raw:: html

   <a href="http://phonopy.github.io/phonopy/" target="_blank">phonopy</a>

.. |plotly_page| raw:: html

   <a href="https://plotly.com/python/" target="_blank">plotly</a>

.. |precommit_pypi| raw:: html

   <a href="https://pypi.org/project/pre-commit/" target="_blank">pre-commit</a>

.. |pytest_pypi| raw:: html

   <a href="https://pypi.org/project/pytest/" target="_blank">pytest</a>

.. |python_styleguide| raw:: html

   <a href="https://www.python.org/dev/peps/pep-0008/" target="_blank">python PEP-8</a>

.. |qe_page| raw:: html

   <a href="https://www.quantum-espresso.org/" target="_blank">Quantum ESPRESSO</a>

.. |sklearn_page| raw:: html

   <a href="https://scikit-learn.org/" target="_blank">scikit learn</a>

.. |sphinx_numpy| raw:: html

   <a href="https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html" target="_blank">here</a>
