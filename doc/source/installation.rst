=================================
Overview and Package Dependencies
=================================

This library should work on linux, windows and mac OS operating systems provided that are working python environment is installed.
Detailed steps on how to install the package are given below:

.. toctree::
    :maxdepth: 1

    inst_windows
    inst_linux

Package Dependencies
====================

This package comes with the following core dependencies:

* |packaging_page| is managing the correct versions of the depending packages.
* |numpy_page| and |scipy_page| are used for numerical calculations.
* |matplotlib_page| and |pandas_page| are used for data representation.
* |ruamel.yaml_page| is used to write and read (yaml) configuration files.
* |h5py_page| is used to write and read (hdf5) data files.
* |ase_page| and |spglib_page| are used to handle atomic structures.
* |tqdm_page| for visualizing progress bars.

Some features demand additional dependencies:

* *aiida*: To use the AiiDA interface and the high-throughput workflows the |aiida_page| package and the |seekpath_page| are needed.
* *crystal_structure_generation*: The generation of random crystals is based on |pyxtal_page|
* *phonons*: To make use of the phonopy interface the |phonopy_page| is needed.
* *database_interfaces*: There are several packages used for the different interfaces to online databases:

  * |requests_page| is the base package that manages the retrieval of data from the online servers.
  * |pymatgen_page|, |msgpack_page| and |boto3_page| are used for the interface to materials project.
  * |qmpy_rester_page| is used the interface to the open quantum materials database.

* *graphs*: To create graphs from atomic structures the |nx_page| and |graphviz_page| package are needed.
* *plots*: The |plotly_page| package serves as a secondary plotting engine.
* *ml*: The machine learning routines are based on |dscribe_page| and |sklearn_page|.
* *tests*: Includes all packages for the testing infrastructure.
* *doc*: Includes all packages to produce the documentation page.

All dependencies can be installed via the following command:

.. code-block:: bash

    pip install aim2dat[aiida,crystal_structure_generation,phonons,database_interfaces,graphs,ml,plots]


.. |aiida_page| raw:: html

   <a href="https://www.aiida.net" target="_blank">AiiDA</a>

.. |ase_page| raw:: html

   <a href="https://wiki.fysik.dtu.dk/ase/" target="_blank">ASE</a>

.. |boto3_page| raw:: html

   <a href="https://pypi.org/project/boto3/" target="_blank">boto3</a>

.. |dscribe_page| raw:: html

   <a href="https://singroup.github.io/dscribe/" target="_blank">Dscribe</a>

.. |graphviz_page| raw:: html

   <a href="https://pypi.org/project/graphviz/" target="_blank">graphviz</a>

.. |h5py_page| raw:: html

   <a href="https://docs.h5py.org/" target="_blank">h5py</a>

.. |matplotlib_page| raw:: html

   <a href="https://matplotlib.org/" target="_blank">matplotlib</a>

.. |msgpack_page| raw:: html

   <a href="https://pypi.org/project/msgpack/" target="_blank">msgpack</a>

.. |numpy_page| raw:: html

   <a href="https://www.numpy.org" target="_blank">NumPy</a>

.. |nx_page| raw:: html

   <a href="https://networkx.org/" target="_blank">NetworkX</a>

.. |qmpy_rester_page| raw:: html

   <a href="https://pypi.org/project/qmpy-rester/" target="_blank">qmpy-rester</a>

.. |pandas_page| raw:: html

   <a href="https://pandas.pydata.org/" target="_blank">pandas</a>

.. |plotly_page| raw:: html

   <a href="https://plotly.com/python/" target="_blank">plotly</a>

.. |packaging_page| raw:: html

   <a href="https://packaging.python.org/" target="_blank">Packaging</a>

.. |phonopy_page| raw:: html

   <a href="http://phonopy.github.io/phonopy/" target="_blank">phonopy</a>

.. |pymatgen_page| raw:: html

   <a href="https://pymatgen.org/" target="_blank">pymatgen</a>

.. |pyxtal_page| raw:: html

   <a href="https://pyxtal.readthedocs.io/" target="_blank">PyXtaL</a>

.. |requests_page| raw:: html

   <a href="https://docs.python-requests.org/" target="_blank">Requests</a>

.. |ruamel.yaml_page| raw:: html

   <a href="https://pypi.org/project/ruamel.yaml/" target="_blank">ruamel.yaml</a>

.. |sklearn_page| raw:: html

   <a href="https://scikit-learn.org/" target="_blank">scikit learn</a>

.. |scipy_page| raw:: html

   <a href="https://scipy.org/" target="_blank">SciPy</a>

.. |seekpath_page| raw:: html

   <a href="https://seekpath.readthedocs.io/" target="_blank">SeeK-path</a>

.. |spglib_page| raw:: html

   <a href="https://spglib.readthedocs.io/" target="_blank">spglib</a>

.. |tqdm_page| raw:: html

   <a href="https://pypi.org/project/tqdm/" target="_blank">tqdm</a>
