=========================
High-throughput Workflows
=========================


.. toctree::
    :maxdepth: 1
    :hidden:

    htw-workflow_builders
    htw-cp2k_calculations


Running large quantity of calculations in an automatized fashion brings along new challenges such
as data management, error handling and back-tracing but also analyzing trends and features of the
results.
This library offers some tools that are designed to make high-throughput calculations more
user-friendly and easy to run and relies on the |aiida_page| python package to handle the computational
workflows and manage the data storage and provenance.
It is therefore strongly recommended to get familiar with the general working principles of the pacakge via its
|aiida_docs|.


AiiDA processes
===============

Basis of any high-throughput study are predefined workflows that are applied on all crystalline or molecular structures of an initial data pool.
Once the library is installed all relevant work chains for high-throughput workflows should be visible when typing:

::

	$ verdi plugin list aiida.workflows

* All implemented AiiDA ``CalcJob`` and ``WorkChain`` classes are listed as part of the :doc:`API <api-aiida_processes>`.
* A detailed overview of the ``CalcJob`` and ``WorkChain`` classes using the |cp2k_page| code is given :doc:`here <htw-cp2k_calculations>`.

Workflow builder classes
========================

The two workflow builder classes, namely the :class:`WorkflowBuilder <aim2dat.aiida_workflows.workflow_builder.WorkflowBuilder>` and
:class:`MultipleWorkflowBuilder <aim2dat.aiida_workflows.workflow_builder.MultipleWorkflowBuilder>` class,
have been implemented to facilitate the design and management of high-throughput workflows, more details are given :doc:`here <htw-workflow_builders>`.

Related examples
================

* :doc:`Querying the structure pool for the Cs-Te binary system <examples/strct-odb_interfaces>`


Related API instances
=====================

* :class:`WorkflowBuilder <aim2dat.aiida_workflows.workflow_builder.WorkflowBuilder>`
* :class:`MultipleWorkflowBuilder <aim2dat.aiida_workflows.workflow_builder.MultipleWorkflowBuilder>`


.. |aiida_page| raw:: html

   <a href="https://www.aiida.net" target="_blank">AiiDA</a>

.. |aiida_docs| raw:: html

   <a href="https://aiida.readthedocs.io" target="_blank">documentation page</a>

.. |cp2k_page| raw:: html

   <a href="https://www.cp2k.org" target="_blank">CP2K</a>
