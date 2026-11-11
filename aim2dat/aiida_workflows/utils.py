"""Common functions used by several work chains."""

# Third party library imports
from aiida.engine import calcfunction

# Internal library imports
import aim2dat.wfl.aiida.utils as aiida_utils


@calcfunction
def find_equivalent_sites(structure):
    """Wrap the 'find_eq_sites_via_coordination' function to be used as a calcfunction."""
    from warnings import warn

    warn(
        "The `aiida_workflows` sub-package will be removed, please use the `wfl` sub-package "
        + "instead.",
        DeprecationWarning,
        2,
    )
    return aiida_utils._find_equivalent_sites(structure)


@calcfunction
def add_molecule(structure, parameters):
    """Wrap the 'add_structure_position' function to be used as a calcfunction."""
    from warnings import warn

    warn(
        "The `aiida_workflows` sub-package will be removed, please use the `wfl` sub-package "
        + "instead.",
        DeprecationWarning,
        2,
    )
    return aiida_utils._add_molecule(structure, parameters)


@calcfunction
def rotate_molecule(structure, parameters):
    """Wrap the 'rotate_structure' function to be used as a calcfunction."""
    from warnings import warn

    warn(
        "The `aiida_workflows` sub-package will be removed, please use the `wfl` sub-package "
        + "instead.",
        DeprecationWarning,
        2,
    )
    return aiida_utils._rotate_molecule(structure, parameters)


@calcfunction
def seekpath_structure_analysis(structure, parameters):
    """Wrap the seekpath function to be used as a calcfunction."""
    from warnings import warn

    warn(
        "The `aiida_workflows` sub-package will be removed, please use the `wfl` sub-package "
        + "instead.",
        DeprecationWarning,
        2,
    )
    return aiida_utils._seekpath_structure_analysis(structure, parameters)


@calcfunction
def create_surface_slab(surface, nr_layers, parameters):
    """Create surface slab from surface data."""
    from warnings import warn

    warn(
        "The `aiida_workflows` sub-package will be removed, please use the `wfl` sub-package "
        + "instead.",
        DeprecationWarning,
        2,
    )
    return aiida_utils._create_surface_slab(surface, nr_layers, parameters)


@calcfunction
def phonopy_generate_displacements(structure, phonopy_parameters, parameters):
    """
    Generate the symmetry-reduced set of displaced supercells.

    Parameters
    ----------
    structure : StructureData
        The optimized unit cell. NOTE: to keep the displaced cell consistent
        with the optimization, ``symprec`` should match the ``eps_symmetry``
        used by the upstream ``cell_opt`` task (default 0.005 in the standard
        protocols) rather than phonopy's looser default.
    phonopy_parameters : aiida.orm.Dict
        Settings dictionary with the keys:
            ``supercell_matrix`` (list[list[int]] | list[int]) - required,
            ``primitive_matrix`` (str | list) - optional, default ``"auto"``,
            ``symprec`` (float) - optional, default ``1e-5``,
            ``calculator`` (str) - optional calculator, default ``cp2k``.
    parameters : aiida.orm.Dict
        Settings dictionary with the keys:
            ``displacement`` (float) - optional displacement amplitude in
            Angstrom, default ``0.01``.

    Returns
    -------
    dict
        ``supercell_0000`` ... ``supercell_NNNN`` : StructureData
            One displaced supercell per displacement, fed to the per-displacement
            force calculations (:class:`ForceWorkChain`, RUN_TYPE ENERGY_FORCE).
    """
    from warnings import warn

    warn(
        "The `aiida_workflows` sub-package will be removed, please use the `wfl` sub-package "
        + "instead.",
        DeprecationWarning,
        2,
    )
    return aiida_utils._phonopy_generate_displacements(structure, phonopy_parameters, parameters)


@calcfunction
def phonopy_calculate_phonons(structure, phonopy_parameters, parameters):
    """
    Assemble force constants and extract the phonon band structure and DOS.

    Parameters
    ----------
    structure : StructureData
        The same optimized unit cell passed to
        :func:`phonopy_generate_displacements`.
    phonopy_parameters : aiida.orm.Dict
        Settings dictionary with the keys:
            ``supercell_matrix`` (list[list[int]] | list[int]) - required,
            ``primitive_matrix`` (str | list) - optional, default ``"auto"``,
            ``symprec`` (float) - optional, default ``1e-5``,
            ``calculator`` (str) - optional calculator, default ``cp2k``.
    parameters : aiida.orm.Dict
        Post-processing settings:
            ``displacement_dataset`` () -
                dataset of displacements retrieved from ``phonopy_generate_displacements``,
            ``force_sets`` (array) of shape ``(n_displacements, n_atoms, 3) -
                forces for each displacement per atom in each direction,
            ``band_path`` (list) - q-point path segments (from SeekPath),
            ``band_labels`` (list) - high-symmetry point labels,
            ``band_npoints`` (int) - points per segment, default ``101``,
            ``with_eigenvectors`` (bool) - default ``False``,
            ``dos_mesh`` (list[int]) - q-mesh for the DOS, default ``[20,20,20]``,
            ``thermal_properties`` (bool) - default ``False``,
            ``temp_range`` (list) - thermal range.

    Returns
    -------
    dict
        ``band_structure`` : aiida.orm.Dict - phonopy band-structure dict.
        ``total_dos`` : XyData - frequency vs. total phonon DOS.
        ``thermal_properties`` : aiida.orm.Dict - present only if requested.
    """
    from warnings import warn

    warn(
        "The `aiida_workflows` sub-package will be removed, please use the `wfl` sub-package "
        + "instead.",
        DeprecationWarning,
        2,
    )
    return aiida_utils._phonopy_calculate_phonons(structure, phonopy_parameters, parameters)


def get_workchain_runtime(workchain):
    """
    Calculate the total runtime of all CalcJobNodes linked to a WorkChainNode.

    Returns
    -------
    total_runtime : datetime.timedelta
    """
    from warnings import warn

    warn(
        "The `aiida_workflows` sub-package will be removed, please use the `wfl` sub-package "
        + "instead.",
        DeprecationWarning,
        2,
    )
    return aiida_utils.get_workchain_runtime(workchain)


def concatenate_workflow_results(
    workflow_results1,
    workflow_results2,
    map_result1="optimized_structure",
    map_result2="parent_node",
):
    """
    Concatenate two results pandas data frames.

    Parameters
    ----------
    workflow_results1 : pandas.DataFrame
        Pandas data frame of the first workflow.
    workflow_results2 : pandas.DataFrame
        Pandas data frame of the second workflow.
    map_result1 : str (optional)
        Result used to connect the two workflows.
    map_result2 : str (optional)
        Result used to connect the two workflows.
    Returns
    -------
    pandas.DataFrame
        New pandas data frame representing results from both workflows.
    """
    from warnings import warn

    warn(
        "The `aiida_workflows` sub-package will be removed, please use the `wfl` sub-package "
        + "instead.",
        DeprecationWarning,
        2,
    )
    return aiida_utils.concatenate_workflow_results(
        workflow_results1,
        workflow_results2,
        map_result1="optimized_structure",
        map_result2="parent_node",
    )


def get_results_cp2k_legacy_wc(aiida_group_labels):
    """
    Get results from the depreciated ElectronicProperties work chain.

    Parameters
    ----------
    aiida_group_labels : str or list
        AiiDA group label or list of labels.

    Returns
    -------
    pandas.DataFrame :
        Data frame containing the results of the workflow.
    """
    from warnings import warn

    warn(
        "The `aiida_workflows` sub-package will be removed, please use the `wfl` sub-package "
        + "instead.",
        DeprecationWarning,
        2,
    )
    return aiida_utils.get_results_cp2k_legacy_wc(aiida_group_labels)


def workflow_queue(maxrun_workflows, running_workflows_list, waiting_time=10.0):
    """
    Helper-function to control the number of workchains run simultaneously. The function is
    called in a loop after the workchain has been submitted.

    Parameters
    ----------
    maxrun_workflows : int
        Maximum number of workchains run in parallel.
    running_workflows_list : list
        List of workchain-nodes that have been started.
    waiting_time : float (optional)
        Time to wait between submissions in minutes. The default value is ``10.0``.
    """
    from warnings import warn

    warn(
        "The `aiida_workflows` sub-package will be removed, please use the `wfl` sub-package "
        + "instead.",
        DeprecationWarning,
        2,
    )
    return aiida_utils.workflow_queue(maxrun_workflows, running_workflows_list, waiting_time=10.0)


def create_aiida_node(value, node_type=None):
    """
    Create AiiDA data node from standard python variable.

    Parameters
    ----------
    value : variable
        Input variable.
    node_type : str (optional)
        AiiDA node type. The default value is ``None``.

    Returns
    -------
    aiida_node : variable
        AiiDA data node.
    """
    from warnings import warn

    warn(
        "The `aiida_workflows` sub-package will be removed, please use the `wfl` sub-package "
        + "instead.",
        DeprecationWarning,
        2,
    )
    return aiida_utils.create_aiida_node(value, node_type=None)


def obtain_value_from_aiida_node(aiida_node):
    """
    Obtain value from AiiDA data node.

    Parameters
    ----------
    aiida_node : aiida.node
        AiiDA data node.

    Returns
    -------
    value : variable
        Content of the node.
    """
    from warnings import warn

    warn(
        "The `aiida_workflows` sub-package will be removed, please use the `wfl` sub-package "
        + "instead.",
        DeprecationWarning,
        2,
    )
    return aiida_utils.obtain_value_from_aiida_node(aiida_node)
