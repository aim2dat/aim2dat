"""Common functions used by several work chains."""

# Standard library imports
import time
from datetime import timedelta

# Third party library imports
import numpy as np
import aiida.orm as aiida_orm
import aiida.tools.data.array.kpoints as aiida_kpoints
from aiida.engine import calcfunction
from aiida.plugins import DataFactory

# Internal library imports
from aim2dat.ext_interfaces.pandas import _turn_dict_into_pandas_df
from aim2dat.ext_interfaces.aiida import _create_structure_node
from aim2dat.strct.surface_utils import (
    _surface_create_slab,
    _transform_slab_to_primitive,
)
from aim2dat.strct import Structure, StructureOperations
from aim2dat.strct.ext_manipulation import add_structure_coord, rotate_structure
from aim2dat.strct.analysis.brillouin_zone_2d import _get_kpath
from aim2dat.utils.dict_tools import dict_retrieve_parameter

StructureData = DataFactory("core.structure")
XyData = DataFactory("core.array.xy")
BandsData = DataFactory("core.array.bands")


@calcfunction
def find_equivalent_sites(structure):
    """Wrap the 'find_eq_sites_via_coordination' function to be used as a calcfunction."""
    if not isinstance(structure, StructureData):
        raise ValueError(f"{type(structure)} is not supported.")
    eq_sites = aiida_orm.Dict(
        dict=StructureOperations(
            [Structure.from_aiida_structuredata(structure, label="aiida")]
        ).find_eq_sites_via_coordination(0)
    )
    output_dict = {
        "eq_sites": eq_sites,
    }
    return output_dict


@calcfunction
def add_molecule(structure, parameters):
    """Wrap the 'add_structure_position' function to be used as a calcfunction."""
    p_dict = parameters.get_dict()
    if not isinstance(structure, StructureData):
        raise ValueError(f"{type(structure)} is not supported.")
    for parameter in [
        "host_indices",
        "guest_structure",
        "bond_length",
    ]:
        if parameter not in parameters:
            raise ValueError(f"{parameter} needs to be set.")
    if hasattr(p_dict, "label_suffix"):
        label = f"{structure.label}_{p_dict.pop('label_suffix')}"
    else:
        label = structure.label
    host_indices = p_dict.pop("host_indices")
    guest_indices = p_dict.pop("guest_indices", 0)
    guest_structure = p_dict.pop("guest_structure")
    bond_length = p_dict.pop("bond_length")
    a2d_structure = Structure.from_aiida_structuredata(structure, label=label)
    a2d_structure._attributes = {}
    output_structure = _create_structure_node(
        add_structure_coord(
            structure=a2d_structure,
            host_indices=host_indices,
            guest_indices=guest_indices,
            guest_structure=guest_structure,
            bond_length=bond_length,
            dist_threshold=0.99 * bond_length,
        )
    )
    # We need to delete the 'structure_node' attribute to avoid circular references
    if len(output_structure.sites) - len(structure.sites) != 0:
        return output_structure


@calcfunction
def rotate_molecule(structure, parameters):
    """Wrap the 'rotate_structure' function to be used as a calcfunction."""
    p_dict = parameters.get_dict()
    if not isinstance(structure, StructureData):
        raise ValueError(f"{type(structure)} is not supported.")
    for parameter in [
        "angles",
        "site_indices",
    ]:
        if parameter not in parameters:
            raise ValueError(f"{parameter} needs to be set.")
    if hasattr(p_dict, "label_suffix"):
        structure.label = f"{structure.label}_{p_dict.pop('label_suffix')}"
    angles = p_dict.pop("angles")
    site_indices = p_dict.pop("site_indices")
    output_structure = _create_structure_node(
        rotate_structure(
            structure=Structure.from_aiida_structuredata(structure),
            angles=angles,
            vector=False,
            site_indices=site_indices,
        )
    )
    return output_structure


@calcfunction
def seekpath_structure_analysis(structure, parameters):
    """Wrap the seekpath function to be used as a calcfunction."""
    add_attributes = {}
    for attr_key in ["source", "source_id", "band_gap"]:
        if attr_key in structure.attributes:
            add_attributes[attr_key] = structure.get_attribute(attr_key)
    output_dict = aiida_kpoints.get_explicit_kpoints_path(structure, **parameters.get_dict())
    if len(add_attributes) != 0:
        for attr_key, attr_value in add_attributes.items():
            output_dict["conv_structure"].set_attribute(attr_key, attr_value)
            output_dict["primitive_structure"].set_attribute(attr_key, attr_value)
    return output_dict


@calcfunction
def create_surface_slab(surface, nr_layers, parameters):
    """Create surface slab from surface data."""
    p_dict = parameters.get_dict()
    label = p_dict.pop("label", "")
    periodic = p_dict.pop("periodic", False)
    vacuum = p_dict.pop("vacuum", 10.0)
    vacuum_factor = p_dict.pop("vacuum_factor", 0.0)
    symmetrize = p_dict.pop("symmetrize", True)
    use_prim_cell = p_dict.pop("return_primitive_slab", False)
    return_path_p = p_dict.pop("return_path_p", False)
    reference_distance = p_dict.pop("reference_distance", 0.015)
    symprec = p_dict.pop("symprec", 0.005)
    aperiodic_dir = surface.aperiodic_dir

    if not use_prim_cell and return_path_p:
        raise ValueError(
            "If `return_path_p` is set to True, " "`return_primitive_slab` must be set to True."
        )

    surf_dict = {
        "repeating_structure": surface.repeating_structure,
        "bottom_structure": surface.bottom_terminating_structure,
        "top_structure": surface.top_terminating_structure,
        "top_structure_nsym": surface.top_terminating_structure_nsym,
    }
    slab = _surface_create_slab(
        surf_dict, nr_layers.value, periodic, vacuum, vacuum_factor, symmetrize
    )
    outputs = {}
    if use_prim_cell:
        slab, lg = _transform_slab_to_primitive(slab, symprec, -1, 0, aperiodic_dir=2)
        if return_path_p:
            path_p = _get_kpath(slab["cell"], aperiodic_dir, lg, reference_distance, symprec)
            path_p["layergroup_number"] = lg
            outputs["parameters"] = aiida_orm.Dict(dict=path_p)
    pbc = [True, True, True]
    if not periodic:
        pbc[aperiodic_dir] = False
    slab["pbc"] = pbc
    slab["label"] = label
    outputs["slab"] = _create_structure_node(Structure(**slab))
    return outputs


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
    from aim2dat.ext_interfaces.phonopy import _build_phonopy

    phonopy_dict = phonopy_parameters.get_dict()
    supercell_matrix = phonopy_dict.pop("supercell_matrix")
    primitive_matrix = phonopy_dict.pop("primitive_matrix", "auto")
    symprec = phonopy_dict.pop("symprec", 1e-5)
    calculator = phonopy_dict.pop("calculator", "cp2k")

    p_dict = parameters.get_dict()
    displacement = p_dict.pop("displacement", 0.01)

    phonopy_atoms = Structure.from_aiida_structuredata(structure).to_phonopy_atoms()
    phonon = _build_phonopy(
        unitcell=phonopy_atoms,
        supercell_matrix=supercell_matrix,
        primitive_matrix=primitive_matrix,
        symprec=symprec,
        calculator=calculator,
    )
    phonon.generate_displacements(distance=displacement)

    supercells = phonon.supercells_with_displacements

    outputs = {"displacement_dataset": aiida_orm.Dict(dict=phonon.dataset)}
    for i, supercell in enumerate(supercells):
        outputs[f"supercell_{i:04d}"] = Structure.from_phonopy_atoms(
            supercell
        ).to_aiida_structuredata()
    return outputs


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
    from aim2dat.ext_interfaces.phonopy import (
        _build_phonopy,
        _extract_band_structure,
        _extract_total_dos,
        _extract_thermal_properties,
    )

    phonopy_dict = phonopy_parameters.get_dict()
    supercell_matrix = phonopy_dict.pop("supercell_matrix")
    primitive_matrix = phonopy_dict.pop("primitive_matrix", "auto")
    symprec = phonopy_dict.pop("symprec", 1e-5)
    calculator = phonopy_dict.pop("calculator", "cp2k")
    phonopy_atoms = Structure.from_aiida_structuredata(structure).to_phonopy_atoms()

    p_dict = parameters.get_dict()
    displacement_dataset = p_dict.pop("displacement_dataset")
    force_sets = p_dict.pop("force_sets")

    band_path = p_dict.pop("band_path")
    band_labels = p_dict.pop("band_labels")
    band_npoints = p_dict.pop("band_npoints", 101)
    with_eigenvectors = p_dict.pop("with_eigenvectors", False)

    dos_mesh = p_dict.pop("dos_mesh", [20, 20, 20])

    thermal_properties = p_dict.pop("thermal_properties", False)
    t_min, t_max, t_step = p_dict.pop("temp_range", [0.0, 1000.0, 10.0])

    phonon = _build_phonopy(
        unitcell=phonopy_atoms,
        supercell_matrix=supercell_matrix,
        primitive_matrix=primitive_matrix,
        symprec=symprec,
        calculator=calculator,
    )
    phonon.dataset = displacement_dataset
    phonon.forces = force_sets
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()

    # --- v2 TODO: non-analytical correction (NAC) -------------------------- #
    # For polar frameworks, set the Born effective charges + dielectric tensor
    # here (computed by CP2K) to capture LO-TO splitting near Gamma:
    #     phonon.nac_params = {"born": ..., "dielectric": ..., "factor": ...}
    # The input spec deliberately reserves a `nac_parameters` slot so adding
    # this in v2 does not break the API.
    # ----------------------------------------------------------------------- #

    outputs = {}

    # Band structure -> BandsData
    band_dict, _ = _extract_band_structure(
        load_parameters={},
        path=band_path,
        path_labels=band_labels,
        npoints=band_npoints,
        with_eigenvectors=with_eigenvectors,
        pre_load=phonon,
    )

    # Parse bands output
    output_bands = {
        "kpoints": [],
        "bands": [],
        "path_labels": [],
    }

    counter_points = 0
    counter_seg = 0
    path_start = 0
    for cont_segment in band_path:
        path_end = path_start + len(cont_segment) - 1
        for qpoints, band_seg in zip(
            band_dict["qpoints"][path_start:path_end],
            band_dict["frequencies"][path_start:path_end],
        ):
            output_bands["path_labels"].append((counter_points, band_labels[counter_seg]))
            counter_seg += 1
            counter_points += band_npoints - 1
            output_bands["kpoints"] += [
                [float(pt[0]), float(pt[1]), float(pt[2])] for pt in qpoints
            ]
            output_bands["bands"] += [[float(ev) for ev in pt] for pt in band_seg]
            output_bands["path_labels"].append((counter_points, band_labels[counter_seg]))
            counter_points += 1
        path_start = path_end
        counter_seg += 1

    bands = BandsData()
    bands.set_cell_from_structure(structuredata=structure)
    bands.set_kpoints(np.array(output_bands["kpoints"]))
    bands.set_bands(np.array(output_bands["bands"]), units="THz")
    bands.labels = output_bands["path_labels"]
    outputs["band_structure"] = bands

    # Total DOS -> XyData
    dos_dict = _extract_total_dos(
        load_parameters={},
        mesh=dos_mesh,
        pre_load=phonon,
    )
    dos = XyData()
    dos.set_x(np.array(dos_dict["frequency_points"]), "frequency", "THz")
    dos.set_y(np.array(dos_dict["total_dos"]), "total densities", "states/THz")
    outputs["total_dos"] = dos

    # Optional thermal properties
    if thermal_properties:
        thermal_dict = _extract_thermal_properties(
            load_parameters={},
            mesh=dos_mesh,
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            pre_load=phonon,
        )
        thermal = XyData()
        thermal.set_x(np.array(thermal_dict["temperatures"]), "temperatures", "K")
        y_arrays = [
            np.array(thermal_dict["free_energy"]),
            np.array(thermal_dict["entropy"]),
            np.array(thermal_dict["heat_capacity"]),
        ]
        y_names = ["helmholtz free energies", "entropies", "heat capacities"]
        y_units = ["kJmol", "J/K/mol", "J/K/mol"]
        thermal.set_y(y_arrays, y_names, y_units)
        outputs["thermal_properties"] = thermal

    return outputs


def get_workchain_runtime(workchain):
    """
    Calculate the total runtime of all CalcJobNodes linked to a WorkChainNode.

    Returns
    -------
    total_runtime : datetime.timedelta
    """
    calcjobs = workchain.base.links.get_outgoing(node_class=aiida_orm.CalcJobNode).all_nodes()
    runtimes = []
    for calc_j in calcjobs:
        if calc_j.computer.scheduler_type != "core.slurm":
            continue
        scheduler_std_out = calc_j.get_detailed_job_info()["stdout"].lower().split("\n")
        elapsed_idx = scheduler_std_out[0].split("|").index("elapsed")
        try:
            # Sometimes, somehow the last row is empty, thus jump to second las row
            if scheduler_std_out[-1]:
                time = scheduler_std_out[-1].split("|")[elapsed_idx]
            else:
                time = scheduler_std_out[-2].split("|")[elapsed_idx]
            h, m, s = map(int, time.split(":"))
            runtime = (h * 60 + m) * 60 + s
        except IndexError:
            runtime = float("inf")
        runtimes.append(runtime)
    total_runtime = timedelta(seconds=round(sum(runtimes)))
    return total_runtime


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
    new_wf_results_dict = {}
    for column in workflow_results1.columns:
        if (
            column not in workflow_results2.columns
            or column == map_result1
            or column == map_result2
        ):
            # print(dir(workflow_results1[column]))
            new_wf_results_dict[column] = workflow_results1[column].values

    new_columns = {column: [] for column in workflow_results2.columns if column != map_result2}
    for _, row in workflow_results1.iterrows():
        map_value = row[map_result1]
        matches = workflow_results2.loc[(workflow_results2[map_result2] == map_value)]
        matches.reset_index(inplace=True)
        if len(matches) == 1:
            for new_col, new_col_val in new_columns.items():  # matches.columns:
                new_col_val.append(matches[new_col][0])
        elif len(matches) > 1:
            for new_col_val in new_columns.values():
                new_col_val.append(None)
            raise ValueError(f"Mapping is not unique found {map_value} {len(matches)} times.")
        else:
            for new_col_val in new_columns.values():
                new_col_val.append(None)
    for new_col, new_col_val in new_columns.items():
        new_wf_results_dict[new_col] = new_col_val
    return _turn_dict_into_pandas_df(new_wf_results_dict)


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
    if not isinstance(aiida_group_labels, list):
        aiida_group_labels = [aiida_group_labels]
    pd_series_dict = {
        "parent_node": [],
        "wc_node": [],
        "exit_status": [],
        "primitive_structure": [],
        "conventional_structure": [],
        "scf_method_level": [],
        "scf_parameter_level": [],
        "scf_smearing_level": [],
        "optimized_structure": [],
        "total_energy (Hartree)": [],
        "space_group": [],
        "band_structure": [],
        "pdos": [],
    }
    outputs = {
        "seekpath_strct": [
            ("primitive_structure", "primitive_structure", None),
            ("conventional_structure", "conv_structure", None),
        ],
        "FindSCFParametersWorkChain": [
            ("scf_method_level", "scf_parameters", ["method_level"]),
            ("scf_parameter_level", "scf_parameters", ["parameter_level"]),
            ("scf_smearing_level", "scf_parameters", ["smearing_level"]),
        ],
        "CellOptWorkChain": [
            ("optimized_structure", "output_structure", None),
            ("total_energy (Hartree)", "output_parameters", ["energy"]),
            ("space_group", "output_parameters", ["spgr_info", "sg_number"]),
        ],
        "BandStructureWorkChain": [
            ("band_structure", "output_bands", None),
        ],
        "PDOSWorkChain": [
            ("pdos", "output_pdos", None),
        ],
    }
    wc_nodes = []
    for group_label in aiida_group_labels:
        queryb = aiida_orm.querybuilder.QueryBuilder()
        queryb.append(aiida_orm.Group, filters={"label": group_label}, tag="group")
        queryb.append(aiida_orm.WorkChainNode, with_group="group")
        wc_nodes += queryb.all(flat=True)

    for wc_node in wc_nodes:
        if wc_node.process_label != "CrystalElectronicPropertiesWorkChain":
            continue
        pd_series_dict["parent_node"].append(wc_node.inputs["structure"].pk)
        pd_series_dict["exit_status"].append(wc_node.exit_status)
        pd_series_dict["wc_node"].append(wc_node.pk)
        called_nodes = wc_node.called
        called_labels = [cn.process_label for cn in called_nodes]
        for output_proc, output_details in outputs.items():
            if (
                output_proc in called_labels
                and called_nodes[called_labels.index(output_proc)].exit_status == 0
            ):
                cn_outputs = called_nodes[called_labels.index(output_proc)].outputs
                for df_label, output_label, dict_tree in output_details:
                    if dict_tree is None:
                        pd_series_dict[df_label].append(cn_outputs[output_label].pk)
                    else:
                        output_dict = cn_outputs[output_label].get_dict()
                        pd_series_dict[df_label].append(
                            dict_retrieve_parameter(output_dict, dict_tree)
                        )
            else:
                for df_label, _, _ in output_details:
                    pd_series_dict[df_label].append(None)
    return _turn_dict_into_pandas_df(pd_series_dict)


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
    while len(running_workflows_list) >= maxrun_workflows:
        print(f"waiting {waiting_time} min...")
        time.sleep(waiting_time * 60)
        for workflow in running_workflows_list.copy():
            if workflow.is_finished or workflow.is_excepted:
                running_workflows_list.remove(workflow)


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
    # TODO include more node types, use DataFactory?
    check_node_type = False
    if node_type is None:
        check_node_type = True

    if node_type == "bool" or (check_node_type and isinstance(value, bool)):
        aiida_node = aiida_orm.Bool(value)
    elif node_type == "int" or (check_node_type and isinstance(value, int)):
        aiida_node = aiida_orm.Int(value)
    elif node_type == "float" or (check_node_type and isinstance(value, float)):
        aiida_node = aiida_orm.Float(value)
    elif node_type == "str" or (check_node_type and isinstance(value, str)):
        aiida_node = aiida_orm.Str(value)
    elif node_type == "list" or (check_node_type and isinstance(value, list)):
        aiida_node = aiida_orm.List(list=value)
    elif node_type == "dict" or (check_node_type and isinstance(value, dict)):
        aiida_node = aiida_orm.Dict(dict=value)
    else:
        raise ValueError(f"{type(value)} is not supported.")
    return aiida_node


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
    value_type = [aiida_orm.Str, aiida_orm.Float, aiida_orm.Int, aiida_orm.Bool]
    value = None
    if type(aiida_node) in value_type:
        value = aiida_node.value
    elif isinstance(aiida_node, aiida_orm.Dict):
        value = aiida_node.get_dict()
    elif isinstance(aiida_node, aiida_orm.List):
        value = aiida_node.get_list()
    else:
        raise ValueError(f"{type(aiida_node)} is not supported.")
    return value
