"""
WorkflowBuilder and auxiliary functions.
"""

# Standard library imports
import os
import copy
import abc
from datetime import timedelta

# Third party library imports
from aiida.plugins import DataFactory, WorkflowFactory, CalculationFactory
from aiida.engine import run_get_node, submit
import aiida.tools.visualization as aiida_vis
from aiida.common.exceptions import MissingEntryPointError
import aiida.orm as aiida_orm
from aiida.orm import QueryBuilder

# Internal library imports
from aim2dat.aiida_workflows._workflow_builder_utils import (
    _load_protocol,
    _wf_states_color_map,
)
from aim2dat.io import read_yaml_file, write_yaml_file
from aim2dat.aiida_workflows.utils import (
    create_aiida_node,
    obtain_value_from_aiida_node,
    get_workchain_runtime,
)
from aim2dat.ext_interfaces.import_opt_dependencies import _check_package_dependencies
from aim2dat.ext_interfaces.aiida import _load_data_node, _create_group
from aim2dat.ext_interfaces.pandas import _turn_dict_into_pandas_df, _apply_color_map
from aim2dat.utils.dict_tools import dict_retrieve_parameter


AiidaCode = DataFactory("core.code")


class WorkflowProtocolError(Exception):
    """Error for incomplete or inconsistent workflow protocols."""

    pass


def _validate_parent_node(aiida_node, protocol):
    """Check if parent node is of the right type."""
    if protocol is None:
        raise ValueError("`protocol` needs to be set.")
    aiida_node = _load_data_node(aiida_node)
    if not isinstance(aiida_node, DataFactory(protocol["parent_node_type"])):
        raise TypeError(
            f"Parent node needs to be of type: `{protocol['parent_node_type']}` for this "
            "protocol."
        )
    return aiida_node


def _validate_input_details(input_details):
    """Validate input details and set default values."""
    input_details["namespace"] = input_details.get("namespace", False)
    if input_details["namespace"]:
        input_details["aiida_node"] = False
    if input_details["aiida_node"] and "value" in input_details:
        input_details["value"] = create_aiida_node(input_details["value"])
    else:
        input_details["value"] = None
    input_details["unstored"] = input_details.get("unstored", False)
    input_details["compare"] = input_details.get("compare", True)
    input_details["optional"] = input_details.get("optional", False)


def _validate_task_details(task_label, task_details):
    if "process" not in task_details:
        raise WorkflowProtocolError(f"No process specified for task '{task_label}'.")
    task_details["dependencies"] = task_details.get("dependencies", {})
    task_details["blacklist_inputs"] = task_details.get("blacklist_inputs", [])
    return task_details


def _add_task_input(tasks, task_labels, input_label):
    """Add input to task and check if input port is already set or blacklisted."""
    for task_label in task_labels:
        if task_label not in tasks:
            raise WorkflowProtocolError(f"Task '{task_label}' does not exist.")
        task_details = tasks[task_label]
        if task_details.get("parent_node", "") == input_label:
            raise WorkflowProtocolError(
                f"Input '{input_label}' clashes with parent node at task '{task_label}'."
            )
        input_list = task_details.get("inputs", [])
        if input_label in input_list:
            raise WorkflowProtocolError(
                f"Input '{input_label}' defined twice for task '{task_label}'."
            )
        if input_label in task_details.get("blacklist_inputs", []):
            raise WorkflowProtocolError(
                f"Input '{input_label}' is on the blacklist of task '{task_label}'."
            )
        input_list.append(input_label)
        task_details["inputs"] = input_list


class _BaseWorkflowBuilder(abc.ABC):
    def __init__(self):
        """Initialize class."""
        self._protocol = None
        self._tasks = None
        self._group = None
        self._user_input = {}
        self._general_input = {}
        self._individual_input = {}
        self._use_uuid = False

    @property
    def use_uuid(self):
        """
        bool : Whether to use the uuid (str) to represent AiiDA nodes instead of the primary key
        (int).
        """
        return self._use_uuid

    @use_uuid.setter
    def use_uuid(self, value):
        self._use_uuid = value
        if hasattr(self, "_wf_builders"):
            for wf_builder in self._wf_builders:
                wf_builder.use_uuid = value

    @property
    def protocol(self):
        """
        Protocol used for the workflow.
        """
        return self._protocol

    @protocol.setter
    def protocol(self, value):
        protocol = _load_protocol(value, os.path.dirname(__file__) + "/protocols/")
        self._protocol = protocol
        protocol = copy.deepcopy(protocol)
        if "dependencies" in protocol:
            _check_package_dependencies(protocol["dependencies"])
        self._parent_node_type = protocol["parent_node_type"]
        self._user_input = {}
        self._general_input = {}
        self._individual_input = {}
        self._tasks = {}

        for task_label, task_details in protocol["tasks"].items():
            self._tasks[task_label] = _validate_task_details(task_label, task_details)
            self._individual_input[task_label] = {}
        for task_label, input_label in protocol["parent_node_input"].items():
            self._tasks[task_label]["parent_node"] = input_label

        for input_cat in ["user_input", "general_input"]:
            cat_attr = getattr(self, "_" + input_cat)
            if input_cat in protocol:
                for input_p, input_details in protocol[input_cat].items():
                    _validate_input_details(input_details)
                    input_p_sp = input_p.split("->")
                    if len(input_p_sp) > 1:
                        input_details["user_input"] = True if input_cat == "user_input" else False
                        self._individual_input[input_p_sp[1]][input_p_sp[0]] = input_details
                        task_labels = [input_p_sp[1]]
                        input_label = input_p_sp[0]
                    else:
                        cat_attr[input_p] = input_details
                        task_labels = input_details.pop("tasks")
                        input_label = input_p
                    _add_task_input(self._tasks, task_labels, input_label)
        self._set_results()

    @property
    def tasks(self):
        """
        Return all tasks of the workflow.
        """
        tasks = {}
        if self._protocol is None:
            print("No protocol loaded.")
        else:
            for task, task_details in self._tasks.items():
                tasks[task] = {
                    "dependencies": task_details["dependencies"],
                    "process": task_details["process"],
                }
        return tasks

    @property
    def user_input(self):
        """
        Input parameters set by the user.
        """
        user_input = {}
        for input_p, input_details in self._user_input.items():
            user_input[input_p] = input_details["value"]
        for task, task_details in self._individual_input.items():
            for input_p, input_details in task_details.items():
                if input_details["user_input"]:
                    user_input[input_p + "->" + task] = input_details["value"]
        return user_input

    def to_file(self, file_path="workflow.yaml"):
        def _transform_namespace(value, new_value):
            for key, val in value.items():
                if isinstance(val, dict):
                    new_value[key] = {}
                    _transform_namespace(val, new_value[key])
                else:
                    new_value[key] = val.uuid

        def _add_aiida_input_node(input_dict, input_label, input_details):
            if input_details["aiida_node"] and input_details["value"].is_stored:
                input_dict[input_label] = input_details["value"].uuid

        def _add_user_input(input_dict, input_dict_nodes, input_label, input_details):
            if input_details["value"] is not None and not isinstance(
                input_details["value"], AiidaCode
            ):
                if input_details["namespace"]:
                    value = {}
                    _transform_namespace(input_details["value"], value)
                    input_dict_nodes[input_label] = value
                elif input_details["aiida_node"]:
                    if input_details["value"].is_stored:
                        input_dict_nodes[input_label] = input_details["value"].uuid
                    else:
                        input_dict[input_label] = obtain_value_from_aiida_node(
                            input_details["value"]
                        )
                else:
                    input_dict[input_label] = input_details["value"]

        content = {}
        if self._protocol is not None:
            content["protocol"] = self._protocol
            content["general_input"] = {}
            content["user_input_nodes"] = {}
            content["user_input"] = {}
            for input_label, input_details in self._general_input.items():
                _add_aiida_input_node(content["general_input"], input_label, input_details)
            for input_label, input_details in self._user_input.items():
                _add_user_input(
                    content["user_input"], content["user_input_nodes"], input_label, input_details
                )
            for task_label, inputs in self._individual_input.items():
                for input_label, input_details in inputs.items():
                    if input_details["user_input"]:
                        _add_user_input(
                            content["user_input"],
                            content["user_input_nodes"],
                            input_label + "->" + task_label,
                            input_details,
                        )
                    else:
                        _add_aiida_input_node(
                            content["general_input"],
                            input_label + "->" + task_label,
                            input_details,
                        )
        self._extract_parent_nodes(content)
        write_yaml_file(file_path, content)

    @classmethod
    def from_file(cls, file_path="workflow.yaml"):
        def _transform_name_space(value):
            for key, val in value.items():
                if isinstance(val, dict):
                    _transform_name_space(val)
                else:
                    value[key] = _load_data_node(val)

        content = read_yaml_file(file_path)
        wf_builder = cls()
        wf_builder.protocol = content["protocol"]
        if "parent_node" in content:
            if hasattr(wf_builder, "_wf_builders"):
                wf_builder.add_parent_node(content["parent_node"])
            else:
                wf_builder.parent_node = content["parent_node"]
        elif "parent_nodes" in content:
            if not hasattr(wf_builder, "_wf_builders"):
                raise ValueError(
                    "Multiple `parent_nodes` found, please use `MultipleWorkflowBuilder` instead."
                )
            for node in content["parent_nodes"]:
                wf_builder.add_parent_node(node)
        for input_port, value in content["user_input"].items():
            wf_builder.set_user_input(input_port, value)
        for input_port, value in content["user_input_nodes"].items():
            if content["protocol"]["user_input"][input_port].get("namespace", False):
                _transform_name_space(value)
            wf_builder.set_user_input(input_port, _load_data_node(value))
        for input_p, value in content["general_input"].items():
            input_p_sp = input_p.split("->")
            if len(input_p_sp) > 1:
                wf_builder._individual_input[input_p_sp[1]][input_p_sp[0]]["value"] = (
                    _load_data_node(value)
                )
            else:
                wf_builder._general_input[input_p]["value"] = _load_data_node(value)
        return wf_builder

    def set_user_input(self, input_port, value):
        """
        Set user input parameters of the workflow.

        Parameters
        ----------
        input_port : str
            Input port of the parameter.
        value :
            Value.
        """
        if self._protocol is None:
            raise ValueError("`protocol` needs to be set.")
        if input_port not in self._protocol["user_input"]:
            raise ValueError(f"Parameter `{input_port}` cannot be set manually.")
        input_p_sp = input_port.split("->")
        if len(input_p_sp) > 1:
            input_port = input_p_sp[0]
            input_details = self._individual_input[input_p_sp[1]][input_p_sp[0]]
        else:
            input_details = self._user_input[input_port]
        if input_details["aiida_node"] and not hasattr(value, "uuid"):
            value = create_aiida_node(value)
        if input_details["validation"] is not None:
            self._validate_work_chain_input(input_port, value, input_details["validation"])
        input_details["value"] = value

    @staticmethod
    def _validate_work_chain_input(input_port, value, validation):
        """Validate work chain input parameter."""
        work_chain = WorkflowFactory(validation)
        work_chain_builder = work_chain.get_builder()
        key_list = input_port.split(".")
        try:
            builder_tree = work_chain_builder
            for key in key_list[:-1]:
                builder_tree = builder_tree[key]
            builder_tree[key_list[-1]] = value
        except ValueError:
            raise ValueError(f"{value} could not be set to {input_port}.")

    def _set_aiida_group(self, value):
        if isinstance(value, str):
            self._group = _create_group(value, "")
        elif isinstance(value, aiida_orm.Group):
            self._group = value
        else:
            raise ValueError("`aiida_group` needs to be of type str or aiida.orm.Group.")

    def _get_aiida_node_identifier(self, aiida_process):
        if self.use_uuid:
            return aiida_process.uuid
        else:
            return aiida_process.pk

    def _set_results(self):
        pass

    @abc.abstractmethod
    def _extract_parent_nodes(self):
        pass


class WorkflowBuilder(_BaseWorkflowBuilder):
    """
    Class to manage complex AiiDA workflows.
    """

    def __init__(
        self,
        parent_node=None,
        protocol=None,
        graph_attributes={"graph_attr": {"size": "10!,6"}},
        use_uuid=False,
    ):
        """Initialize class."""
        super().__init__()
        self._parent_node = None
        self._completed_tasks = {}
        self._running_tasks = {}
        self._failed_tasks = {}
        self._results = {}
        self._result_dict = {}

        if parent_node is not None:
            self.parent_node = parent_node
        if protocol is not None:
            self.protocol = protocol
        self.graph_attributes = graph_attributes
        self.use_uuid = use_uuid

    @property
    def parent_node(self):
        """
        Parent AiiDA node.
        """
        return self._parent_node

    @parent_node.setter
    def parent_node(self, value):
        self._parent_node = _validate_parent_node(value, self._protocol)

    @property
    def aiida_group(self):
        """
        str or aiida.orm.Group : Only consider processes in the AiiDA group.
        """
        return self._group

    @aiida_group.setter
    def aiida_group(self, value):
        self._completed_tasks = {}
        self._running_tasks = {}
        self._failed_tasks = {}
        self._set_aiida_group(value)

    @property
    def completed_tasks(self):
        """
        Completed tasks.
        """
        self.determine_workflow_state()
        return {task: proc_details[0] for task, proc_details in self._completed_tasks.items()}

    @property
    def running_tasks(self):
        """
        Tasks which have not yet finished.
        """
        self.determine_workflow_state()
        return self._running_tasks

    @property
    def failed_tasks(self):
        """
        Failed tasks.
        """
        self.determine_workflow_state()
        return self._failed_tasks

    @property
    def results(self):
        """
        Results of the workflow.
        """
        self.determine_workflow_state()
        return self._results

    def adopt_input_nodes_from_workflow(self, parent_node):
        """
        Import input parameters for the workflow based on another parent node.

        Parameters
        ----------
        parent_node : aiida.node
            Parent AiiDA node.
        """
        for user_inp_key, user_inp in self._user_input.items():
            if "value" not in user_inp and user_inp["compare"] and not user_inp["optional"]:
                raise ValueError(f"User input `{user_inp_key}` needs to be set.")
        self._completed_tasks = {}
        self._running_tasks = {}
        self._failed_tasks = {}
        self._determine_workflow_state(parent_node)
        self._completed_tasks = {}
        self._running_tasks = {}
        self._failed_tasks = {}

    def set_user_input(self, input_port, value):
        """
        Set user input parameters of the workflow.

        Parameters
        ----------
        input_port : str
            Input port of the parameter.
        value :
            Value.
        """
        self._completed_tasks = {}
        self._running_tasks = {}
        self._failed_tasks = {}
        super().set_user_input(input_port, value)

    def determine_workflow_state(self):
        """
        Determine which tasks of the workflow have been completed and which ones can be started
        next.
        """
        if self._parent_node is None:
            raise ValueError("`parent_node` needs to be set.")
        for user_inp_key, user_inp in self._user_input.items():
            if (
                "dependency" in user_inp
                and self._user_input[user_inp["dependency"]]["value"] is not None
            ):
                continue
            if user_inp["value"] is None and user_inp["compare"] and not user_inp["optional"]:
                raise ValueError(f"User input `{user_inp_key}` needs to be set.")
        for task, task_input in self._individual_input.items():
            for ind_key, ind_inp in task_input.items():
                if (
                    ind_inp["user_input"]
                    and ind_inp["value"] is None
                    and ind_inp["compare"]
                    and not ind_inp["optional"]
                ):
                    raise ValueError(f"User input `{ind_key}->{task}` needs to be set.")
        return self._determine_workflow_state(self._parent_node)

    def generate_provenance_graph(self):
        """
        Generate provenance graph of the workflow.

        Returns
        -------
        : aiida.tools.visualization.graph.Graph
            Provenance graph.
        """
        self.determine_workflow_state()
        graph = aiida_vis.Graph(**self.graph_attributes)
        for proc_details in self._completed_tasks.values():
            graph.add_incoming(proc_details[0].uuid, annotate_links="both")
            graph.add_outgoing(proc_details[0].uuid, annotate_links="both")
        return graph

    def run_task(self, task):
        """
        Run the underlying AiiDA process of the task.

        Parameters
        ----------
        task : str
            Workflow task.

        Returns
        -------
        proc_node : aiida.node
            AiiDA process node.
        result : aiida.node
            Results of the AiiDA process.
        """
        builder = self.generate_inputs(task)
        if isinstance(builder, dict):
            result, proc_node = run_get_node(**builder)
        else:
            result, proc_node = run_get_node(builder)
        if self._group is not None:
            self._group.add_nodes(proc_node)
        return proc_node, result

    def submit_task(self, task):
        """
        Submit the underlying AiiDA process of the task.

        Parameters
        ----------
        task : str
            Workflow task.

        Returns
        -------
        proc_node : aiida.node
            AiiDA process node.
        """
        builder = self.generate_inputs(task)
        if isinstance(builder, dict):
            proc_node = submit(**builder)
        else:
            proc_node = submit(builder)
        if self._group is not None:
            self._group.add_nodes(proc_node)
        return proc_node

    def generate_inputs(self, task):
        """
        Generate a builder for an AiiDA work chain or calculation job.

        Parameters
        ----------
        task : str
            Workflow task.

        Returns
        -------
        builder : aiida.work_chain or dict
            If the underlying AiiDA process is a work chain the builder of the work chain is
            returnedBuilder, otherwise a dictionary of the input parameters and the process is
            returned.
        """

        def check_input_parameter(task, process_builder, input_port, input_details):
            """Set input parameter for an AiiDA process."""
            if not self._check_input_parameter_dependency(task, input_details):
                return False
            elif input_details.get("value") is None:
                if input_details["optional"]:
                    return False
                else:
                    if input_port in self._individual_input[task]:
                        input_port += "->" + task
                    raise ValueError(f"Input parameter `{input_port}` needs to be set.")
            else:
                return True

        if task not in self._tasks.keys():
            raise ValueError(f"`{task}` not part of the workflow.")
        tasks = self.determine_workflow_state()
        if task in tasks["completed_tasks"]:
            raise ValueError(f"`{task}` already completed.")
        elif task not in tasks["next_possible_tasks"]:
            raise ValueError(f"Dependencies not met for `{task}`.")

        task_details = self._tasks[task]
        process = self._load_process_class(task_details["process"])
        try:
            process_builder = process.get_builder()
        except AttributeError:
            process_builder = {"process": process}

        set_inputs = list(self._individual_input[task].keys())

        # Set parent node:
        if "parent_node" in task_details:
            self._set_input_parameter(
                process_builder, task_details["parent_node"], self._parent_node
            )
            set_inputs.append(task_details["parent_node"])

        # Set task specific parameters:
        for input_port, input_details in self._individual_input[task].items():
            if check_input_parameter(task, process_builder, input_port, input_details):
                self._set_input_parameter(process_builder, input_port, input_details["value"])

        # Set remaining parameters:
        if "inputs" in task_details:
            for input_port in task_details["inputs"]:
                # individual_input has priority over general_input:
                if input_port in set_inputs:
                    continue
                if input_port in self._user_input:
                    input_details = self._user_input[input_port]
                elif input_port in self._general_input:
                    input_details = self._general_input[input_port]
                else:
                    raise WorkflowProtocolError(
                        f"Workflow protocol is incomplete, {input_port} "
                        f"of task {task} is not covered."
                    )
                if check_input_parameter(task, process_builder, input_port, input_details):
                    self._set_input_parameter(process_builder, input_port, input_details["value"])

        # Set dependencies from previous tasks:
        if "dependencies" in task_details:
            for dep_task, dep_inputs in task_details["dependencies"].items():
                dep_outputs = self._completed_tasks[dep_task][2]
                for dep_input in dep_inputs:
                    self._set_input_parameter(
                        process_builder, dep_input[1], dep_outputs[dep_input[0]]
                    )
        return process_builder

    def _set_results(self):
        """Set results from the workflow protocol."""
        self._result_dict = {task: [] for task in self._protocol["tasks"]}
        if "results" in self._protocol and self._protocol["results"] is not None:
            for res_label, res_details in self._protocol["results"].items():
                self._result_dict[res_details["task"]].append(res_label)

    def _determine_workflow_state(self, parent_node):
        """
        Check the current state of the workflow.
        """
        if self._protocol is None:
            raise ValueError("`protocol` needs to be set.")
        self._running_tasks = {}
        tasks = {"next_possible_tasks": []}
        task_list = [task for task in self._tasks.keys() if task not in self._completed_tasks]
        candidate_procs = {task: [] for task in task_list}
        for task, proc in self._completed_tasks.items():
            dependencies = self._tasks[task]["dependencies"]
            dep_procs = {}
            for dep in dependencies:
                dep_procs[dep] = self._completed_tasks[dep]
            candidate_procs[task] = [(proc[0], proc[1], proc[2], dep_procs)]

        # Check initial tasks:
        tasks_to_delete = []
        for task in task_list:
            task_details = self._tasks[task]
            if len(task_details["dependencies"]) == 0:
                self._check_initial_task(task, candidate_procs, parent_node)
                tasks_to_delete.append(task)
        for task in tasks_to_delete:
            task_list.remove(task)

        # Check follow-up tasks:
        prev_nr_unfinished_tasks = len(task_list) + 1
        while len(task_list) != prev_nr_unfinished_tasks:
            prev_nr_unfinished_tasks = len(task_list)
            tasks_to_delete = []
            for task in task_list:
                dependencies = self._tasks[task]["dependencies"]
                if all([len(candidate_procs[dep]) > 0 for dep in dependencies]):
                    if self._check_task_with_dependencies(task, candidate_procs, parent_node):
                        tasks_to_delete.append(task)
            for task in tasks_to_delete:
                if task in task_list:
                    task_list.remove(task)

        # Update completed tasks:
        for task, candidates in candidate_procs.items():
            if len(candidates) > 0:
                nr_deps = len(candidates[0][3])
                mtime = candidates[0][0].mtime
                for candidate in candidates:
                    if len(candidate[3]) > nr_deps or candidate[0].mtime >= mtime:
                        if candidate[2] is not None:
                            self._completed_tasks[task] = candidate[0:3]
                        self._update_input_nodes(task, candidate[1])
                        for dep_task, dep_process in candidate[3].items():
                            self._completed_tasks[dep_task] = dep_process

        for task in self._completed_tasks:
            for res_label in self._result_dict[task]:
                self._retrieve_result(res_label, self._completed_tasks[task])
        # Include completed, running and failed tasks:
        # Running or failed processes are not included if there is already a completed or running
        # process of the same task.. Maybe not such a good idea?
        tasks["completed_tasks"] = [task for task in self._completed_tasks.keys()]
        tasks["running_tasks"] = [
            task for task in self._running_tasks.keys() if task not in self._completed_tasks
        ]
        tasks["failed_tasks"] = [
            task
            for task in self._failed_tasks.keys()
            if task not in (self._completed_tasks or self._running_tasks)
        ]

        # Check next tasks:
        for task in self._tasks:
            dependencies = self._tasks[task]["dependencies"]
            if all(
                [dep in tasks["completed_tasks"] for dep in dependencies.keys()]
            ) and task not in (tasks["running_tasks"] + tasks["completed_tasks"]):
                tasks["next_possible_tasks"].append(task)

        return tasks

    def _check_initial_task(self, task, candidate_procs, parent_node):
        """Find processes for a task without dependencies."""
        if hasattr(parent_node, "uuid"):
            task_details = self._tasks[task]
            proc_cls = self._load_process_class(task_details["process"])
            try:
                proc_cls = proc_cls.node_class
            except AttributeError:
                pass
            p_node_cls = type(parent_node)
            qb = QueryBuilder()
            qb.append(
                cls=p_node_cls,
                filters={"id": {"==": parent_node.pk}},
                tag="p_node",
                edge_filters={"label": task_details["parent_node"]},
            )
            qb.append(cls=proc_cls, with_incoming="p_node")
            for proc in qb.all(flat=True):
                if self._group is not None and proc not in self._group.nodes:
                    continue
                if task_details["process"] != proc.process_type.split(":")[-1]:
                    continue
                inputs = self._load_process_port(proc, "incoming")
                # outputs = self._load_process_port(proc, "outgoing")
                if self._check_process_inputs(task, inputs, parent_node):
                    if proc.is_finished_ok:
                        outputs = self._load_process_port(proc, "outgoing")
                        candidate_procs[task].append((proc, inputs, outputs, {}))
                    elif proc.is_failed or proc.is_killed or proc.is_excepted:
                        self._failed_tasks[task] = proc
                    elif not proc.is_terminated:
                        self._running_tasks[task] = proc
                        candidate_procs[task].append((proc, inputs, None, {}))

    def _check_task_with_dependencies(self, task, candidate_procs, parent_node):
        """Find processes for a task with dependencies."""
        found_candidate = False
        task_details = self._tasks[task]
        proc_cls = self._load_process_class(task_details["process"])
        try:
            proc_cls = proc_cls.node_class
        except AttributeError:
            pass

        # Check initial dependency
        process_pool = []
        dep_tasks = list(task_details["dependencies"].keys())
        dep_task = dep_tasks[0]
        deps = task_details["dependencies"][dep_task]
        for dep_cand_proc in candidate_procs[dep_task]:
            # We don't consider running processes here..
            if dep_cand_proc[2] is None:
                continue
            dep_proc = dep_cand_proc[0]
            output_node = dep_cand_proc[2][deps[0][0]]
            qb = QueryBuilder()
            qb.append(
                cls=type(output_node),
                filters={"id": {"==": output_node.pk}},
                tag="dep_node",
                edge_filters={"label": deps[0][1]},
            )
            qb.append(cls=proc_cls, with_incoming="dep_node")

            for proc in qb.all(flat=True):
                if self._group is not None and proc not in self._group.nodes:
                    continue
                # if self._group is not None and
                inputs = self._load_process_port(proc, "incoming")
                deps_match = True
                for dep_labels in deps[1:]:
                    if (dep_labels[1] not in inputs) or (
                        dep_cand_proc[2][dep_labels[0]].uuid != inputs[dep_labels[1]].uuid
                    ):
                        deps_match = False
                if deps_match:
                    process_pool.append(
                        [proc, inputs, {dep_task: (dep_proc, dep_cand_proc[1], dep_cand_proc[2])}]
                    )

        # Check if all other deps match:
        for proc_idx, proc_details in enumerate(process_pool):
            for dep_task in list(task_details["dependencies"].keys())[1:]:
                deps = task_details["dependencies"][dep_task]
                for dep_cand_proc in candidate_procs[dep_task]:
                    # We don't consider running processes here..
                    if dep_cand_proc[2] is None:
                        continue
                    deps_match = True
                    for dep_labels in deps:
                        if (
                            dep_cand_proc[2][dep_labels[0]].uuid
                            != proc_details[1][dep_labels[1]].uuid
                        ):
                            deps_match = False
                            # We break if one input parameter does not match the dependency
                            # condition.
                            break
                    if deps_match:
                        proc_details[2][dep_task] = (
                            dep_cand_proc[0],
                            dep_cand_proc[1],
                            dep_cand_proc[2],
                        )
                        # We break if the dependent process is found.
                        break
                if dep_task not in proc_details[2]:
                    # We break if dependent inputs cannot be referenced to a dependent process.
                    break

        # Check if all dependencies could be fulfilled and the input parameters match:
        for proc_details in process_pool:
            if any([dep_task not in proc_details[2] for dep_task in dep_tasks]):
                continue
            if self._check_process_inputs(task, proc_details[1], parent_node):
                proc = proc_details[0]
                if proc.is_finished_ok:
                    outputs = self._load_process_port(proc, "outgoing")
                    found_candidate = True
                    candidate_procs[task].append((proc, proc_details[1], outputs, proc_details[2]))
                elif proc.is_failed or proc.is_killed or proc.is_excepted:
                    self._failed_tasks[task] = proc
                elif not proc.is_terminated:
                    self._running_tasks[task] = proc
                    candidate_procs[task].append((proc, proc_details[1], None, proc_details[2]))
        return found_candidate

    def _retrieve_result(self, result_label, proc_details):
        """Retrieve result of the process."""
        result_details = self._protocol["results"][result_label]

        if result_details["output_port"] in proc_details[2]:
            output_node = proc_details[2][result_details["output_port"]]
            if "retrieve_value" in result_details:
                self._results[result_label] = {
                    "value": dict_retrieve_parameter(
                        output_node.get_dict(), result_details["retrieve_value"]
                    )
                }
                if "unit" in result_details and result_details["unit"] is not None:
                    self._results[result_label]["unit"] = result_details["unit"]
            else:
                if hasattr(output_node, "value"):
                    self._results[result_label] = {"value": output_node.value}
                    if "unit" in result_details and result_details["unit"] is not None:
                        self._results[result_label]["unit"] = result_details["unit"]
                else:
                    self._results[result_label] = self._get_aiida_node_identifier(output_node)
        elif any(
            outp.startswith(result_details["output_port"]) for outp in proc_details[2].keys()
        ):
            # Check dynamic output ports:
            result = {}
            for outp_p, outp_v in proc_details[2].items():
                if outp_p.startswith(result_details["output_port"]):
                    port_tree = outp_p[len(result_details["output_port"]) + 1 :].split(".")
                    res_setter = result
                    for port in port_tree[:-1]:
                        res_setter = res_setter[port]
                    res_setter[port_tree[-1]] = self._get_aiida_node_identifier(outp_v)
            self._results[result_label] = {"value": result}

    def _check_process_inputs(self, task, inputs, parent_node):
        """Check if process inputs are the same as specified in the protocol and by the user."""

        def compare_values(input_node, input_details):
            input_value = input_node
            comp_value = input_details["value"]
            if hasattr(input_value, "uuid"):
                input_value = obtain_value_from_aiida_node(input_value)
            if hasattr(comp_value, "uuid"):
                comp_value = obtain_value_from_aiida_node(comp_value)
            return input_value == comp_value

        def check_input_parameter(task, input_details, input_port, inputs):
            if not input_details["compare"]:
                return True
            if self._check_input_parameter_dependency(task, input_details):
                if input_details["optional"] and input_details["value"] is None:
                    return input_port not in inputs
                elif input_port not in inputs:
                    return False
                else:
                    return compare_values(inputs[input_port], input_details)
            else:
                return True

        task_details = self._tasks[task]
        checked_inputs = []
        for dep_list in task_details["dependencies"].values():
            checked_inputs += [dep[1] for dep in dep_list]

        same_input_p = True
        # Check parent node:
        if "parent_node" in task_details:
            p_node = inputs.get(task_details["parent_node"])
            if p_node is None or p_node.uuid != parent_node.uuid:
                same_input_p = False
            checked_inputs.append(task_details["parent_node"])

        # Check blacklist inputs:
        if same_input_p and any(
            inp_p in task_details["blacklist_inputs"] for inp_p in inputs.keys()
        ):
            same_input_p = False

        # Check task specific parameters:
        if same_input_p:
            for input_port, input_details in self._individual_input[task].items():
                if not check_input_parameter(task, input_details, input_port, inputs):
                    # if not adopt_user_input and not input_details["user_input"]:
                    same_input_p = False
                    break
                checked_inputs.append(input_port)

        # Check the remaining parameters:
        if same_input_p and "inputs" in task_details:
            for input_port in task_details["inputs"]:
                if input_port in checked_inputs:
                    continue
                if input_port in self._user_input:
                    input_details = self._user_input[input_port]
                elif input_port in self._general_input:
                    input_details = self._general_input[input_port]
                else:
                    raise WorkflowProtocolError(
                        f"Workflow protocol is incomplete, {input_port} of task {task} is not "
                        "covered."
                    )
                if not check_input_parameter(task, input_details, input_port, inputs):
                    same_input_p = False
                    break
                checked_inputs.append(input_port)
        return same_input_p

    def _check_input_parameter_dependency(self, task, input_details):
        """Check whether an input parameter is dependent on another parameter."""
        dep_met = True
        dep_input_p = input_details.get("dependency")
        if dep_input_p is not None:
            dep_met = False
            for category in [self._individual_input[task], self._user_input, self._general_input]:
                for input_p, content in category.items():
                    if input_p == dep_input_p:
                        if content.get("value") is not None:
                            dep_met = True
        return dep_met

    def _update_input_nodes(self, task, proc_inputs):
        """Update internal input parameters based on finished AiiDA processes."""
        task_details = self._tasks[task]
        updated_p = []
        # Add dependencies:
        for deps in task_details["dependencies"].values():
            updated_p += [dep[1] for dep in deps]

        for input_p, input_details in self._individual_input[task].items():
            if "code" in input_p:
                continue
            if not input_details["unstored"]:
                if input_details["optional"] and input_details["value"] is None:
                    updated_p.append(input_p)
                    continue
                if not self._check_input_parameter_dependency(task, input_details):
                    updated_p.append(input_p)
                    continue
                input_details["value"] = proc_inputs[input_p]
            updated_p.append(input_p)
        if "inputs" in task_details:
            for input_p in task_details["inputs"]:
                if "code" in input_p:
                    continue
                if input_p in updated_p:
                    continue
                if input_p in self._user_input:
                    input_details = self._user_input[input_p]
                else:
                    input_details = self._general_input[input_p]
                if input_details["optional"] and input_details["value"] is None:
                    continue
                if not self._check_input_parameter_dependency(task, input_details):
                    continue
                if input_details["namespace"]:
                    continue
                if not input_details["unstored"]:
                    input_details["value"] = proc_inputs[input_p]

    def _extract_parent_nodes(self, content):
        if self._parent_node is not None:
            content["parent_node"] = self._parent_node.uuid

    @staticmethod
    def _set_input_parameter(process_builder, input_port, value):
        """Set input parameter."""
        input_setter = process_builder
        input_path = input_port.split(".")
        for keyword in input_path[:-1]:
            input_setter = input_setter[keyword]
        input_setter[input_path[-1]] = value

    @staticmethod
    def _load_process_port(process, link_direction):
        """Get one input or output parameter from the AiiDA process."""
        link_list = process.base.links.get_stored_link_triples(link_direction=link_direction)
        return {
            link.link_label.replace("__", "."): link.node
            for link in link_list
            if link.link_label != "CALL"
        }

    @staticmethod
    def _load_process_class(entry_point):
        """Load process class based on the entry point."""
        try:
            process = WorkflowFactory(entry_point)
        except MissingEntryPointError:
            process = CalculationFactory(entry_point)
        return process


class MultipleWorkflowBuilder(_BaseWorkflowBuilder):
    """
    Workflow builder that can manage a workflow for several parent nodes at once.
    """

    def __init__(
        self,
        aiida_group=None,
        protocol=None,
        graph_attributes={"graph_attr": {"size": "10!,6"}},
        use_uuid=False,
    ):
        """
        Initialize the object.
        """
        _BaseWorkflowBuilder.__init__(self)
        self._wf_builders = []
        self._task_queue = []
        self._graph_attributes = None
        if aiida_group is not None:
            self.aiida_group = aiida_group
        if protocol is not None:
            self.protocol = protocol
        self.graph_attributes = graph_attributes
        self.use_uuid = use_uuid

    @property
    def task_queue(self):
        """
        Return the task queue.
        """
        return self._task_queue

    @property
    def aiida_group(self):
        """
        str or aiida.orm.Group : Only consider processes in the AiiDA group.
        """
        return self._group

    @aiida_group.setter
    def aiida_group(self, value):
        self._set_aiida_group(value)
        for wf_builder in self._wf_builders:
            wf_builder.aiida_group = self._group

    @property
    def graph_attributes(self):
        """
        Graphiz graph attributes for the provenance graph.
        """
        return self._graph_attributes

    @graph_attributes.setter
    def graph_attributes(self, value):
        self._graph_attributes = value
        for wf_builder in self._wf_builders:
            wf_builder.graph_attributes = value

    def add_to_task_queue(self, task_label, run_type="run"):
        """
        Add a task to the task queue.

        Parameters
        ----------
        task_label : str
            Label of the task.
        run_type : str (optional)
            Run type of the process, either ``run`` or ``submit``. The default value is ``run``.
        """
        if task_label not in self._tasks:
            raise ValueError(f"Task `{task_label}` not part of the workflow.")
        if run_type not in ["run", "submit"]:
            raise ValueError(f"run_type `{run_type}` needs to be 'run' or 'submit'.")
        self._task_queue.append((task_label, run_type))

    def reset_task_queue(self):
        """
        Reset the task queue.
        """
        self._task_queue = []

    def set_user_input(self, input_port, value):
        """
        Set a user input parameter of the workflow for all parent nodes.

        Parameters
        ----------
        input_port : str
            Input port of the Parameter.
        value : variable
            Value of the input parameter.
        """
        super().set_user_input(input_port, value)
        for wf_builder in self._wf_builders:
            wf_builder._user_input = self._user_input.copy()
            wf_builder._individual_input = self._individual_input.copy()

    def add_parent_node(self, parent_node):
        """
        Add new parent node.

        Parameters
        ----------
        parent_node : aiida.orm.Node
        """
        if self._protocol is None:
            raise ValueError("A protocol needs to be specified before adding parent nodes.")
        parent_node = _load_data_node(parent_node)
        wf_builder = WorkflowBuilder()
        wf_builder.protocol = copy.deepcopy(self._protocol)
        wf_builder._user_input = copy.deepcopy(self._user_input)
        wf_builder._general_input = copy.deepcopy(self._general_input)
        wf_builder._individual_input = copy.deepcopy(self._individual_input)
        wf_builder.parent_node = parent_node
        wf_builder._group = self._group
        wf_builder.graph_attributes = self.graph_attributes
        self._wf_builders.append(wf_builder)

    def import_parent_nodes_from_pandas_df(self, data_frame, parent_node="structure_node"):
        """
        Extract parent nodes from a pandas data frame. The data frame needs to have a column
        called 'aiida_uuid' with the universally unique identifier (uuid) of the nodes.

        Parameters
        ----------
        data_frame : pandas.DataFrame
            Pandas data frame containing the uuids, primary keys of the AiiDA nodes or the AiiDA
            nodes themselves.
        parent_node : str (optional)
            Label of the column containing the parent nodes for the workflow. The default value is
            ``'structure_node'``.
        """
        for p_node_value in data_frame[parent_node].sort_values():
            try:
                self.add_parent_node(p_node_value)
            except TypeError:
                continue

    def import_parent_nodes_from_aiida_db(self, group_labels):
        """
        Import parent nodes from the AiiDA database.

        Parameters
        ----------
        group_labels : str or list
            AiiDA group label or list of group labels.
        """
        if self._protocol is None:
            raise ValueError("A protocol needs to be specified before adding parent nodes.")
        if not isinstance(group_labels, list):
            group_labels = [group_labels]
        for group_label in group_labels:
            PNData = DataFactory(self._protocol["parent_node_type"])
            queryb = aiida_orm.querybuilder.QueryBuilder()
            if group_label is None:
                queryb.append(PNData)
            else:
                queryb.append(aiida_orm.Group, filters={"label": group_label}, tag="group")
                queryb.append(PNData, with_group="group")
            for node in queryb.all(flat=True):
                self.add_parent_node(node)

    def adopt_input_nodes_from_workflow(self, parent_node):
        """
        Import input parameters for the workflow based on another parent node.

        Parameters
        ----------
        parent_node : aiida.node
            Parent AiiDA node.
        """
        for wf_builder in self._wf_builders:
            wf_builder.adopt_input_nodes_from_workflow(parent_node)

    def return_process_nodes(self):
        """
        Return a pandas data frame containing the process nodes of all completed tasks.

        Returns
        -------
        pandas.DataFrame
            Pandas data frame.
        """
        pd_series_dict = {"parent_node": []}
        for task in self.tasks.keys():
            pd_series_dict[task] = []
        for wf_builder in self._wf_builders:
            pd_series_dict["parent_node"].append(
                self._get_aiida_node_identifier(wf_builder.parent_node)
            )
            wf_builder.determine_workflow_state()
            for task in self.tasks.keys():
                if task in wf_builder._completed_tasks:
                    pd_series_dict[task].append(
                        self._get_aiida_node_identifier(wf_builder._completed_tasks[task][0])
                    )
                elif task in wf_builder._running_tasks:
                    pd_series_dict[task].append(
                        self._get_aiida_node_identifier(wf_builder._running_tasks[task])
                    )
                elif task in wf_builder._failed_tasks:
                    pd_series_dict[task].append(
                        self._get_aiida_node_identifier(wf_builder._failed_tasks[task])
                    )
                else:
                    pd_series_dict[task].append(None)
        return _turn_dict_into_pandas_df(pd_series_dict)

    def return_results(self, include="all", exclude=[]):
        """
        Return a pandas data frame containing the calculated results of all completed tasks.

        Parameters
        ----------
        include : list or str (optional)
            List of results that are included in the pandas data frame. The value ``'all'``
            returns all results. The default value is ``'all'``.
        exclude : list (optional)
            List of results that are excluded in the pandas data frame. The default value is
            ``[]``.

        Returns
        -------
        pandas.DataFrame
            Pandas data frame.
        """
        if isinstance(include, str):
            if include == "all":
                include = list(self._protocol["results"].keys())
            else:
                raise ValueError("`include` needs to have the value 'all'")
        pandas_df = None
        if "results" in self._protocol and self._protocol["results"] is not None:
            results = {}
            for r_label, r_value in self._protocol["results"].items():
                if r_label not in exclude and r_label in include:
                    results[r_label] = r_value

            pd_series_dict = {"parent_node": []}
            for res_label, res_details in results.items():
                pandas_label = res_label
                if "unit" in res_details:
                    pandas_label += f" ({res_details['unit']})"
                pd_series_dict[pandas_label] = []
            for wf_builder in self._wf_builders:
                wf_builder.use_uuid = self.use_uuid
                pd_series_dict["parent_node"].append(
                    self._get_aiida_node_identifier(wf_builder.parent_node)
                )
                wf_results = wf_builder.results
                for res_label, res_details in results.items():
                    pandas_label = res_label
                    value = None
                    if "unit" in res_details:
                        pandas_label += f" ({res_details['unit']})"
                    if res_label in wf_results:
                        if (
                            isinstance(wf_results[res_label], dict)
                            and "value" in wf_results[res_label]
                        ):
                            value = wf_results[res_label]["value"]
                        else:
                            value = wf_results[res_label]
                    pd_series_dict[pandas_label].append(value)
            pandas_df = _turn_dict_into_pandas_df(pd_series_dict)
        return pandas_df

    def return_remote_runtimes(self):
        """
        Return a pandas data frame containing the total remote runtimes of all completed tasks.

        Returns
        -------
        pandas.DataFrame
            Pandas data frame.
        """
        pd_series_dict = {"parent_node": []}
        for task in self.tasks.keys():
            pd_series_dict[task] = []
        for wf_builder in self._wf_builders:
            pd_series_dict["parent_node"].append(
                self._get_aiida_node_identifier(wf_builder.parent_node)
            )
            acc_tasks = wf_builder.completed_tasks
            for task in self.tasks.keys():
                if task in acc_tasks:
                    task_node = wf_builder._completed_tasks[task][0]
                    if isinstance(task_node, aiida_orm.WorkChainNode):
                        total_runtime = get_workchain_runtime(task_node)
                    elif isinstance(task_node, aiida_orm.CalcFunctionNode):
                        total_runtime = timedelta(seconds=0)  # No remote computing
                    pd_series_dict[task].append(total_runtime)
                if task not in acc_tasks:
                    pd_series_dict[task].append(None)
        return _turn_dict_into_pandas_df(pd_series_dict)

    def return_runtimes(self, unit=None):
        """
        Return a pandas data frame containing the runtimes of all completed tasks.

        Returns
        -------
        pandas.DataFrame
            Pandas data frame.
        """
        pd_series_dict = {"parent_node": []}
        for task in self.tasks.keys():
            pd_series_dict[task] = []
        for wf_builder in self._wf_builders:
            pd_series_dict["parent_node"].append(
                self._get_aiida_node_identifier(wf_builder.parent_node)
            )
            acc_tasks = wf_builder.completed_tasks
            for task in self.tasks.keys():
                if task in acc_tasks:
                    runtime = acc_tasks[task].mtime - acc_tasks[task].ctime
                    if unit is not None:
                        runtime = getattr(runtime, unit)
                    pd_series_dict[task].append(runtime)
                else:
                    pd_series_dict[task].append(None)
        return _turn_dict_into_pandas_df(pd_series_dict)

    def return_workflow_states(self, unformatted=False):
        """
        Return a pandas data frame containing the current state of  workflow.

        Parameters
        ---------
        unformatted : bool (optional)
            If set to ``True`` an unformatted pandas.DataFrame object is returned otherwise a
            pandas.Styler object is returned.

        Returns
        -------
        pandas.DataFrame or pandas.Styler
            Pandas data frame.
        """
        pd_series_dict = {"parent_node": [], "parent_label": []}
        for task in self.tasks.keys():
            pd_series_dict[task] = []
        for wf_builder in self._wf_builders:
            wf_state = wf_builder.determine_workflow_state()
            pd_series_dict["parent_node"].append(
                self._get_aiida_node_identifier(wf_builder.parent_node)
            )
            if wf_builder.parent_node.label == "":
                pd_series_dict["parent_label"].append(None)
            else:
                pd_series_dict["parent_label"].append(wf_builder.parent_node.label)
            for task in self.tasks.keys():
                if task in wf_state["completed_tasks"]:
                    pd_series_dict[task].append("completed")
                elif task in wf_state["running_tasks"]:
                    pd_series_dict[task].append("running")
                elif task in wf_state["failed_tasks"]:
                    pd_series_dict[task].append(
                        f"failed [{wf_builder._failed_tasks[task].exit_status}]"
                    )
                elif task in wf_state["next_possible_tasks"]:
                    pd_series_dict[task].append("deps. met")
                else:
                    pd_series_dict[task].append("missing deps.")
        pandas_df = _turn_dict_into_pandas_df(pd_series_dict)
        if not unformatted:
            pandas_df = _apply_color_map(pandas_df, _wf_states_color_map)
        return pandas_df

    def generate_inputs(self, parent_node, task):
        """
        Generate a dictionary or builder for a certain task of the workflow for a specific parent
        node.

        Parameters
        ----------
        parent_node : aiida.orm.node or str or int
            Uuid, primary key or the AiiDA node itself.
        task : str
            Task label.
        """
        inputs = None
        for wf_builder in self._wf_builders:
            if isinstance(parent_node, str):
                wf_parent_node = wf_builder.parent_node.uuid
            elif isinstance(parent_node, int):
                wf_parent_node = wf_builder.parent_node.pk
            else:
                wf_parent_node = wf_builder.parent_node
            if wf_parent_node == parent_node:
                inputs = wf_builder.generate_inputs(task)
                break
        return inputs

    def run_task(self, task_label, interval=None, node_ids=None):
        """
        Run the AiiDA processes of the task for all parent nodes.

        Parameters
        ----------
        task_label : str
            Label of the workflow task.
        interval : tuple (optional)
            Lower and upper bound of the interval of workflows that are submitted. ``None`` to run
            all workflows. The default value is ``None``.
        node_ids : list (optional)
            List of primary keys or uuids of the parent nodes of the workflows that are submitted.
            ``None`` to run all workflows.
        """
        self._perform_task(task_label, "run_task", interval, node_ids)

    def submit_task(self, task_label, interval=None, node_ids=None):
        """
        Submit the AiiDA processes of the task for all parent nodes.

        Parameters
        ----------
        task_label : str
            Label of the workflow task.
        interval : tuple (optional)
            Lower and upper bound of the interval of workflows that are submitted. ``None`` to run
            all workflows. The default value is ``None``.
        node_ids : list (optional)
            List of primary keys or uuids of the parent nodes of the workflows that are submitted.
            ``None`` to run all workflows.
        """
        self._perform_task(task_label, "submit_task", interval, node_ids)

    def execute_task_queue(self, interval=None, node_ids=None):
        """
        Execute a series of AiiDA processes for several tasks consecutively for all parent nodes
        as defined in the ``task_queue`` attribute.

        Parameters
        ----------
        interval : tuple (optional)
            Lower and upper bound of the interval of workflows that are executed. ``None`` to run
            all workflows. The default value is ``None``.
        node_ids : list (optional)
            List of primary keys or uuids of the parent nodes of the workflows that are started.
            ``None`` to run all workflows.
        """
        for task_label, run_type in self._task_queue:
            self._perform_task(task_label, run_type + "_task", interval, node_ids)

    def generate_provenance_graph(self, parent_node_index):
        """
        Generate provenance graph of the workflow.

        Returns
        -------
        : aiida.tools.visualization.graph.Graph
            Provenance graph.
        """
        return self._wf_builders[parent_node_index].generate_provenance_graph()

    def _perform_task(self, task_label, run_type, interval, node_ids):
        """Perform task (run or submit)."""
        if task_label not in self.tasks.keys():
            raise ValueError(f"Task '{task_label}' is not part wof the workflow.")
        wf_builder_indices = []
        if node_ids is not None:
            for builder_idx, wf_builder in enumerate(self._wf_builders):
                if (
                    wf_builder.parent_node.pk in node_ids
                    or wf_builder.parent_node.uuid in node_ids
                ):
                    wf_builder_indices.append(builder_idx)
        if interval is not None:
            wf_builder_indices += list(
                range(interval[0], min(interval[1], len(self._wf_builders)))
            )
        if interval is None and node_ids is None:
            wf_builder_indices = list(range(0, len(self._wf_builders)))
        for builder_idx in set(wf_builder_indices):
            wf_builder = self._wf_builders[builder_idx]
            tasks = wf_builder.determine_workflow_state()
            if task_label in tasks["next_possible_tasks"]:
                function = getattr(wf_builder, run_type)
                function(task_label)

    def _extract_parent_nodes(self, content):
        if len(self._wf_builders) > 0:
            content["parent_nodes"] = [
                wf_builder.parent_node.uuid for wf_builder in self._wf_builders
            ]
