"""Test workflow builder."""

import os


# Third party library imports
import pytest
from aiida.orm import Float, Int
from aiida.engine import run_get_node

# Internal library imports
from workflow_utils import generate_work_function_node
from aim2dat.aiida_workflows.workflow_builder import WorkflowBuilder, WorkflowProtocolError
from aim2dat.aiida_workflows._workflow_builder_utils import _load_protocol
from aim2dat.aiida_workflows.utils import obtain_value_from_aiida_node
from aim2dat.io import read_yaml_file

MAIN_PATH = os.path.dirname(__file__) + "/workflow_builder/"


class NodeContainer:
    """Container to store unique nodes for later tests."""

    def __init__(self):
        """Initialize object."""
        self._nodes = {}

    def get_node(self, label):
        """Return node."""
        return self._nodes[label]

    def add_node(self, label, node):
        """Add node."""
        self._nodes[label] = node


node_container = NodeContainer()


@pytest.mark.aiida
def test_workflow_validation():
    """Test protocol validation."""
    protocol = dict(read_yaml_file(MAIN_PATH + "flawed_protocols/missing-process.yaml"))
    wf_builder = WorkflowBuilder()
    with pytest.raises(WorkflowProtocolError) as error:
        wf_builder.protocol = protocol
    assert str(error.value) == "No process specified for task 'task_1'."

    protocol = dict(read_yaml_file(MAIN_PATH + "flawed_protocols/missing-task.yaml"))
    wf_builder = WorkflowBuilder()
    with pytest.raises(WorkflowProtocolError) as error:
        wf_builder.protocol = protocol
    assert str(error.value) == "Task 'task_2' does not exist."

    protocol = dict(read_yaml_file(MAIN_PATH + "flawed_protocols/parent-node-clash.yaml"))
    wf_builder = WorkflowBuilder()
    with pytest.raises(WorkflowProtocolError) as error:
        wf_builder.protocol = protocol
    assert str(error.value) == "Input 'input_variable' clashes with parent node at task 'task_1'."

    protocol = dict(read_yaml_file(MAIN_PATH + "flawed_protocols/input-defined-twice.yaml"))
    wf_builder = WorkflowBuilder()
    with pytest.raises(WorkflowProtocolError) as error:
        wf_builder.protocol = protocol
    assert str(error.value) == "Input 'input_variable' defined twice for task 'task_1'."

    protocol = dict(read_yaml_file(MAIN_PATH + "flawed_protocols/input-defined-twice.yaml"))
    wf_builder = WorkflowBuilder()
    with pytest.raises(WorkflowProtocolError) as error:
        wf_builder.protocol = protocol
    assert str(error.value) == "Input 'input_variable' defined twice for task 'task_1'."

    protocol = dict(read_yaml_file(MAIN_PATH + "flawed_protocols/blacklist-clash.yaml"))
    wf_builder = WorkflowBuilder()
    with pytest.raises(WorkflowProtocolError) as error:
        wf_builder.protocol = protocol
    assert str(error.value) == "Input 'input_variable' is on the blacklist of task 'task_1'."


@pytest.mark.aiida
def test_inputs(aiida_profile):
    """Test parent node and protocol input."""
    wf_builder = WorkflowBuilder(protocol="arithmetic-testing")
    with pytest.raises(TypeError) as error:
        wf_builder.parent_node = Int(3.0)
    assert str(error.value) == "Parent node needs to be of type: `float` for this protocol."
    with pytest.raises(ValueError) as error:
        wf_builder = WorkflowBuilder(parent_node=Int(3.0))
    assert str(error.value) == "`protocol` needs to be set."


@pytest.mark.aiida
def test_arithmetic_two_consecutive_processes(aiida_profile):
    """Test two consecutive processes of the same task."""
    p_node = Float(0.2)
    y = Float(0.3)
    y_task41 = Float(0.6)
    p_node.store()
    y.store()
    y_task41.store()
    node_container.add_node("p_node", p_node)
    node_container.add_node("y", y)
    node_container.add_node("y_task41", y_task41)

    wf_builder = WorkflowBuilder()
    wf_builder.protocol = "arithmetic-testing"
    wf_builder.parent_node = p_node
    wf_builder.set_user_input("y", y)
    wf_builder.set_user_input("y->task_4.1", y_task41)
    builder = wf_builder.generate_inputs("task_1.1")
    _, proc_node1 = run_get_node(**builder)
    _, proc_node2 = run_get_node(**builder)
    completed_tasks = wf_builder.completed_tasks
    assert len(completed_tasks) == 1, "Wrong number of completed tasks."
    assert "task_1.1" in completed_tasks, "Wrong task in completed tasks."
    assert completed_tasks["task_1.1"].uuid == proc_node2.uuid, "Wrong task."


@pytest.mark.aiida
def test_arithmetic_workflow_success(aiida_profile):
    """Test the result of the arithmetic-testing workflow."""

    def add_multiply(x, y, z):
        return (x + y) * z

    p_node = node_container.get_node("p_node")
    y = node_container.get_node("y")
    y_task41 = node_container.get_node("y_task41")

    results = {
        "task_1.1": add_multiply(p_node.value, y.value, 2.0),
        "task_1.2": add_multiply(p_node.value, y.value, 4.0),
        "task_1.3": add_multiply(10.0, y.value, p_node.value),
    }
    results["task_2.1"] = add_multiply(y.value, results["task_1.2"], results["task_1.3"])
    results["task_2.2"] = add_multiply(p_node.value, results["task_1.3"], 2.0)
    results["task_3.1"] = add_multiply(
        results["task_1.1"], results["task_2.1"], results["task_2.2"]
    )
    results["task_4.1"] = add_multiply(results["task_3.1"], y_task41.value, results["task_1.2"])

    wf_builder = WorkflowBuilder()
    wf_builder.protocol = "arithmetic-testing"
    wf_builder.parent_node = p_node
    wf_builder.set_user_input("y", y)
    wf_builder.set_user_input("y->task_4.1", y_task41)
    for task, result in results.items():
        comp_tasks = wf_builder.completed_tasks
        if task not in comp_tasks:
            _, result_wf = wf_builder.run_task(task)
        else:
            result_wf = comp_tasks[task].outputs["result"]
        assert abs(result_wf.value - result) < 1.0e-6, f"Task {task} is wrong."

    wf_state = wf_builder.determine_workflow_state()
    assert sorted(wf_state["completed_tasks"]) == list(
        results.keys()
    ), "Completed tasks are wrong."
    assert wf_state["next_possible_tasks"] == [], "Next tasks are wrong."
    assert wf_state["running_tasks"] == [], "Running tasks are wrong."
    assert wf_state["failed_tasks"] == [], "Failed tasks are wrong."
    wf_results = wf_builder.results
    assert abs(wf_results["res_1"]["value"] - results["task_4.1"]) < 1.0e-6, "Result is wrong."
    assert wf_results["res_1"]["unit"] == "test_unit", "Unit of result is wrong."


@pytest.mark.aiida
def test_arithmetic_workflow_states(aiida_profile):
    """Test the detection of failed and running processes."""
    p_node = node_container.get_node("p_node")
    y = node_container.get_node("y")
    y_task41 = node_container.get_node("y_task41")

    wf_builder = WorkflowBuilder()
    wf_builder.protocol = "arithmetic-testing"
    wf_builder.parent_node = p_node
    wf_builder.set_user_input("y", y)
    wf_builder.set_user_input("y->task_4.1", y_task41)
    wf_builder.set_user_input("y->task_4.1", Float(0.3))
    wf_state = wf_builder.determine_workflow_state()
    assert wf_state["next_possible_tasks"] == ["task_4.1"], "Updating workflow state not working."

    for proc_state in ["killed", "excepted", "running"]:
        inputs = wf_builder.generate_inputs("task_4.1")
        inputs.pop("process")
        proc = generate_work_function_node(
            "arithmetic.add_multiply",
            proc_state,
            inputs=inputs,
        )
        for input_node in inputs.values():
            input_node.store()
        proc.store()
        if proc_state in ["killed", "excepted"]:
            assert (
                proc == wf_builder.failed_tasks["task_4.1"]
            ), "Failed tasks not properly detected."
        else:
            assert (
                proc == wf_builder.running_tasks["task_4.1"]
            ), "Running tasks not properly detected."


@pytest.mark.aiida
def test_arithmetic_adopt_input_parameters(aiida_profile):
    """Test the adoptation of input nodes from another parent node."""
    p_node = node_container.get_node("p_node")
    y = node_container.get_node("y")
    y_task41 = node_container.get_node("y_task41")

    wf_builder = WorkflowBuilder()
    wf_builder.protocol = "arithmetic-testing"
    wf_builder.parent_node = Float(2.5)
    wf_builder.set_user_input("y", y)
    wf_builder.set_user_input("y->task_4.1", y_task41)
    wf_builder.adopt_input_nodes_from_workflow(p_node)
    assert wf_builder._user_input["y"]["value"].uuid == y.uuid, "User input node is not adopted."
    assert (
        wf_builder._individual_input["task_4.1"]["y"]["value"].uuid == y_task41.uuid
    ), "Individual user input is not adopted."
    assert wf_builder._parent_node.value == 2.5, "Parent node has been changed."


@pytest.mark.aiida
def test_arithmetic_aiida_group(aiida_profile):
    """Test the group constraint of the workflow builder class."""
    p_node = node_container.get_node("p_node")
    y = node_container.get_node("y")
    y_task41 = node_container.get_node("y_task41")

    wf_builder = WorkflowBuilder()
    wf_builder.protocol = "arithmetic-testing"
    wf_builder.aiida_group = "test_group"
    wf_builder.parent_node = p_node
    wf_builder.set_user_input("y", y)
    wf_builder.set_user_input("y->task_4.1", y_task41)
    wf_state = wf_builder.determine_workflow_state()
    assert wf_state["next_possible_tasks"] == [
        "task_1.1",
        "task_1.2",
        "task_1.3",
    ], "Group constraint not working."


@pytest.mark.aiida
def test_file_support(aiida_profile, nested_dict_comparison):
    """Test storing/reading workflow data to/from file."""
    p_node = Float(0.2)
    y = Float(0.3)
    y_task41 = Float(0.6)
    p_node.store()
    y.store()
    y_task41.store()

    wf_builder = WorkflowBuilder()
    wf_builder.protocol = "arithmetic-testing"
    wf_builder.parent_node = p_node
    wf_builder.set_user_input("y", y)
    wf_builder.set_user_input("y->task_4.1", y_task41)
    wf_builder._general_input["z"]["value"].store()
    wf_builder._individual_input["task_1.2"]["z"]["value"].store()
    wf_builder.to_file()

    wf_builder_2 = WorkflowBuilder.from_file()
    nested_dict_comparison(wf_builder.protocol, wf_builder_2.protocol)
    assert wf_builder.parent_node.uuid == wf_builder_2.parent_node.uuid
    assert wf_builder._user_input["y"]["value"].uuid == wf_builder_2._user_input["y"]["value"].uuid
    assert (
        wf_builder._general_input["z"]["value"].uuid
        == wf_builder_2._general_input["z"]["value"].uuid
    )
    assert (
        wf_builder._individual_input["task_1.2"]["z"]["value"].uuid
        == wf_builder_2._individual_input["task_1.2"]["z"]["value"].uuid
    )
    assert (
        wf_builder._individual_input["task_4.1"]["y"]["value"].uuid
        == wf_builder_2._individual_input["task_4.1"]["y"]["value"].uuid
    )


@pytest.mark.parametrize(
    "protocol,version",
    [("protocol", 1.2), ("protocol_v1.2", 1.2), ("protocol_v1", 1.0), ({"version": 2.3}, 2.3)],
)
def test_protocol_version(protocol, version):
    """Test correct versioning of _load_protocol function."""
    protocol_dict = _load_protocol(protocol, MAIN_PATH + "protocol_versions/")
    assert protocol_dict["version"] == version


@pytest.mark.parametrize(
    "protocol",
    [
        "seekpath-standard",
        "cp2k-crystal-standard",
        "cp2k-crystal-standard-keep-angles",
        "cp2k-crystal-mofs",
        "cp2k-crystal-preopt",
        "cp2k-surface-standard",
    ],
)
def test_protocols(nested_dict_comparison, protocol):
    """Test protocol that all standard protocols are valid."""
    ref = read_yaml_file(MAIN_PATH + "protocols/" + protocol + "_ref.yaml")
    wf_builder = WorkflowBuilder()
    wf_builder.protocol = protocol
    for label, input_det in wf_builder._general_input.items():
        if input_det["aiida_node"]:
            input_det["value"] = obtain_value_from_aiida_node(input_det["value"])
    for task, task_input in wf_builder._individual_input.items():
        for label, input_det in task_input.items():
            if input_det["aiida_node"] and input_det["value"] is not None:
                input_det["value"] = obtain_value_from_aiida_node(input_det["value"])
    nested_dict_comparison(wf_builder._protocol, ref["protocol"])
    nested_dict_comparison(wf_builder._user_input, ref["user_input"])
    nested_dict_comparison(wf_builder._general_input, ref["general_input"])
    nested_dict_comparison(wf_builder._individual_input, ref["Individual_input"])
    nested_dict_comparison(wf_builder._tasks, ref["tasks"])
    nested_dict_comparison(wf_builder._result_dict, ref["result_dict"])
