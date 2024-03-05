"""Test MultipleWorkflowBuilder class."""

# Third party library imports
from aiida.orm import Float, load_node
import pandas as pd
import pytest

# Internal library imports
from aim2dat.aiida_workflows.workflow_builder import MultipleWorkflowBuilder
from aim2dat.ext_interfaces.aiida import _create_group


@pytest.mark.aiida
def test_multiple_workflow_builder(aiida_profile):
    """Test general function principle."""
    m_wfb = MultipleWorkflowBuilder()
    m_wfb.protocol = "arithmetic-testing"

    pn_pks = []
    pn_uuids = []
    for val in range(0, 10):
        p_node = Float(val)
        p_node.store()
        pn_pks.append(p_node.pk)
        pn_uuids.append(p_node.uuid)
        m_wfb.add_parent_node(p_node)

    m_wfb.set_user_input("y", 10.0)
    m_wfb.set_user_input("y->task_4.1", 2.0)
    ws_df = m_wfb.return_workflow_states(unformatted=True)
    for pn_pk, (_, row) in zip(pn_pks, ws_df.iterrows()):
        assert pn_pk == row["parent_node"]
        for task in m_wfb.tasks.keys():
            if float(task.split("_")[1]) < 2.0:
                assert row[task] == "deps. met"
            else:
                assert row[task] == "missing deps."
    m_wfb.use_uuid = True
    ws_df = m_wfb.return_workflow_states(unformatted=True)
    for pn_uuid, (_, row) in zip(pn_uuids, ws_df.iterrows()):
        assert pn_uuid == row["parent_node"]
    m_wfb.use_uuid = False
    interval = [2, 5]
    m_wfb.run_task("task_1.1", interval=interval)
    ws_df = m_wfb.return_workflow_states(unformatted=True)
    for pn_pk, (idx0, row) in zip(pn_pks, ws_df.iterrows()):
        assert pn_pk == row["parent_node"]
        if idx0 >= interval[0] and idx0 < interval[1]:
            assert row["task_1.1"] == "completed"
        else:
            assert row["task_1.1"] == "deps. met"
    proc_df = m_wfb.return_process_nodes()
    assert len(proc_df.columns) == 2
    proc_uuids = []
    for pn_pk, (idx0, row) in zip(pn_pks, proc_df.iterrows()):
        assert pn_pk == row["parent_node"]
        if idx0 >= interval[0] and idx0 < interval[1]:
            proc_uuids.append(load_node(pk=row["task_1.1"]).uuid)
            assert isinstance(row["task_1.1"], int)
        else:
            proc_uuids.append(None)
            assert row["task_1.1"] is pd.NA
    m_wfb.use_uuid = True
    proc_df = m_wfb.return_process_nodes()
    for proc_uuid, (idx0, row) in zip(proc_uuids, proc_df.iterrows()):
        if idx0 >= interval[0] and idx0 < interval[1]:
            assert proc_uuid == row["task_1.1"]
        else:
            assert row["task_1.1"] is pd.NA
    m_wfb.use_uuid = False

    m_wfb.submit_task("task_1.2", node_ids=pn_pks[1:4])
    m_wfb.run_task("task_1.3", interval=interval)

    ws_df = m_wfb.return_workflow_states(unformatted=True)
    for pn_pk, (idx0, row) in zip(pn_pks, ws_df.iterrows()):
        assert pn_pk == row["parent_node"]
        if idx0 >= 1 and idx0 < 4:
            assert row["task_1.2"] == "running"
        else:
            assert row["task_1.2"] == "deps. met"
        assert row["task_2.1"] == "missing deps."
        assert row["task_3.1"] == "missing deps."
        assert row["task_4.1"] == "missing deps."
        if idx0 >= interval[0] and idx0 < interval[1]:
            assert row["task_1.1"] == "completed"
            assert row["task_1.3"] == "completed"
            assert row["task_2.2"] == "deps. met"
        else:
            assert row["task_1.1"] == "deps. met"
            assert row["task_2.2"] == "missing deps."

    res_df = m_wfb.return_results()
    results = [None, None, 40.0, 60.0, 80.0, None, None, None, None, None]
    for pn_pk, res, (idx0, row) in zip(pn_pks, results, res_df.iterrows()):
        assert pn_pk == row["parent_node"]
        if idx0 >= interval[0] and idx0 < interval[1]:
            assert row["res_2 (test_unit)"] == res
        else:
            assert row["res_2 (test_unit)"] is pd.NA

    m_wfb.add_to_task_queue("task_2.2", run_type="run")
    m_wfb.execute_task_queue()
    ws_df = m_wfb.return_workflow_states(unformatted=True)
    for pn_pk, (idx0, row) in zip(pn_pks, ws_df.iterrows()):
        assert pn_pk == row["parent_node"]
        if idx0 >= 1 and idx0 < 4:
            assert row["task_1.2"] == "running"
        else:
            assert row["task_1.2"] == "deps. met"
        assert row["task_2.1"] == "missing deps."
        assert row["task_3.1"] == "missing deps."
        assert row["task_4.1"] == "missing deps."
        if idx0 >= interval[0] and idx0 < interval[1]:
            assert row["task_1.1"] == "completed"
            assert row["task_1.3"] == "completed"
            assert row["task_2.2"] == "completed"
        else:
            assert row["task_1.1"] == "deps. met"
            assert row["task_2.2"] == "missing deps."

    inputs = m_wfb.generate_inputs(pn_pks[0], "task_1.2")
    assert inputs["x"].value == 0.0
    assert inputs["y"].value == 10.0
    assert inputs["z"].value == 4.0

    with pytest.raises(ValueError) as error:
        m_wfb.run_task("test_task")
    assert str(error.value) == "Task 'test_task' is not part wof the workflow."
    with pytest.raises(ValueError) as error:
        m_wfb.return_results(include="test")
    assert str(error.value) == "`include` needs to have the value 'all'"


def test_import_parent_nodes_from_aiida_db(aiida_profile):
    """Test import_parent_nodes_from_aiida_db function."""
    m_wfb = MultipleWorkflowBuilder()
    m_wfb.protocol = "arithmetic-testing"

    aiida_group = _create_group("test_group", None)
    pn_pks = []
    pn_uuids = []
    for val in range(0, 10):
        p_node = Float(val)
        p_node.store()
        pn_pks.append(p_node.pk)
        pn_uuids.append(p_node.uuid)
        aiida_group.add_nodes(p_node)
    m_wfb.import_parent_nodes_from_aiida_db("test_group")
    m_wfb.set_user_input("y", 10.0)
    m_wfb.set_user_input("y->task_4.1", 2.0)
    ws_df = m_wfb.return_workflow_states(unformatted=True)
    for pn_pk, (_, row) in zip(pn_pks, ws_df.iterrows()):
        assert pn_pk == row["parent_node"]
        for task in m_wfb.tasks.keys():
            if float(task.split("_")[1]) < 2.0:
                assert row[task] == "deps. met"
            else:
                assert row[task] == "missing deps."
