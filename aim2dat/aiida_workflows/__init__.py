"""Deprecated AiiDA workflows sub-package."""

from aim2dat.aiida_workflows.workflow_builder import (
    WorkflowBuilder,
    MultipleWorkflowBuilder,
    WorkflowProtocolError,
)

__all__ = ["WorkflowBuilder", "MultipleWorkflowBuilder", "WorkflowProtocolError"]
