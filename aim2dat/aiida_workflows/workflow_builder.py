"""
WorkflowBuilder and auxiliary functions.
"""

# Internal library imports
import aim2dat.wfl as a2d_wfl


class WorkflowProtocolError(a2d_wfl.WorkflowProtocolError):
    """Error for incomplete or inconsistent workflow protocols."""

    from warnings import warn

    warn(
        "This class will be removed, please use `aim2dat.wfl.WorkflowProtocolError` instead.",
        DeprecationWarning,
        2,
    )


def WorkflowBuilder(*args, **kwargs):
    """Depreciated workflow builder class."""
    from warnings import warn

    warn(
        "This class will be removed, please use `aim2dat.wfl.WorkflowBuilder` instead.",
        DeprecationWarning,
        2,
    )
    return a2d_wfl.WorkflowBuilder(*args, **kwargs)


def MultipleWorkflowBuilder(*args, **kwargs):
    """Depreciated workflow builder class."""
    from warnings import warn

    warn(
        "This class will be removed, please use `aim2dat.wfl.MultipleWorkflowBuilder` instead.",
        DeprecationWarning,
        2,
    )
    return a2d_wfl.MultipleWorkflowBuilder(*args, **kwargs)
