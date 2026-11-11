"""Deprecated module redirecting to the new wfl sub-package."""

# Internal library imports
import aim2dat.wfl.aiida.critic2 as aiida_critic2


class Critic2Calculation(aiida_critic2.calcjobs.Critic2Calculation):
    """Deprecated work chain class."""

    def __init__(self, *args, **kwargs):
        """Initialize class."""
        from warnings import warn

        warn(
            "This class will be removed, please use "
            + "`aim2dat.wfl.aiida.critic2.calcjobs.Critic2Calculation` instead.",
            DeprecationWarning,
            2,
        )
        super().__init__(*args, **kwargs)
