"""Deprecated AiiDA CP2K calcjob."""

# Internal library imports
import aim2dat.wfl.aiida.cp2k as aiida_cp2k


class Cp2kCalculation(aiida_cp2k.calcjobs.Cp2kCalculation):
    """Placeholder."""

    def __init__(self, *args, **kwargs):
        """Initialize class."""
        from warnings import warn

        warn(
            "This class will be removed, please use "
            + "`aim2dat.wfl.aiida.cp2k.calcjobs.Cp2kCalculation` instead.",
            DeprecationWarning,
            2,
        )
        super().__init__(*args, **kwargs)
