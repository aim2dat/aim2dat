"""Deprecated module redirecting to the new wfl sub-package."""

# Internal library imports
import aim2dat.wfl.aiida.cubecruncher as aiida_cubecruncher


class CubecruncherCalculation(aiida_cubecruncher.calcjobs.CubecruncherCalculation):
    """Deprecated work chain class."""

    def __init__(self, *args, **kwargs):
        """Initialize class."""
        from warnings import warn

        warn(
            "This class will be removed, please use "
            + "`aim2dat.wfl.aiida.cubecruncher.calcjobs.CubecruncherCalculation` instead.",
            DeprecationWarning,
            2,
        )
        super().__init__(*args, **kwargs)
