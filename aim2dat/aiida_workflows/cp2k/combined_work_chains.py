"""Deprecated module redirecting to the new wfl sub-package."""

# Internal library imports
import aim2dat.wfl.aiida.cp2k as aiida_cp2k


class SurfaceOptWorkChain(aiida_cp2k.combined_work_chains.SurfaceOptWorkChain):
    """Deprecated work chain class."""

    def __init__(self, *args, **kwargs):
        """Initialize class."""
        from warnings import warn

        warn(
            "This class will be removed, please use "
            + "`aim2dat.wfl.aiida.cp2k.combined_work_chains.SurfaceOptWorkChain` instead.",
            DeprecationWarning,
            2,
        )
        super().__init__(*args, **kwargs)


class ElectronicPropertiesWorkChain(aiida_cp2k.combined_work_chains.ElectronicPropertiesWorkChain):
    """Deprecated work chain class."""

    def __init__(self, *args, **kwargs):
        """Initialize class."""
        from warnings import warn

        warn(
            "This class will be removed, please use "
            + "`aim2dat.wfl.aiida.cp2k.combined_work_chains.ElectronicPropertiesWorkChain` "
            + "instead.",
            DeprecationWarning,
            2,
        )
        super().__init__(*args, **kwargs)
