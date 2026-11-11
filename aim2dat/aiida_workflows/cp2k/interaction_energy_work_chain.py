"""Deprecated module redirecting to the new wfl sub-package."""

# Internal library imports
import aim2dat.wfl.aiida.cp2k as aiida_cp2k


class InteractionEnergyWorkChain(
    aiida_cp2k.interaction_energy_work_chain.InteractionEnergyWorkChain
):
    """Deprecated work chain class."""

    def __init__(self, *args, **kwargs):
        """Initialize class."""
        from warnings import warn

        warn(
            "This class will be removed, please use "
            + "`aim2dat.wfl.aiida.cp2k.interaction_energy_work_chain.InteractionEnergyWorkChain` "
            + " instead.",
            DeprecationWarning,
            2,
        )
        super().__init__(*args, **kwargs)
