"""Deprecated module redirecting to the new wfl sub-package."""

# Internal library imports
import aim2dat.wfl.aiida.cp2k as aiida_cp2k


class ChargeDensityDifferenceWorkChain(
    aiida_cp2k.charge_density_difference_work_chain.ChargeDensityDifferenceWorkChain
):
    """Deprecated work chain class."""

    def __init__(self, *args, **kwargs):
        """Initialize class."""
        from warnings import warn

        warn(
            "This class will be removed, please use "
            + "`aim2dat.wfl.aiida.cp2k.charge_density_difference_work_chain."
            + "ChargeDensityDifferenceWorkChain` instead.",
            DeprecationWarning,
            2,
        )
        super().__init__(*args, **kwargs)
