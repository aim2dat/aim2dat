"""
AiiDA work chain to optimize the atomic positions using CP2K.
"""

# Internal library imports
from aim2dat.aiida_workflows.cp2k.base_opt_work_chain import _BaseOptimizationWorkChain


class GeoOptWorkChain(_BaseOptimizationWorkChain):
    """
    AiiDA work chain to optimize the atomic positions.
    """

    _initial_scf_guess = "RESTART"
