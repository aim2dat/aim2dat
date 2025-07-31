"""
AiiDA work chain to optimize the atomic positions and the unit cell using CP2K.
"""

# Third party library imports
import aiida.orm as aiida_orm

# Internal library imports
from aim2dat.aiida_workflows.cp2k.base_opt_work_chain import _BaseOptimizationWorkChain
from aim2dat.utils.dict_tools import (
    dict_set_parameter,
    dict_create_tree,
)


def _validate_ref_cell_scaling_factor(ref_cell_scaling_factor, _):
    """Validate scf-method input."""
    if ref_cell_scaling_factor.value <= 0.0:
        return "'optimization_p.ref_cell_scaling_factor' needs to be larger than 0.0."


class CellOptWorkChain(_BaseOptimizationWorkChain):
    """
    AiiDA work chain to optimize the unit cell of a periodic system.
    """

    _opt_type = "cell_opt"

    @classmethod
    def define(cls, spec):
        """
        Specify inputs, outputs and the workflow.
        """
        super().define(spec)
        spec.input(
            "optimization_p.keep_symmetry",
            valid_type=aiida_orm.Bool,
            required=False,
            help="Constrain the lattice symmetry during cell optimization.",
        )
        spec.input(
            "optimization_p.cell_symmetry",
            valid_type=aiida_orm.Str,
            required=False,
            help="The applied cell symmetry which is constrained during cell optimization.",
        )
        spec.input(
            "optimization_p.keep_angles",
            valid_type=aiida_orm.Bool,
            required=False,
            help="Constrain the lattice angles during cell optimization.",
        )
        spec.input(
            "optimization_p.pressure_tolerance",
            valid_type=aiida_orm.Float,
            required=False,
            help="Specifies the Pressure tolerance to achieve during the cell optimization.",
        )
        spec.input(
            "optimization_p.ref_cell_scaling_factor",
            valid_type=aiida_orm.Float,
            validator=_validate_ref_cell_scaling_factor,
            required=False,
            help="Scaling factor for the reference cell (CELL_REF).",
        )

    def setup_wc_specific_inputs(self):
        """Set stress tensor calculation to analytical."""
        # Set stress tensor for cell optimization and add symmetry constraints:
        parameters = self.ctx.inputs.parameters.get_dict()
        dict_create_tree(parameters, ["FORCE_EVAL"])
        dict_set_parameter(parameters, ["FORCE_EVAL", "STRESS_TENSOR"], "ANALYTICAL")
        self.ctx.inputs.parameters = aiida_orm.Dict(dict=parameters)
