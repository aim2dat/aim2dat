"""Deprecated module redirecting to the new wfl sub-package."""

# Third party library imports
import aiida.orm as aiida_orm

# Internal library imports
from aim2dat.aiida_workflows.cp2k.base_core_work_chain import _BaseCoreWorkChain
from aim2dat.aiida_workflows.cp2k.core_work_chain_handlers import _switch_to_atomic_scf_guess
from aim2dat.utils.dict_tools import dict_set_parameter

from aiida.engine import process_handler, ExitCode


class ForceWorkChain(_BaseCoreWorkChain):
    """
    AiiDA work chain for a single-point CP2K energy+forces evaluation.
    """

    _keep_scf_method_fixed = False
    _keep_smearing_fixed = False
    _initial_scf_guess = "ATOMIC"

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.input(
            "adjust_scf_parameters",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(False),
            help="Restart with adjusted parameters if the SCF-cycles do not converge.",
        )

    def setup_wc_specific_inputs(self):
        """Inject the ENERGY_FORCE run type and enable the forces print block."""
        parameters = self.ctx.inputs.parameters.get_dict()
        dict_set_parameter(parameters, ["GLOBAL", "RUN_TYPE"], "ENERGY_FORCE")
        # Print the atomic forces into the CP2K output (read later by phonopy).
        dict_set_parameter(parameters, ["FORCE_EVAL", "PRINT", "FORCES", "_"], "ON")
        # If multiple calculations are run the storage gets full quickly.
        # Also the WFN/KP is not needed for further processes.
        dict_set_parameter(
            parameters, ["FORCE_EVAL", "DFT", "SCF", "PRINT", "RESTART", "_"], "OFF"
        )
        self.ctx.inputs.parameters = aiida_orm.Dict(dict=parameters)
        self.ctx.inputs.settings = aiida_orm.Dict(dict={"output_check_scf_conv": True})

    @process_handler(priority=402, exit_codes=ExitCode(0))
    def switch_to_atomic_scf_guess(self, calc):
        """Switch to an atomic guess if the SCF-cycles do not converge."""
        return self._execute_error_handler(calc, _switch_to_atomic_scf_guess)
