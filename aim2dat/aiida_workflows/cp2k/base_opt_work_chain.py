"""Base core work chains."""

# Standard library imports
import os

# Third party library imports
import aiida.orm as aiida_orm
from aiida.engine import (
    while_,
    process_handler,
    ExitCode,
)

# Internal library imports
from aim2dat.aiida_workflows.cp2k.base_core_work_chain import _BaseCoreWorkChain
from aim2dat.aiida_workflows.cp2k.core_work_chain_opt import _initialize_opt_parameters
from aim2dat.aiida_workflows.cp2k.core_work_chain_handlers import (
    _resubmit_calculation,
    _resubmit_unconverged_geometry,
)


cwd = os.path.dirname(__file__)


class _BaseOptimizationWorkChain(_BaseCoreWorkChain):
    _opt_type = "geo_opt"
    _keep_scf_method_fixed = True
    _keep_smearing_fixed = True
    _initial_scf_guess = "RESTART"

    @classmethod
    def define(cls, spec):
        """
        Specify inputs, outputs and the workflow.
        """
        super().define(spec)
        spec.input(
            "adjust_scf_parameters",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(False),
            help="Restart calculation with adjusted parameters if SCF-cycles are not converged.",
        )
        spec.input(
            "ignore_convergence_failure",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(False),
            help="If true, only a warning is issued if an SCF iteration has not converged.",
        )
        spec.input(
            "always_add_unocc_states",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(False),
            help="Always include some unoccupied states even if smearing is not used.",
        )
        spec.input_namespace(
            "optimization_p",
            required=False,
            help="Additional optimization parameters set in the CP2K input file.",
        )
        spec.input(
            "optimization_p.max_force",
            valid_type=aiida_orm.Float,
            required=False,
            help="Convergence criterion for the maximum force component.",
        )
        spec.input(
            "optimization_p.rms_force",
            valid_type=aiida_orm.Float,
            required=False,
            help="Convergence criterion for the root mean square (RMS) force.",
        )
        spec.input(
            "optimization_p.max_dr",
            valid_type=aiida_orm.Float,
            required=False,
            help="Convergence criterion for the maximum geometry change.",
        )
        spec.input(
            "optimization_p.rms_dr",
            valid_type=aiida_orm.Float,
            required=False,
            help="Convergence criterion for the root mean square (RMS) geometry change.",
        )
        spec.input(
            "optimization_p.keep_space_group",
            valid_type=aiida_orm.Bool,
            required=False,
            help="Constrain the space group of the structure via spglib.",
        )
        spec.input(
            "optimization_p.eps_symmetry",
            valid_type=aiida_orm.Float,
            required=False,
            help="The tolerance parameter used to determine the space group.",
        )
        spec.input(
            "optimization_p.fixed_atoms",
            valid_type=aiida_orm.List,
            required=False,
            help="Fix atoms.",
        )
        spec.input(
            "custom_opt_method",
            valid_type=aiida_orm.List,
            required=False,
            help="Custom set of parameters used for the optimization.",
        )
        spec.input(
            "initial_opt_parameters",
            valid_type=aiida_orm.Int,
            required=False,
            help="....",
        )
        spec.output(
            "space_group_info",
            valid_type=aiida_orm.Dict,
            required=False,
            help="Information of the constrained space group (in case 'keep_spacegroup' is set).",
        )
        spec.outline(
            cls.setup_inputs,
            cls.setup_wc_specific_inputs,
            cls.initialize_opt_parameters,
            cls.initialize_scf_parameters,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.post_processing,
            cls.wc_specific_post_processing,
            cls.opt_post_processing,
        )

    def should_run_process(self):
        """
        Check conditions whether to run the calculation or not.
        """
        cond_cell_opt = self.ctx.opt_level < len(self.ctx.opt_method_p)
        cond_scf_adjustment = True
        if self.inputs.adjust_scf_parameters.value:
            cond_scf_adjustment = self.ctx.cur_scf_p["method_level"] < len(
                self.ctx.scf_method_p
            ) or self.ctx.cur_scf_p["smearing_level"] < len(self._smearing_levels)
        should_run = (not self.ctx.is_finished) and cond_cell_opt and cond_scf_adjustment
        return should_run

    def initialize_opt_parameters(self):
        _initialize_opt_parameters(self.inputs, self.ctx, self.exit_codes, self._opt_type)

    def opt_post_processing(self):
        if self.ctx.children[-1].is_finished_ok:
            self.report("Geometry converged.")
        else:
            return self.exit_codes.ERROR_OPTIMIZATION_NOT_CONVERGED

    @process_handler(
        priority=402,
        exit_codes=ExitCode(500),
    )
    def resubmit_unconverged_geometry(self, calc):
        """
        Resubmit if geometry is unconverged and choose tighter settings for the
        optimization algorithm.
        """
        return self._execute_error_handler(calc, _resubmit_unconverged_geometry)

    @process_handler(
        priority=402,
        exit_codes=ExitCode(400),
    )
    def _resubmit_calculation(self, calc):
        """Resubmit the geometry in case the walltime is hit."""
        return self._execute_error_handler(calc, _resubmit_calculation)

    # @process_handler(
    #     priority=402,
    #     exit_codes=[ExitCode(0), ExitCode(400), ExitCode(401), ExitCode(405), ExitCode(500)],
    # )
    # def switch_to_atomic_scf_guess(self, calc):
    #     output_p_dict = calc.outputs["output_parameters"].get_dict()
    #     if "scf_converged" not in output_p_dict:
    #         return ProcessHandlerReport(exit_code=self.exit_codes.ERROR_CALCULATION_ABORTED)
    #     if not output_p_dict["scf_converged"] and self.ctx.scf_guess == "RESTART":
    #         self.ctx.scf_guess = "ATOMIC"
    #         self.update_scf_parameters(
    #             self.ctx.cur_scf_p,
    #             self.ctx.cur_scf_p_set["allow_smearing"],
    #             self.ctx.cur_scf_p_set["allow_uks"],
    #         )
    #         return ProcessHandlerReport(do_break=True)

    def set_additional_optimization_p(self, parameters):
        """
        Place holder for additional optimization parameters set in the CP2K input dictionary.

        Parameters
        ----------
        parameters : dict
            Input parameters for the CP2K calculation.
        """
        pass
