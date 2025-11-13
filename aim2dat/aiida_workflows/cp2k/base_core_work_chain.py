"""Base core work chains."""

# Standard library imports
import os

# Third party library imports
import aiida.orm as aiida_orm
from aiida.engine import (
    while_,
    BaseRestartWorkChain,
    process_handler,
    ExitCode,
)
from aiida.plugins import CalculationFactory
from aiida.common import AttributeDict

# Internal library imports
from aim2dat.aiida_workflows.cp2k.auxiliary_functions import (
    return_scf_parameters,
    return_runtime_stats,
)
from aim2dat.aiida_workflows.cp2k.core_work_chain_inputs import _set_input_parameters
from aim2dat.aiida_workflows.cp2k.core_work_chain_scf import (
    _initialize_scf_parameters,
    _compare_scf_p,
)
from aim2dat.aiida_workflows.cp2k.work_chain_specs import (
    structural_p_specs,
    numerical_p_specs,
    core_work_chain_exit_codes,
)
from aim2dat.aiida_workflows.cp2k.core_work_chain_handlers import (
    _resubmit_calculation,
    _switch_scf_parameters,
    _switch_to_broyden_mixing,
    _switch_to_open_shell_ks,
)


cwd = os.path.dirname(__file__)
Cp2kCalculation = CalculationFactory("aim2dat.cp2k")


def _validate_input_scf_method(scf_method, _):
    """Validate scf-method input."""
    if scf_method.value.split("-")[0] + "_p.yaml" not in os.listdir(cwd + "/scf_parameter_files/"):
        return "Unsupported value for 'scf_method'."


class _BaseCoreWorkChain(BaseRestartWorkChain):
    _smearing_levels = [0.0, 250.0, 500.0, 1000.0]
    _allowed_system_character = ["metallic", "insulator", "unknown"]
    _conv_warning = "One or more SCF run did not converge."
    _process_class = Cp2kCalculation
    _high_verbosity = False
    _keep_scf_method_fixed = False
    _keep_smearing_fixed = False
    _initial_scf_guess = "ATOMIC"

    @classmethod
    def define(cls, spec):
        """
        Specify inputs, outputs and the workflow.
        """
        super().define(spec)
        spec = structural_p_specs(spec)
        spec = numerical_p_specs(spec)
        spec.input(
            "max_iterations",
            valid_type=aiida_orm.Int,
            default=lambda: aiida_orm.Int(100),
            help="Maximum number of iterations the work chain will restart the process "
            "to finish successfully.",
        )
        spec.input(
            "factor_unocc_states",
            valid_type=aiida_orm.Float,
            default=lambda: aiida_orm.Float(0.75),
            help="Factor to determine the number of unoccupied orbitals.",
        )
        spec.input(
            "scf_method",
            valid_type=aiida_orm.Str,
            default=lambda: aiida_orm.Str("density_mixing"),
            validator=_validate_input_scf_method,
            help="Method used to converge the SCF-cycles, options are: `density_mixing` or "
            "`orbital_transformation`",
        )
        spec.input(
            "custom_scf_method",
            valid_type=aiida_orm.List,
            required=False,
            help="Custom set of parameters used to converge the Kohn-Sham equations.",
        )
        spec.input(
            "scf_extended_system",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(False),
            help="Apply larger number of scf-cycles for extended systems.",
        )
        spec.input(
            "enable_roks",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(False),
            help="Use restricted open-shell instead of unrestricted open-shell calculations.",
        )
        spec.input(
            "disable_cholesky",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(False),
            help="Disables cholesky method used for computing the inverse of S.",
        )
        spec.input(
            "numerical_p.kpoints_ref_dist",
            valid_type=aiida_orm.Float,
            help="Reference distance between two k-points in reciprocal space.",
            required=False,
        )
        spec.input(
            "numerical_p.cutoff_radius",
            valid_type=aiida_orm.Float,
            help="Cutoff radius (in Angstroms) for the truncated 1/r potential."
            "Only valid when doing truncated calculation.",
            required=False,
        )
        spec.input(
            "numerical_p.max_memory",
            valid_type=aiida_orm.Int,
            help="Defines the maximum amount of memory [MiB]"
            "to be consumed by the full HFX module.",
            required=False,
        )
        spec.expose_inputs(
            Cp2kCalculation, namespace="cp2k", exclude=["structure", "basissets", "pseudos"]
        )
        spec.output(
            "scf_parameters",
            valid_type=aiida_orm.Dict,
            required=True,
            help="Information on the SCF-Parameters that converge the Kohn-Sham equations.",
        )
        spec.output(
            "run_time_stats",
            valid_type=aiida_orm.Dict,
            required=True,
            help="Information on the run time of the work chain.",
        )
        spec.expose_outputs(
            Cp2kCalculation,
            namespace="cp2k",
            namespace_options={"help": "Output parameters of CP2K."},
        )
        spec = core_work_chain_exit_codes(spec)
        spec.outline(
            cls.setup_inputs,
            cls.setup_wc_specific_inputs,
            cls.initialize_scf_parameters,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.post_processing,
            cls.wc_specific_post_processing,
        )

    def setup_inputs(self):
        super().setup()
        self.ctx.inputs = AttributeDict(self.exposed_inputs(Cp2kCalculation, "cp2k"))
        error = _set_input_parameters(
            self.inputs,
            self.ctx,
            self._keep_scf_method_fixed,
            self._keep_smearing_fixed,
            self._smearing_levels,
            self._initial_scf_guess,
        )
        if error is not None:
            error_f = getattr(self.exit_codes, error[0])
            error_f(**error[1])

    def setup_wc_specific_inputs(self):
        pass

    def initialize_scf_parameters(self):
        error, reports = _initialize_scf_parameters(self.inputs, self.ctx)
        for report in reports:
            self.report(report)
        if error is not None:
            error_f = getattr(self.exit_codes, error[0])
            error_f(**error[1])

    def post_processing(self):
        # super().results()
        if "odd_kpoints" not in self.ctx.cur_scf_p:
            self.report(
                f"SCF method level {self.ctx.cur_scf_p['method_level']} with parameter level "
                f"{self.ctx.cur_scf_p['parameter_level']}, smearing level and "
                f"{self.ctx.cur_scf_p['smearing_level']}."
            )
        elif self.ctx.cur_scf_p["odd_kpoints"]:
            self.report(
                f"SCF method level {self.ctx.cur_scf_p['method_level']} with parameter level "
                f"{self.ctx.cur_scf_p['parameter_level']}, smearing level "
                f"{self.ctx.cur_scf_p['smearing_level']} and odd nr of kpoints chosen."
            )
        else:
            self.report(
                f"SCF method level {self.ctx.cur_scf_p['method_level']} with parameter level "
                f"{self.ctx.cur_scf_p['parameter_level']}, smearing level "
                f"{self.ctx.cur_scf_p['smearing_level']} and even nr of kpoints chosen."
            )
        if "scf_parameters" in self.inputs.structural_p and _compare_scf_p(
            self.inputs.structural_p.scf_parameters.get_dict(), self.ctx.cur_scf_p
        ):
            self.out("scf_parameters", self.inputs.structural_p.scf_parameters)
        else:
            scf_p_output = return_scf_parameters(aiida_orm.Dict(dict=self.ctx.cur_scf_p))
            self.out("scf_parameters", scf_p_output)

        output_parameters = {
            "calc_" + str(idx): calc_j.outputs["output_parameters"]
            for idx, calc_j in enumerate(self.ctx.children)
        }
        self.out("run_time_stats", return_runtime_stats(**output_parameters))
        self.out_many(
            self.exposed_outputs(self.ctx.children[-1], Cp2kCalculation, namespace="cp2k")
        )
        output_p_dict = self.ctx.children[-1].outputs["output_parameters"].get_dict()
        if not output_p_dict["scf_converged"]:
            return self.exit_codes.ERROR_SCF_CONVERGENCE_NOT_REACHED

    def on_terminated(self):
        """Clean working directories of the calculations."""
        super().on_terminated()

        if self.inputs.clean_workdir.value:
            self.report("Remote folders will be cleaned.")

            cleaned_calcs = []

            for called_descendant in self.node.called_descendants:
                if isinstance(called_descendant, aiida_orm.CalcJobNode):
                    try:
                        called_descendant.outputs.remote_folder._clean()
                        cleaned_calcs.append(called_descendant.pk)
                    except (IOError, OSError, KeyError):
                        pass

            if cleaned_calcs:
                self.report(f"{', '.join(map(str, cleaned_calcs))} calculations hve been cleaned.")

    def wc_specific_post_processing(self):
        pass

    @process_handler(priority=402, exit_codes=ExitCode(310))
    def resubmit_calculation(self, calc):
        """Resubmit in case the calculation did not start."""
        return self._execute_error_handler(calc, _resubmit_calculation)

    @process_handler(
        priority=401,
        exit_codes=[
            ExitCode(0),
            ExitCode(400),
            ExitCode(401),
            ExitCode(404),
            ExitCode(405),
            ExitCode(501),
        ],
    )
    def check_scf_convergence(self, calc):
        """
        Check if the scf-calculation is converged and increments the
        internal level of mixing parameters.
        """
        return self._execute_error_handler(calc, _switch_scf_parameters)

    @process_handler(priority=401, exit_codes=ExitCode(402))
    def switch_to_open_shell_ks(self, calc):
        """
        Turn on restricted or unrestricted open-shell Kohn-Sham equations in case of an odd
        number of electrons.
        """
        return self._execute_error_handler(calc, _switch_to_open_shell_ks)

    @process_handler(priority=402, exit_codes=ExitCode(405))
    def switch_to_broyden_mixing(self, calc):
        """
        Switch to the Broyden mixing scheme in case of numerical instabilities with the Pulay
        mixing scheme.
        """
        return self._execute_error_handler(calc, _switch_to_broyden_mixing)

    def _execute_error_handler(self, calc, handler_function):
        proc_handler_report, reports = handler_function(
            self.inputs, self.ctx, self.exit_codes, calc
        )
        for rep in reports:
            self.report(rep)
        return proc_handler_report
