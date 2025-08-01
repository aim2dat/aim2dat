"""
Aiida work chains for cp2k to find parameters that converge the Kohn-Sham equations.
"""

# Third party library imports
import aiida.orm as aiida_orm
from aiida.plugins import CalculationFactory
from aiida.engine import (
    while_,
    process_handler,
    ExitCode,
)
from aiida.common import AttributeDict

# Internal library imports
from aim2dat.aiida_workflows.cp2k.base_core_work_chain import _BaseCoreWorkChain
from aim2dat.aiida_workflows.cp2k.core_work_chain_handlers import _switch_to_atomic_scf_guess
from aim2dat.utils.dict_tools import dict_set_parameter, dict_create_tree

Critic2Calculation = CalculationFactory("aim2dat.critic2")
ChargemolCalculation = CalculationFactory("aim2dat.chargemol")


class PartialChargesWorkChain(_BaseCoreWorkChain):
    """AiiDA work chain to calculate the partial charges of the atoms."""

    _keep_scf_method_fixed = True
    _keep_smearing_fixed = True
    _initial_scf_guess = "RESTART"

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.input(
            "adjust_scf_parameters",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(False),
            help="Restart calculation with adjusted parameters if SCF-cycles are not converged.",
        )
        spec.input(
            "always_add_unocc_states",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(False),
            help="Always include some unoccupied states even if smearing is not used.",
        )
        spec.input(
            "store_cubes",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(False),
            help="Whether to store the electron density cube file (only supported for chargemol "
            "and critic2).",
        )
        spec.expose_inputs(
            Critic2Calculation,
            namespace="critic2",
            exclude=("charge_density_folder", "kind_info"),
            namespace_options={"required": False, "populate_defaults": False, "help": "..."},
        )
        spec.expose_inputs(
            ChargemolCalculation,
            namespace="chargemol",
            exclude=("charge_density_folder", "charge_density_filename", "kind_info"),
            namespace_options={"required": False, "populate_defaults": False, "help": "..."},
        )
        spec.expose_outputs(
            Critic2Calculation,
            namespace="critic2",
            namespace_options={"required": False, "help": "Output parameters of critic2."},
        )
        spec.expose_outputs(
            ChargemolCalculation,
            namespace="chargemol",
            namespace_options={"required": False, "help": "Output parameters of chargemol."},
        )
        spec.outline(
            cls.setup_inputs,
            cls.setup_wc_specific_inputs,
            cls.initialize_scf_parameters,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.post_processing,
            cls.setup_external_partial_charge_analysis,
            cls.wc_specific_post_processing,
        )

    def setup_wc_specific_inputs(self):
        """Set input parameters to calculate partial charges."""
        self.ctx.inputs.metadata.options.parser_name = "aim2dat.cp2k.partial_charges"
        parameters = self.ctx.inputs.parameters.get_dict()
        dict_create_tree(parameters, ["GLOBAL"])
        dict_set_parameter(parameters, ["GLOBAL", "PRINT_LEVEL"], "MEDIUM")
        if "critic2" in self.inputs or "chargemol" in self.inputs:
            extra_sections = {"E_DENSITY_CUBE": {"STRIDE": 1}}
            dict_create_tree(parameters, ["FORCE_EVAL", "DFT", "PRINT"])
            parameters["FORCE_EVAL"]["DFT"]["PRINT"].update(extra_sections)
        calcjob_settings = {"output_check_scf_conv": True}
        if "store_cubes" in self.inputs:
            calcjob_settings["additional_retrieve_temporary_list"] = ["*.cube"]
        self.ctx.inputs.settings = aiida_orm.Dict(dict=calcjob_settings)
        self.ctx.inputs.parameters = aiida_orm.Dict(dict=parameters)

    def setup_external_partial_charge_analysis(self):
        """Set input parameters for external post-processing codes."""
        if "critic2" in self.inputs:
            inputs = AttributeDict(self.exposed_inputs(Critic2Calculation, "critic2"))
            inputs.charge_density_folder = self.ctx.children[-1].outputs.remote_folder
            inputs.kind_info = self.ctx.children[-1].outputs.output_kind_info
            running = self.submit(Critic2Calculation, **inputs)
            self.report(f"Launching  <{running.pk}>.")
            self.to_context(critic2=running)
        if "chargemol" in self.inputs:
            inputs = AttributeDict(self.exposed_inputs(ChargemolCalculation, "chargemol"))
            inputs.charge_density_folder = self.ctx.children[-1].outputs.remote_folder
            inputs.charge_density_filename = aiida_orm.Str("aiida-ELECTRON_DENSITY-1_0.cube")
            inputs.kind_info = self.ctx.children[-1].outputs.output_kind_info
            running = self.submit(ChargemolCalculation, **inputs)
            self.report(f"Launching  <{running.pk}>.")
            self.to_context(chargemol=running)

    def wc_specific_post_processing(self):
        """Expose outputs of the external codes."""
        if "critic2" in self.inputs:
            self.out_many(
                self.exposed_outputs(self.ctx.critic2, Critic2Calculation, namespace="critic2")
            )
        if "chargemol" in self.inputs:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.chargemol, ChargemolCalculation, namespace="chargemol"
                )
            )

    @process_handler(
        priority=402,
        exit_codes=ExitCode(0),
    )
    def switch_to_atomic_scf_guess(self, calc):
        """
        Switch to atomic guess for the case that the scf-cycles do not converge.
        """
        return self._execute_error_handler(calc, _switch_to_atomic_scf_guess)
