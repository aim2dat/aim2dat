"""
Aiida work chains for cp2k to find parameters that converge the Kohn-Sham equations.
"""

# Third party library imports
import aiida.orm as aiida_orm
from aiida.engine import (
    process_handler,
    ExitCode,
)

# Internal library imports
from aim2dat.aiida_workflows.cp2k.base_core_work_chain import _BaseCoreWorkChain
from aim2dat.aiida_workflows.cp2k.core_work_chain_handlers import _switch_to_atomic_scf_guess
from aim2dat.utils.dict_tools import (
    dict_retrieve_parameter,
    dict_set_parameter,
    dict_create_tree,
)


class EigenvaluesWorkChain(_BaseCoreWorkChain):
    """
    AiiDA work chain to calculate the band structure with cp2k.
    """

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

    def setup_wc_specific_inputs(self):
        """Set k-points to full grid and the print statement."""
        self.ctx.inputs.metadata.options.parser_name = "aim2dat.cp2k.standard"
        self.ctx.adj_kpoints = True
        self.ctx.always_add_unocc_states = True
        parameters = self.ctx.inputs.parameters.get_dict()
        # Add the print-command for eigenvalues:
        extra_sections = {"MO": {"EIGENVALUES": True, "EACH": {"QS_SCF": 1000}}}
        dict_create_tree(parameters, ["FORCE_EVAL", "DFT", "PRINT"])
        parameters["FORCE_EVAL"]["DFT"]["PRINT"].update(extra_sections)

        # Change k-point parallelization to print all k-points:
        kpoints = dict_retrieve_parameter(parameters, ["FORCE_EVAL", "DFT", "KPOINTS"])
        if kpoints is not None:
            dict_set_parameter(parameters, ["FORCE_EVAL", "DFT", "KPOINTS", "FULL_GRID"], False)
            dict_set_parameter(
                parameters, ["FORCE_EVAL", "DFT", "KPOINTS", "PARALLEL_GROUP_SIZE"], 0
            )
        self.ctx.inputs.parameters = aiida_orm.Dict(dict=parameters)
        self.ctx.inputs.settings = aiida_orm.Dict(dict={"output_check_scf_conv": True})

    @process_handler(
        priority=402,
        exit_codes=ExitCode(0),
    )
    def switch_to_atomic_scf_guess(self, calc):
        """
        Switch to atomic guess for the case that the scf-cycles do not converge.
        """
        return self._execute_error_handler(calc, _switch_to_atomic_scf_guess)
