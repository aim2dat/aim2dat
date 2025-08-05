"""
Aiida work chain for cp2k to calculate cube files.
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
from aim2dat.utils.dict_tools import dict_create_tree

_supported_cube_types = [
    "efield",
    "elf",
    "external_potential",
    "e_density",
    "tot_density",
    "v_hartree",
    "v_xc",
]


def _validate_cube_types(value, _):
    value = value.get_list()
    if not all(val0 in _supported_cube_types for val0 in value):
        return "Only '" + "', '".join(_supported_cube_types) + "' are supported cube types."
    if len(set(value)) != len(value):
        return "Field types cannot be calculated twice."


def _setup_wc_specific_inputs(ctx, inputs):
    parameters = ctx.inputs.parameters.get_dict()
    extra_sections = {}
    for cube_type in inputs.cube_types.get_list():
        extra_sections[cube_type.upper() + "_CUBE"] = {"STRIDE": 1}
    dict_create_tree(parameters, ["FORCE_EVAL", "DFT", "PRINT"])
    parameters["FORCE_EVAL"]["DFT"]["PRINT"].update(extra_sections)
    ctx.inputs.parameters = aiida_orm.Dict(dict=parameters)
    ctx.inputs.settings = aiida_orm.Dict(
        dict={"additional_retrieve_temporary_list": ["*.cube"], "output_check_scf_conv": True}
    )


class CubeWorkChain(_BaseCoreWorkChain):
    """
    AiiDA work chain to calculate and store cube files using CP2K.
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
        spec.input(
            "always_add_unocc_states",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(False),
            help="Always include some unoccupied states even if smearing is not used.",
        )
        spec.input(
            "cube_types",
            valid_type=aiida_orm.List,
            validator=_validate_cube_types,
            help="List of cubes that are calculated and stored.",
        )

    def setup_wc_specific_inputs(self):
        """Add print commands for the cube files."""
        _setup_wc_specific_inputs(self.ctx, self.inputs)

    @process_handler(
        priority=402,
        exit_codes=ExitCode(0),
    )
    def switch_to_atomic_scf_guess(self, calc):
        """
        Switch to atomic guess for the case that the scf-cycles do not converge.
        """
        return self._execute_error_handler(calc, _switch_to_atomic_scf_guess)
