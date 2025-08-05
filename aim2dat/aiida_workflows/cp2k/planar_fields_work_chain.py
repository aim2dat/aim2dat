"""
Aiida work chains for cp2k to calculate 2d fields based on the electron density.
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
from aim2dat.utils.dict_tools import dict_create_tree

Critic2Calculation = CalculationFactory("aim2dat.critic2")


def _validate_plane_vectors(value, _):
    value = value.get_list()
    if len(value) != 11:
        return "`plane_vectors` must have 11 entries."
    if not all(isinstance(val0, (float, int)) for val0 in value[:9]):
        return "First nine entries of `plane_vectors` need to be of type int or float."
    if not all(isinstance(val0, int) for val0 in value[9:]):
        return "Last two entries of `plane_vectors` need to be of type int."


def _validate_field_types(value, _):
    value = value.get_list()
    if not all(val0 in ["elf", "deformation_density", "total_density"] for val0 in value):
        return "Only 'elf', 'deformation_density' and 'total_density' are supported field types."
    if len(set(value)) != len(value):
        return "Field types cannot be calculated twice."


def _create_critic2_input_parameters(f_type, plane, structure):
    system = "molecule"
    if any(structure.pbc):
        system = "crystal"
    if f_type == "deformation_density":
        parameters = [
            system + " aiida-ELECTRON_DENSITY-1_0.cube",
            "load aiida-ELECTRON_DENSITY-1_0.cube core zpsp",
            'load as "$1-$0"',
            "plane " + " ".join(str(val) for val in plane) + " field 2 file rhodef",
        ]
    elif f_type == "elf":
        parameters = [
            system + " aiida-ELF_S1-1_0.cube",
            "load aiida-ELF_S1-1_0.cube",
            "plane " + " ".join(str(val) for val in plane) + " field 1 file elf",
        ]
    elif f_type == "total_density":
        parameters = [
            system + " aiida-TOTAL_DENSITY-1_0.cube",
            "load aiida-TOTAL_DENSITY-1_0.cube",
            "plane " + " ".join(str(val) for val in plane) + " field 1 file total_density",
        ]
    return aiida_orm.List(list=parameters)


class PlanarFieldsWorkChain(_BaseCoreWorkChain):
    """AiiDA work chain to calculate the planar fields."""

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
            help="Whether to store the cube files that are input for the planes.",
        )
        spec.input(
            "plane_vectors",
            valid_type=aiida_orm.List,
            validator=_validate_plane_vectors,
            help="Plane vectors and number of points given in crystallographic coordinates and "
            "angstrom for (partial) periodic boundary conditions and non-periodic boundary "
            "conditions, respectively.",
        )
        spec.input(
            "field_types",
            valid_type=aiida_orm.List,
            validator=_validate_field_types,
            default=lambda: aiida_orm.List(list=["deformation_density", "elf", "total_density"]),
            help="The field type that is calculated, up to now only 'deformation_density' and "
            "'elf' are supported.",
        )
        spec.expose_inputs(
            Critic2Calculation,
            namespace="critic2",
            exclude=("charge_density_folder", "kind_info", "parameters"),
            namespace_options={
                "required": False,
                "populate_defaults": False,
                "help": "Input parameters of critic2.",
            },
        )
        spec.expose_outputs(
            Critic2Calculation,
            "critic2.total_density",
            namespace_options={
                "required": False,
                "help": "Critic2 outputs of the total density plane calculation.",
            },
        )
        spec.expose_outputs(
            Critic2Calculation,
            "critic2.deformation_density",
            namespace_options={
                "required": False,
                "help": "Critic2 outputs of the deformation density plane calculation.",
            },
        )
        spec.expose_outputs(
            Critic2Calculation,
            "critic2.elf",
            namespace_options={
                "required": False,
                "help": "Critic2 outputs of the elf plane calculation.",
            },
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
            cls.setup_critic2_calculation,
            cls.wc_specific_post_processing,
        )

    def setup_wc_specific_inputs(self):
        """Set input parameters to calculate partial charges."""
        self.ctx.inputs.metadata.options.parser_name = "aim2dat.cp2k.partial_charges"
        self.ctx.field_types = self.inputs.field_types.get_list()
        parameters = self.ctx.inputs.parameters.get_dict()
        extra_sections = {}
        if "deformation_density" in self.ctx.field_types:
            extra_sections["E_DENSITY_CUBE"] = {"STRIDE": 1}
        if "elf" in self.ctx.field_types:
            extra_sections["ELF_CUBE"] = {"STRIDE": 1}
        if "total_density" in self.ctx.field_types:
            extra_sections["TOT_DENSITY_CUBE"] = {"STRIDE": 1}
        dict_create_tree(parameters, ["FORCE_EVAL", "DFT", "PRINT"])
        parameters["FORCE_EVAL"]["DFT"]["PRINT"].update(extra_sections)
        self.ctx.inputs.parameters = aiida_orm.Dict(dict=parameters)
        calcjob_settings = {"output_check_scf_conv": True}
        if "store_cubes" in self.inputs:
            calcjob_settings["additional_retrieve_temporary_list"] = ["*.cube"]
        self.ctx.inputs.settings = aiida_orm.Dict(dict=calcjob_settings)

    def setup_critic2_calculation(self):
        """Set input parameters for external post-processing codes."""
        for f_type in self.ctx.field_types:
            inputs = AttributeDict(self.exposed_inputs(Critic2Calculation, "critic2"))
            inputs.charge_density_folder = self.ctx.children[-1].outputs.remote_folder
            inputs.kind_info = self.ctx.children[-1].outputs.output_kind_info
            inputs.parameters = _create_critic2_input_parameters(
                f_type, self.inputs.plane_vectors.get_list(), self.ctx.inputs.structure
            )
            running = self.submit(Critic2Calculation, **inputs)
            self.report(f"Launching {f_type} field:  <{running.pk}>.")
            self.to_context(**{f_type: running})

    def wc_specific_post_processing(self):
        """Expose outputs of the external codes."""
        for f_type in self.ctx.field_types:
            critic2_calc = getattr(self.ctx, f_type)
            if not critic2_calc.is_finished_ok:
                return self.exit_codes.ERROR_CALCULATION_ABORTED
            self.out_many(
                self.exposed_outputs(
                    getattr(self.ctx, f_type), Critic2Calculation, namespace="critic2." + f_type
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
