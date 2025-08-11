"""
Aiida work chains for cp2k to find parameters that converge the Kohn-Sham equations.
"""

# Third party library imports
import aiida.orm as aiida_orm

# Internal library imports
from aim2dat.aiida_workflows.cp2k.base_core_work_chain import _BaseCoreWorkChain
from aim2dat.elements import get_group


class FindSCFParametersWorkChain(_BaseCoreWorkChain):
    """
    AiiDA work chain to find the mixing parameters to converge the Kohn-Sham
    equations of a specific system.
    """

    @classmethod
    def define(cls, spec):
        """
        Specify inputs, outputs and the workflow.
        """
        super().define(spec)
        spec.input(
            "always_add_unocc_states",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(False),
            help="Always include some unoccupied states even if smearing is not used.",
        )
        spec.input(
            "structural_p.system_character",
            valid_type=aiida_orm.Str,
            required=False,
            help="Electronic character of the system, possible options are 'metallic' or "
            "'insulator'. In case this parameter is set to 'metallic' ('insulator') electronic "
            "smearing is always (never) applied.",
        )

    # def should_run_process(self):
    #     """Condition to restart the calculation with a different set of parameters."""
    #     return not self.ctx.is_finished

    def setup_wc_specific_inputs(self):
        """Check whether to add unoccupied states and the system character."""
        if "system_character" in self.inputs.structural_p:
            for sys_char in ["metallic", "insulator"]:
                if self.inputs.structural_p.system_character.value.upper() == sys_char.upper():
                    self.ctx.scf_m_info["system_character"] = sys_char
        elif all(
            kind in get_group("metal")
            for kind in self.inputs.structural_p.structure.get_kind_names()
        ):
            self.ctx.scf_m_info["system_character"] = "metallic"
        elif "band_gap" in self.ctx.inputs.structure.attributes:
            band_gap = self.ctx.inputs.structure.get_attribute("band_gap", None)
            if band_gap is not None:
                if band_gap < 0.025:
                    self.ctx.scf_m_info["system_character"] = "metallic"
                elif band_gap > 0.25:
                    self.ctx.scf_m_info["system_character"] = "insulator"
