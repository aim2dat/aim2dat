"""
Aiida work chains for cp2k to calculate the band structure.
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
from aim2dat.aiida_workflows.cp2k.auxiliary_functions import calculate_added_mos
from aim2dat.aiida_workflows.utils import seekpath_structure_analysis
from aim2dat.utils.dict_tools import dict_create_tree
from aim2dat.aiida_workflows.cp2k.work_chain_specs import seekpath_p_specs


class BandStructureWorkChain(_BaseCoreWorkChain):
    """
    AiiDA work chain to calculate the band structure using CP2K.
    """

    _keep_scf_method_fixed = True
    _keep_smearing_fixed = True
    _initial_scf_guess = "RESTART"

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec = seekpath_p_specs(spec)
        spec.input(
            "adjust_scf_parameters",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(False),
            help="Restart calculation with adjusted parameters if SCF-cycles are not converged.",
        )
        spec.input(
            "seekpath_parameters",
            valid_type=aiida_orm.Dict,
            default=lambda: aiida_orm.Dict(
                dict={
                    "reference_distance": 0.015,
                    "symprec": 0.005,
                }
            ),
            help="Additional arguments passed to the SeekPath analysis.",
        )
        spec.input(
            "path_parameters",
            valid_type=aiida_orm.Dict,
            help="Dictionary containing parameters based on SeekPath-output.",
            required=False,
        )

    def setup_wc_specific_inputs(self):
        """Set the k-path for the band structure calculation."""
        # self.ctx.inputs = AttributeDict(self.exposed_inputs(Cp2kCalculation, "cp2k"))
        if "path_parameters" in self.inputs:
            self.report("Using input parameters for the k-path.")
            self.ctx.path_parameters = self.inputs.path_parameters.get_dict()
        else:
            self.report("Run SeekPath to determine the k-path.")
            # Create k-path with seekpath
            kpoints_path = seekpath_structure_analysis(
                self.inputs.structural_p.structure, self.inputs.seekpath_parameters
            )
            self.ctx.path_parameters = kpoints_path.get("parameters").get_dict()
            self.ctx.inputs.structure = kpoints_path.get("primitive_structure")

            # Return path parameters:
            self.out("seekpath.path_parameters", kpoints_path.get("parameters"))
            self.out("seekpath.primitive_structure", kpoints_path.get("primitive_structure"))
            self.out("seekpath.conv_structure", kpoints_path.get("conv_structure"))
            self.out("seekpath.explicit_kpoints", kpoints_path.get("explicit_kpoints"))

        self.ctx.adj_kpoints = True
        self.ctx.always_add_unocc_states = True
        parameters = self.ctx.inputs.parameters.get_dict()
        self.ctx.n_unocc_states = calculate_added_mos(
            self.ctx.inputs.structure, parameters, self.inputs.factor_unocc_states.value
        )

        # Parse k-path:
        special_points = self.ctx.path_parameters.get("point_coords")
        paths = self.ctx.path_parameters.get("path")
        segments = self.ctx.path_parameters.get("explicit_segments")
        cp2k_kpoint_path = []
        for path, segment in zip(paths, segments):
            sp_point_start = special_points.get(path[0])
            sp_point_end = special_points.get(path[1])
            cp2k_segment = {
                "UNITS": "B_VECTOR",
                "SPECIAL_POINT": [
                    f"{path[0]} {sp_point_start[0]} {sp_point_start[1]} {sp_point_start[2]}",
                    f"{path[1]} {sp_point_end[0]} {sp_point_end[1]} {sp_point_end[2]}",
                ],
                "NPOINTS": segment[1] - segment[0] - 1,
            }
            cp2k_kpoint_path.append(cp2k_segment)

        cp2k_print = {
            "BAND_STRUCTURE": {
                "KPOINT_SET": cp2k_kpoint_path,
                "ADDED_MOS": self.ctx.n_unocc_states,
            }
        }
        dict_create_tree(parameters, ["FORCE_EVAL", "DFT", "PRINT"])
        parameters["FORCE_EVAL"]["DFT"]["PRINT"].update(cp2k_print)
        self.report(f"Using parser: {self.ctx.inputs.metadata.options.parser_name}")
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
