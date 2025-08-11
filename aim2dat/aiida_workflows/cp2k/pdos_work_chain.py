"""
Aiida work chains for cp2k to find parameters that converge the Kohn-Sham equations.
"""

# Standard library imports
import math

# Third party library imports
import aiida.orm as aiida_orm
from aiida.plugins import DataFactory

# Internal library imports
from aim2dat.aiida_workflows.cp2k.base_core_work_chain import _BaseCoreWorkChain
from aim2dat.aiida_workflows.cp2k.auxiliary_functions import calculate_added_mos
from aim2dat.utils.dict_tools import (
    dict_retrieve_parameter,
    dict_set_parameter,
    dict_create_tree,
)

StructureData = DataFactory("core.structure")


class PDOSWorkChain(_BaseCoreWorkChain):
    """
    AiiDA work chain to calculate the projected density of states.
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
            help="Restart calculation with adjusted parameters if SCF-cycles are not converged.",
        )
        spec.input(
            "minimum_cell_length",
            valid_type=aiida_orm.Float,
            default=lambda: aiida_orm.Float(25.0),
        )
        spec.input(
            "maximum_cell_length",
            valid_type=aiida_orm.Float,
            default=lambda: aiida_orm.Float(40.0),
        )
        spec.input(
            "resolve_atoms", valid_type=aiida_orm.Bool, default=lambda: aiida_orm.Bool(False)
        )
        spec.input(
            "ldos",
            valid_type=aiida_orm.List,
            required=False,
            help="Print the local PDOS, projected on subsets of atoms given through lists.",
        )
        spec.input(
            "wfn_n_homo",
            valid_type=aiida_orm.Int,
            default=lambda: aiida_orm.Int(0),
            help="Print a certain of occupied orbitals as cube data.",
        )
        spec.input(
            "wfn_n_lumo",
            valid_type=aiida_orm.Int,
            default=lambda: aiida_orm.Int(0),
            help="Print a certain of unoccupied orbitals as cube data.",
        )
        spec.input(
            "wfn_cube_list",
            valid_type=aiida_orm.List,
            required=False,
            help="List of orbitals stored as cube data.",
        )

    def setup_wc_specific_inputs(self):
        """Derive super cell and set input parameter."""
        self.ctx.inputs.metadata.options.parser_name = "aim2dat.cp2k.standard"
        self.ctx.scf_m_info["always_add_unocc_states"] = True
        parameters = self.ctx.inputs.parameters.get_dict()

        # Calculate the number of repetition in x, y, z and
        min_cell_length = self.inputs.minimum_cell_length.value
        self.ctx.mult_unit_cell = [1, 1, 1]
        cell_lengths = self.ctx.inputs.structure.cell_lengths
        for dir_idx in range(3):
            while min_cell_length > cell_lengths[dir_idx] * self.ctx.mult_unit_cell[dir_idx]:
                self.ctx.mult_unit_cell[dir_idx] += 1

        mult_unit_cell_str = " ".join([str(mult0) for mult0 in self.ctx.mult_unit_cell])
        dict_create_tree(parameters, ["FORCE_EVAL", "SUBSYS", "CELL"])
        dict_set_parameter(
            parameters, ["FORCE_EVAL", "SUBSYS", "CELL", "MULTIPLE_UNIT_CELL"], mult_unit_cell_str
        )
        dict_create_tree(parameters, ["FORCE_EVAL", "SUBSYS", "TOPOLOGY"])
        dict_set_parameter(
            parameters,
            ["FORCE_EVAL", "SUBSYS", "TOPOLOGY", "MULTIPLE_UNIT_CELL"],
            mult_unit_cell_str,
        )
        self.ctx.scf_m_info["factor_unocc_states"] *= self.ctx.scf_m_info[
            "factor_unocc_states"
        ] * math.prod(self.ctx.mult_unit_cell)
        # dict_create_tree(parameters, ["FORCE_EVAL", "DFT", "SCF"])
        # dict_set_parameter(parameters, ["FORCE_EVAL", "DFT", "SCF", "ADDED_MOS"], n_unocc_states)

        self.report(f"{self.ctx.mult_unit_cell} repetitions chosen.")
        self.report(f"{self.ctx.scf_m_info['factor_unocc_states']} factor for unoccupied states.")

        # Delete k-points section in input-parameters:
        kpoints_p = dict_retrieve_parameter(parameters, ["FORCE_EVAL", "DFT", "KPOINTS"])
        if kpoints_p is not None:
            del parameters["FORCE_EVAL"]["DFT"]["KPOINTS"]
        parameters["GLOBAL"]["PRINT_LEVEL"] = "LOW"

        # Add pdos-files to temporary retrieve-list
        calcjob_settings = {
            "additional_retrieve_temporary_list": ["*.pdos"],
            "output_check_scf_conv": True,
        }

        # Add the print-command for pdos and cubes:
        extra_sections = {"PDOS": {"COMPONENTS": True, "NLUMO": -1}}
        if self.inputs.resolve_atoms.value:
            structure = self.ctx.inputs.structure
            dict_set_parameter(
                extra_sections,
                ["PDOS", "LDOS"],
                [{"LIST": str(i + 1), "COMPONENTS": True} for i in range(len(structure.sites))],
            )
        elif "ldos" in self.inputs:
            ldos_list = self.inputs.ldos.get_list()
            ldos_str_list = []
            if all(isinstance(i, int) for i in ldos_list):
                ldos_list = [ldos_list]
            ldos_list = [[ldos] if isinstance(ldos, int) else ldos for ldos in ldos_list]
            if min(i for ldos in ldos_list for i in ldos) == 0:
                ldos_list = [[i + 1 for i in ldos] for ldos in ldos_list]
            if not all(isinstance(ldos, list) for ldos in ldos_list):
                return self.exit_codes.ERROR_INPUT_WRONG_VALUE
            for ldos in ldos_list:
                ldos_sorted = sorted(ldos)
                if len(ldos) > 4 and ldos_sorted[-1] - ldos_sorted[0] + 1 == len(ldos_sorted):
                    ldos_str = f"{ldos_sorted[0]}..{ldos_sorted[-1]}"
                else:
                    atoms_string = [str(s) for s in ldos]
                    ldos_str = " ".join(atoms_string)
                ldos_str_list.append({"LIST": ldos_str, "COMPONENTS": True})
            dict_set_parameter(
                extra_sections,
                ["PDOS", "LDOS"],
                ldos_str_list,
            )

        if "wfn_cube_list" in self.inputs:
            _, n_electrons = calculate_added_mos(
                self.ctx.inputs.structure,
                parameters,
                self.ctx.scf_m_info["factor_unocc_states"],
                return_n_electrons=True,
            )
            cube_list = self.inputs.wfn_cube_list.get_list()
            n_occ_states = int(0.5 * n_electrons) * math.prod(self.ctx.mult_unit_cell)
            n_homo = 0
            n_lumo = 0
            for wfn_idx in cube_list:
                wfn_idx_sc = wfn_idx * math.prod(self.ctx.mult_unit_cell)
                if (wfn_idx_sc - n_occ_states - 1) < -1 * n_homo:
                    n_homo = -1 * (wfn_idx_sc - n_occ_states - 1)
                elif (wfn_idx_sc - n_occ_states - math.prod(self.ctx.mult_unit_cell) + 1) > n_lumo:
                    n_lumo = wfn_idx_sc - n_occ_states - math.prod(self.ctx.mult_unit_cell) + 1
                    wfn_idx_sc -= math.prod(self.ctx.mult_unit_cell) - 1
                wfn_idx_str = "0" * (5 - len(str(wfn_idx_sc))) + str(wfn_idx_sc)
                calcjob_settings["additional_retrieve_temporary_list"].append(
                    f"aiida-WFN_{wfn_idx_str}_1-1_0.cube"
                )
                calcjob_settings["additional_retrieve_temporary_list"].append(
                    f"aiida-WFN_{wfn_idx_str}_2-1_0.cube"
                )
            extra_sections["MO_CUBES"] = {
                "STRIDE": 1,
                "NHOMO": n_homo,
                "NLUMO": n_lumo,
            }
            extra_sections["MO"] = {"EIGENVALUES": True, "EACH": {"QS_SCF": 1000}}
        elif self.inputs.wfn_n_homo.value + self.inputs.wfn_n_lumo.value > 0:
            extra_sections["MO_CUBES"] = {
                "STRIDE": 1,
                "NHOMO": self.inputs.wfn_n_homo.value,
                "NLUMO": self.inputs.wfn_n_lumo.value,
            }
            extra_sections["MO"] = {"EIGENVALUES": True, "EACH": {"QS_SCF": 1000}}
            calcjob_settings["additional_retrieve_temporary_list"].append("*.cube")
        dict_create_tree(parameters, ["FORCE_EVAL", "DFT", "PRINT"])
        parameters["FORCE_EVAL"]["DFT"]["PRINT"].update(extra_sections)

        # Add pdos-files to retrieve-list, add parameters and set parser:
        self.ctx.inputs.settings = aiida_orm.Dict(dict=calcjob_settings)
        self.ctx.inputs.parameters = aiida_orm.Dict(dict=parameters)
