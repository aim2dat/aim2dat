"""
Calcjobs for the CP2K software package.
"""

# Standard library imports
import io

# Third party library imports
from aiida.engine import CalcJob
from aiida.plugins import DataFactory
from aiida.common import CalcInfo, CodeInfo, InputValidationError
from aiida.orm import List, Dict, RemoteData, SinglefileData

# Internal library imports
from aim2dat.utils.dict_tools import (
    dict_set_parameter,
    dict_retrieve_parameter,
)

BandsData = DataFactory("core.array.bands")
XyData = DataFactory("core.array.xy")
StructureData = DataFactory("core.structure")


def _transform_dict_to_cp2k(input_dict):
    """Transform python dictionary to a list of strings of cp2k input-parameters."""

    def parse_value(value):
        if isinstance(value, bool):
            if value:
                return ".TRUE."
            else:
                return ".FALSE."
        elif value is None:
            return "NONE"
        else:
            return str(value)

    def recursive_transform(output_list, spaces, key, value):
        if isinstance(value, dict):
            output_list.append(spaces + "&" + key)
            spaces0 = spaces + "  "
            for key0, value0 in value.items():
                recursive_transform(output_list, spaces0, key0, value0)
            output_list.append(spaces + "&END " + key)
        elif isinstance(value, list):
            for value0 in value:
                recursive_transform(output_list, spaces, key, value0)
        elif key == "_" and len(output_list) > 0:
            output_list[-1] += " " + parse_value(value)
        else:
            output_list.append(spaces + key + " " + parse_value(value))

    output_list = []
    spaces = ""
    for key, value in input_dict.items():
        recursive_transform(output_list, spaces, key, value)
    return output_list


def _set_dict_to_upper(input_dict):
    """Set all keywords in dict to upper-case."""

    def recursive_to_upper(input_dict, output_dict):
        for key, value in input_dict.items():
            key_upper = key.upper()
            if isinstance(value, dict):
                output_dict[key_upper] = {}
                recursive_to_upper(value, output_dict[key_upper])
            elif isinstance(value, list):
                output_dict[key_upper] = []
                for value0 in value:
                    if isinstance(value0, dict):
                        output_dict[key_upper].append({})
                        recursive_to_upper(value0, output_dict[key_upper][-1])
                    else:
                        output_dict[key_upper].append(value0)
            else:
                output_dict[key_upper] = value

    output_dict = {}
    recursive_to_upper(input_dict, output_dict)
    return output_dict


class Cp2kCalculation(CalcJob):
    """Calcjob based on the Cp2kCalculation from the official cp2k-plugin."""

    _PROJECT_NAME = "aiida"
    _PARENT_CALC_FLDR_NAME = "parent_calc/"
    _COORDS_FILE_NAME = "aiida.coords.xyz"

    @classmethod
    def define(cls, spec):
        """Define input/output and outline."""
        super().define(spec)
        spec.input("parameters", valid_type=Dict, help="The input parameters.")
        spec.input(
            "structure", valid_type=StructureData, required=False, help="The main input structure."
        )
        spec.input("settings", valid_type=Dict, required=False, help="Optional input parameters.")
        spec.input(
            "parent_calc_folder",
            valid_type=RemoteData,
            required=False,
            help="Working directory of a previously ran calculation to restart from.",
        )
        spec.input_namespace(
            "file",
            valid_type=SinglefileData,
            required=False,
            help="Additional input files.",
            dynamic=True,
        )
        spec.inputs["metadata"]["options"]["input_filename"].default = "aiida.in"
        spec.inputs["metadata"]["options"]["output_filename"].default = "aiida.out"
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
        spec.inputs["metadata"]["options"]["withmpi"].default = True
        spec.inputs["metadata"]["options"]["parser_name"].default = "aim2dat.cp2k.standard"

        spec.output_namespace(
            "output_cubes",
            dynamic=True,
            required=False,
            help="Calculated cubes.",
        )
        spec.output(
            "output_parameters",
            valid_type=Dict,
            required=True,
            help="The output dictionary containing results of the calculation.",
        )
        spec.output(
            "output_structure",
            valid_type=StructureData,
            required=False,
            help="The relaxed output structure.",
        )
        spec.output(
            "output_bands",
            valid_type=BandsData,
            required=False,
            help="Computed electronic band structure.",
        )
        spec.output(
            "output_pdos",
            valid_type=XyData,
            required=False,
            help="Calculated projected density of states.",
        )
        spec.output(
            "output_eigenvalues",
            valid_type=Dict,
            required=False,
            help="Calculated electronic eigenvalues.",
        )
        spec.output(
            "output_kind_info",
            valid_type=List,
            required=False,
            help="Kind information including the number of core/valence electrons for each kind.",
        )
        spec.output(
            "output_motion_step_info",
            valid_type=List,
            required=False,
            help="Information of each step for an optimization or molecular dynamics simulation.",
        )
        spec.output(
            "output_mulliken_populations",
            valid_type=List,
            required=False,
            help="Calculated final Mulliken charges.",
        )
        spec.output(
            "output_hirshfeld_populations",
            valid_type=List,
            required=False,
            help="Calculated final Hirshfeld charges.",
        )
        spec.default_output_node = "output_parameters"
        spec.exit_code(303, "ERROR_OUTPUT_INCOMPLETE", message="The output file was incomplete.")
        spec.exit_code(
            304,
            "ERROR_OUTPUT_CONTAINS_ABORT",
            message="The output file contains the word 'ABORT'.",
        )
        spec.exit_code(
            305,
            "ERROR_INCOMPATIBLE_CODE_VERSION",
            message="The version of the CP2K code needs to be newer than 8.1.",
        )
        spec.exit_code(
            310, "ERROR_READING_OUTPUT_FILE", message="The output file could not be read."
        )
        spec.exit_code(
            320, "ERROR_INVALID_OUTPUT", message="The output file contains invalid output."
        )
        spec.exit_code(
            400,
            "ERROR_OUT_OF_WALLTIME",
            message="The calculation stopped prematurely because it ran out of walltime.",
        )
        spec.exit_code(
            401, "ERROR_INTERRUPTED", message="The calculation did not finish properly."
        )
        spec.exit_code(
            402,
            "ERROR_ODD_NR_ELECTRONS",
            message="Odd number of electrons, UKS or ROKS has to be used.",
        )
        spec.exit_code(
            403, "ERROR_NEED_ADDED_MOS", message="Unoccupied orbitals have to be added."
        )
        spec.exit_code(
            404,
            "ERROR_ILL_CONDITIONED_MATRIX",
            message="Cholesky decompose failed due to ill-conditioned matrix.",
        )
        spec.exit_code(
            405,
            "ERROR_BAD_CONDITION_NUMBER",
            message="Bad condition number R_COND (smaller than the machine working precision).",
        )
        spec.exit_code(
            500,
            "ERROR_GEOMETRY_CONVERGENCE_NOT_REACHED",
            message="The ionic minimization cycle did not converge for the given thresholds.",
        )
        spec.exit_code(
            501,
            "ERROR_SCF_CONVERGENCE_NOT_REACHED",
            message="No parameters found to converge the Kohn-Sham equations.",
        )

    def prepare_for_submission(self, folder):
        """Prepare input for calculation."""
        parameters = _set_dict_to_upper(self.inputs.parameters.get_dict())
        dict_set_parameter(parameters, ["GLOBAL", "PROJECT"], self._PROJECT_NAME)

        if "structure" in self.inputs:
            self._write_structure(self.inputs.structure, folder, self._COORDS_FILE_NAME)
            cell_section = dict_retrieve_parameter(parameters, ["FORCE_EVAL", "SUBSYS", "CELL"])
            if cell_section is not None:
                for conf_key in ["A", "B", "C", "ABC", "ALPHA_BETA_GAMMA", "CELL_FILE_NAME"]:
                    if conf_key in cell_section:
                        raise InputValidationError(
                            f"Key '{conf_key}' cannot be set in combination with 'structure'."
                        )
            for letter, cell_v in zip(["A", "B", "C"], self.inputs.structure.cell):
                dict_set_parameter(
                    parameters,
                    ["FORCE_EVAL", "SUBSYS", "CELL", letter],
                    " ".join(str(coord) for coord in cell_v),
                )

            coord_section = dict_retrieve_parameter(parameters, ["FORCE_EVAL", "SUBSYS", "COORD"])
            if coord_section is not None:
                raise InputValidationError(
                    "'&COORD' section cannot be set in combination with 'structure'."
                )
            top_section = dict_retrieve_parameter(parameters, ["FORCE_EVAL", "SUBSYS", "TOPOLOGY"])
            if top_section is not None and "COORD_FILE_NAME" in top_section:
                raise InputValidationError(
                    "'COORD_FILE_NAME' cannot be set in combination with 'structure'."
                )

            dict_set_parameter(
                parameters,
                ["FORCE_EVAL", "SUBSYS", "TOPOLOGY", "COORD_FILE_NAME"],
                self._COORDS_FILE_NAME,
            )
            dict_set_parameter(
                parameters, ["FORCE_EVAL", "SUBSYS", "TOPOLOGY", "COORD_FILE_FORMAT"], "XYZ"
            )
        with folder.open(self.options.input_filename, "w", encoding="utf8") as handle:
            for line in _transform_dict_to_cp2k(parameters):
                handle.write(line + "\n")

        settings = self.inputs.settings.get_dict() if "settings" in self.inputs else {}
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.options.output_filename
        codeinfo.cmdline_params = settings.pop("CMDLINE", []) + ["-i", self.options.input_filename]

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.remote_symlink_list = []
        calcinfo.remote_copy_list = []
        if "file" in self.inputs:
            for name, obj in self.inputs.file.items():
                calcinfo.local_copy_list.append((obj.uuid, obj.filename, obj.filename))
        calcinfo.retrieve_list = [self.options.output_filename, self._PROJECT_NAME + "-1.restart"]
        calcinfo.retrieve_list += settings.pop("additional_retrieve_list", [])
        calcinfo.retrieve_temporary_list = []
        calcinfo.retrieve_temporary_list += settings.pop("additional_retrieve_temporary_list", [])

        if "parent_calc_folder" in self.inputs:
            comp_uuid = self.inputs.parent_calc_folder.computer.uuid
            remote_path = self.inputs.parent_calc_folder.get_remote_path()
            copy_info = []
            try:
                files_list = self.inputs.parent_calc_folder.listdir()
                copy_info.append((comp_uuid, remote_path, self._PARENT_CALC_FLDR_NAME))
                if "aiida-RESTART.kp" in files_list:
                    copy_info.append(
                        (comp_uuid, remote_path + "/aiida-RESTART.kp", "aiida-RESTART.kp")
                    )
                elif "aiida-RESTART.wfn" in files_list:
                    copy_info.append(
                        (comp_uuid, remote_path + "/aiida-RESTART.wfn", "aiida-RESTART.wfn")
                    )
            except OSError:
                pass

            if self.inputs.code.computer.uuid == comp_uuid:
                calcinfo.remote_symlink_list += copy_info
            else:
                self.report("Transfer between two different computers not yet supported.")
                # self.report(
                #    f"Transferring files from {self.inputs.parent_calc_folder.computer.label} to "
                #    f"{self.inputs.code.computer.label}."
                # )
                # calcinfo.remote_copy_list += copy_info

        return calcinfo

    @staticmethod
    def _write_structure(structure, folder, name):
        """Adapted function to fix bug in kind names."""
        xyz = f"{len(structure.sites)}\n\n"
        for site in structure.sites:
            xyz += site.kind_name + " ".join(["" for idx0 in range(10 - len(site.kind_name))])
            xyz += " ".join([f"{coord:25.16f}" for coord in site.position])
            xyz += "\n"
        with io.open(folder.get_abs_path(name), mode="w", encoding="utf-8") as fobj:
            fobj.write(xyz)
