"""
Calcjobs for the chargemol software package.
"""

# Third party library imports
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.engine import CalcJob
import aiida.orm as aiida_orm


class ChargemolCalculation(CalcJob):
    """
    Calcjob for the chargemol software package.
    """

    @classmethod
    def define(cls, spec):
        """Define input/output and outline."""
        super().define(spec)
        spec.input(
            "parameters",
            valid_type=aiida_orm.Dict,
            help="Dictionary containing the input parameters.",
        )
        spec.input(
            "charge_density_folder",
            valid_type=aiida_orm.RemoteData,
            help="Folder containing the charge-density cube files",
        )
        spec.input(
            "charge_density_filename",
            valid_type=aiida_orm.Str,
            required=False,
            help="Name of the cube-file containing the valence density of the system.",
        )
        spec.input(
            "spin_density_filename",
            valid_type=aiida_orm.Str,
            required=False,
            help="Name of the cube-file containing the spin density of the system.",
        )
        spec.input(
            "kind_info",
            valid_type=aiida_orm.List,
            required=False,
            help="List containing the atomic number and number of core electrons.",
        )
        spec.input(
            "path_atomic_densities",
            valid_type=aiida_orm.Str,
            required=False,
            help="Absolute path to the atomic densities needed for the chargemol calculation.",
        )
        spec.inputs["metadata"]["options"]["input_filename"].default = "job_control.txt"
        spec.inputs["metadata"]["options"][
            "output_filename"
        ].default = "valence_cube_DDEC_analysis.output"
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
        spec.inputs["metadata"]["options"]["withmpi"].default = False
        spec.inputs["metadata"]["options"]["parser_name"].default = "aim2dat.chargemol"
        spec.output(
            "output_parameters",
            valid_type=aiida_orm.Dict,
            required=True,
            help="The output dictionary containing results of the calculation.",
        )
        spec.output(
            "output_ddec3_populations",
            valid_type=aiida_orm.List,
            required=False,
            help="Calculated DDEC3 populations.",
        )
        spec.output(
            "output_ddec6_populations",
            valid_type=aiida_orm.List,
            required=False,
            help="Calculated DDEC6 populations.",
        )
        spec.exit_code(
            310, "ERROR_READING_OUTPUT_FILE", message="The output file could not be read."
        )
        spec.exit_code(
            320, "ERROR_INVALID_OUTPUT", message="The output file contains invalid output."
        )
        spec.exit_code(
            401,
            "ERROR_CHARGE_DENSITY_FILES",
            message="Charge density files not found in remote folder.",
        )
        spec.exit_code(
            402,
            "ERROR_ABORT",
            message="Calculation was not successful.",
        )
        spec.exit_code(
            403,
            "ERROR_INSUFFICIENT_ACCURACY",
            message="Integration volumes are not sufficiently accurate.",
        )

    def prepare_for_submission(self, folder):
        """Prepare for submission."""
        cd_folder = self.inputs.charge_density_folder
        remote_path = self.inputs.charge_density_folder.get_remote_path()
        comp_uuid = self.inputs.charge_density_folder.computer.uuid
        copy_links = []
        if "charge_density_filename" in self.inputs:
            cube_f = self.inputs.charge_density_filename.value
            if cube_f not in cd_folder.listdir():
                return self.exit_codes.ERROR_CHARGE_DENSITY_FILES
            copy_links.append((comp_uuid, remote_path + "/" + cube_f, "valence_density.cube"))
        if "spin_density_filename" in self.inputs:
            cube_f = self.inputs.spin_density_filename.value
            if cube_f not in cd_folder.listdir():
                return self.exit_codes.ERROR_SPIN_DENSITY_FILES
            copy_links.append((comp_uuid, remote_path + "/" + cube_f, "spin_density.cube"))

        with folder.open(self.options.input_filename, "w", encoding="utf8") as handle:
            parameters = self.inputs.parameters.get_dict()
            for key, item in parameters.items():
                handle.write(f"<{key}>\n")
                if isinstance(item, (list, tuple)):
                    for list_item in item:
                        handle.write(f"{list_item}\n")
                else:
                    handle.write(f"{item}\n")
                handle.write(f"</{key}>\n\n")
            if (
                "path_atomic_densities" in self.inputs
                and "atomic densities directory complete path" not in parameters
            ):
                handle.write("<atomic densities directory complete path>\n")
                handle.write(f"{self.inputs.path_atomic_densities.value}\n")
                handle.write("</atomic densities directory complete path>\n\n")
            if "kind_info" in self.inputs and "number of core electrons" not in parameters:
                handle.write("<number of core electrons>\n")
                element_list = []
                for kind_info in self.inputs.kind_info.get_list():
                    if kind_info["element"] not in element_list:
                        element_list.append(kind_info["element"])
                        handle.write(f"{kind_info['atomic_nr']} {kind_info['core_electrons']}\n")
                handle.write("</number of core electrons>\n")

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdin_name = self.options.input_filename
        codeinfo.stdout_name = self.options.output_filename
        codeinfo.cmdline_params = [self.options.input_filename]

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        if self.inputs.code.computer.uuid == cd_folder.computer.uuid:
            calcinfo.remote_symlink_list = copy_links
        else:
            self.report(
                f"Transfering files from {cd_folder.computer.label} to "
                f"{self.inputs.code.computer.label}."
            )
            calcinfo.remote_copy_list = copy_links
        calcinfo.retrieve_list = [self.options.output_filename, ("*.xyz", ".", 0)]
        return calcinfo
