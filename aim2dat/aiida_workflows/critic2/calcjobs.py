"""
Calcjobs for the critic2 software package.
"""

# Third party library imports
from aiida.common import CalcInfo, CodeInfo, InputValidationError
from aiida.engine import CalcJob
import aiida.orm as aiida_orm


class Critic2Calculation(CalcJob):
    """
    Calcjob for the critic2 software package.
    """

    @classmethod
    def define(cls, spec):
        """Define input/output and outline."""
        super().define(spec)
        spec.input(
            "parameters",
            valid_type=aiida_orm.List,
            help="List of input parameters (each item is parsed as a line in the input file).",
        )
        spec.input(
            "charge_density_folder",
            valid_type=aiida_orm.RemoteData,
            help="Folder containing the carge-density cube files",
        )
        spec.input(
            "kind_info",
            valid_type=aiida_orm.List,
            help="List containing the number of valence and core electrons for each element.",
            required=False,
        )
        spec.inputs["metadata"]["options"]["input_filename"].default = "aiida.in"
        spec.inputs["metadata"]["options"]["output_filename"].default = "aiida.out"
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
        spec.inputs["metadata"]["options"]["withmpi"].default = False
        spec.inputs["metadata"]["options"]["parser_name"].default = "aim2dat.critic2"
        spec.output(
            "output_parameters",
            valid_type=aiida_orm.Dict,
            required=True,
            help="The output dictionary containing results of the calculation.",
        )
        spec.output(
            "output_bader_populations",
            valid_type=aiida_orm.List,
            required=False,
            help="Calculated Bader populations.",
        )
        spec.output_namespace(
            "output_planes",
            dynamic=True,
            required=False,
            help="Calculated planes.",
        )
        spec.default_output_node = "output_parameters"
        spec.exit_code(
            310, "ERROR_READING_OUTPUT_FILE", message="The output file could not be read."
        )
        spec.exit_code(
            320, "ERROR_INVALID_OUTPUT", message="The output file contains invalid output."
        )
        spec.exit_code(
            402,
            "ERROR_ABORT",
            message="Calculation was not successful.",
        )

    def prepare_for_submission(self, folder):
        """Prepare for submission."""
        cube_files = []
        retrieve_list = [self.options.output_filename]
        with folder.open(self.options.input_filename, "w", encoding="utf8") as handle:
            for input_item in self.inputs.parameters.get_list():
                cube_files += [arg for arg in input_item.split() if "cube" in arg]
                if input_item.split()[-1] == "zpsp" and "kind_info" in self.inputs:
                    input_item += self._add_zpsp_section()
                if input_item.split()[0] == "plane":
                    retrieve_list += self._get_plane_file_name(input_item)
                handle.write(input_item + "\n")
        cube_files = list(set(cube_files))

        cd_folder = self.inputs.charge_density_folder
        remote_path = self.inputs.charge_density_folder.get_remote_path()
        comp_uuid = self.inputs.charge_density_folder.computer.uuid
        copy_links = []
        for cube_f in cube_files:
            if cube_f in cd_folder.listdir():
                copy_links.append((comp_uuid, remote_path + "/" + cube_f, cube_f))
            else:
                raise InputValidationError(f"{cube_f} not found in `charge_density_folder`.")

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
        calcinfo.retrieve_list = retrieve_list
        return calcinfo

    def _add_zpsp_section(self):
        element_dict = {}
        for kind_info in self.inputs.kind_info.get_list():
            element_dict[kind_info["element"]] = kind_info["valence_electrons"]
        return "".join([f" {el} {val_e}" for el, val_e in element_dict.items()])

    @staticmethod
    def _get_plane_file_name(input_item):
        line_splitted = input_item.split()
        if "file" in line_splitted:
            return [line_splitted[line_splitted.index("file") + 1]]
        else:
            return ["stdin_plane.dat"]
