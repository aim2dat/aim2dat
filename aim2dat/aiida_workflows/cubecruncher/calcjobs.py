"""
Calcjobs for the cubecruncher software package. The calculations are performed via cubecruncher.
Please make sure to set the path `cubecruncher_executable` correctly.
The calculation will the be submitted via `bash`.
Make sure to have the code `bash` implemented `verdi code create core.code.installed`.
"""

# Third party library imports
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.engine import CalcJob
import aiida.orm as aiida_orm
from aiida.plugins import DataFactory


GCubeData = DataFactory("aim2dat.gaussian_cube")


class CubecruncherCalculation(CalcJob):
    """
    Calcjob for the cubecruncher software package. The calculations are performed via cubecruncher.
    Please make sure to set the path `cubecruncher_executable` correctly. The calculation will the
    be submitted via `bash`.
    Make sure to have the code `bash` implemented `verdi code create core.code.installed`.
    """

    @classmethod
    def define(cls, spec):
        """Define input/output and outline."""
        super().define(spec)
        spec.input(
            "parameters",
            valid_type=aiida_orm.List,
            default=lambda: aiida_orm.List([]),
            help=""
            "List of input parameters for the final charge-density-difference cube calculation.",
        )
        spec.input(
            "charge_density_folder",
            valid_type=aiida_orm.RemoteData,
            help="Folder containing the charge-density cube files",
        )
        spec.inputs["metadata"]["options"]["output_filename"].default = "cdd.cube"
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
        spec.inputs["metadata"]["options"]["withmpi"].default = False
        spec.inputs["metadata"]["options"]["parser_name"].default = "aim2dat.cubecruncher"
        spec.output(
            "cdd_cube",
            valid_type=GCubeData,
            help="Charge-density-difference cube file.",
        )
        spec.exit_code(
            310, "ERROR_READING_OUTPUT_FILE", message="The output file could not be read."
        )
        spec.exit_code(
            401,
            "ERROR_CHARGE_DENSITY_FILES",
            message="Charge density files not found in remote folder.",
        )

    def prepare_for_submission(self, folder):
        """Prepare for submission."""
        cd_folder = self.inputs.charge_density_folder
        remote_path = self.inputs.charge_density_folder.get_remote_path()
        comp_uuid = self.inputs.charge_density_folder.computer.uuid
        copy_links = []
        cube_files = sorted([file for file in cd_folder.listdir() if file.endswith(".cube")])
        for cube_f in cube_files:
            copy_links.append((comp_uuid, remote_path + "/" + cube_f, cube_f))
        cube_files = list(reversed(cube_files))

        calcinfo = CalcInfo()
        calcinfo.codes_info = []
        parameters = self.inputs.parameters.get_list()
        parameters_str = [f"-{parameter}" for parameter in parameters]
        for i, cube_f in enumerate(cube_files[1:], start=1):
            input_cube = cube_files[0] if i == 1 else f"tmp{i-1}.cube"
            output_cube = "cdd.cube" if cube_f == cube_files[-1] else f"tmp{i}.cube"
            codeinfo = CodeInfo()
            codeinfo.code_uuid = self.inputs.code.uuid
            cmd = [
                "-i",
                input_cube,
                "-o",
                output_cube,
                "-subtract",
                cube_f,
            ]
            if output_cube == "cdd.cube":
                cmd += parameters_str
            codeinfo.cmdline_params = cmd
            calcinfo.codes_info.append(codeinfo)

        calcinfo.local_copy_list = []
        if self.inputs.code.computer.uuid == cd_folder.computer.uuid:
            calcinfo.remote_symlink_list = copy_links
        else:
            self.report(
                f"Transferring files from {cd_folder.computer.label} to "
                f"{self.inputs.code.computer.label}."
            )
            calcinfo.remote_copy_list = copy_links
        calcinfo.retrieve_temporary_list = [self.options.output_filename]
        return calcinfo
