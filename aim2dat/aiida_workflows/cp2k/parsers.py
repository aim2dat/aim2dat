"""
Module containing custom parsers for AiiDA.
"""

# Standard library imports
import re
import os

# Third party library imports
import numpy as np
from aiida.parsers import Parser
from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import Dict, List
from aiida.plugins import DataFactory
from aiida.common import OutputParsingError

# Internal library imports
from aim2dat.io import (
    read_cp2k_stdout,
    read_cp2k_restart_structure,
    read_cp2k_proj_dos,
)


StructureData = DataFactory("core.structure")
BandsData = DataFactory("core.array.bands")
XyData = DataFactory("core.array.xy")
GCubeData = DataFactory("aim2dat.gaussian_cube")


class _Cp2kBaseParser(Parser):
    """Minimum parser for CP2K."""

    parser_type = "standard"
    extra_output_functions = []

    def parse(self, **kwargs):
        """Receives as input a dictionary of retrieved nodes. Does all the logic here."""
        try:
            _ = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        retrieved_temporary_folder = kwargs.get("retrieved_temporary_folder", None)

        # Parse main output
        result_dict = self._parse_stdout()

        # Parse extra output
        parse_extra_output = True
        settings = self.node.inputs.settings.get_dict() if "settings" in self.node.inputs else {}
        if settings.get("output_check_scf_conv", False):
            if not result_dict.get("scf_converged", False):
                parse_extra_output = False
        if parse_extra_output:
            for output_f_label in self.extra_output_functions:
                output_f = getattr(self, output_f_label)
                output_dict = output_f(retrieved_temporary_folder)
                result_dict.update(output_dict)

        # Parse structure
        output_structure = None
        try:
            output_structure = self._parse_output_structure()
            if isinstance(output_structure, StructureData):
                self.out("output_structure", output_structure)
            else:  # in case this is an error code
                return output_structure
        except exceptions.NotExistent:
            pass

        self._create_output_nodes(result_dict, output_structure)

        # All exit_codes from the main-output are triggered here
        if "geo_not_converged" in result_dict:
            return self.exit_codes.ERROR_GEOMETRY_CONVERGENCE_NOT_REACHED
        elif not result_dict.get("scf_converged", True):
            return self.exit_codes.ERROR_SCF_CONVERGENCE_NOT_REACHED
        elif "odd_nr_electrons" in result_dict:
            return self.exit_codes.ERROR_ODD_NR_ELECTRONS
        elif "need_added_mos" in result_dict:
            return self.exit_codes.ERROR_NEED_ADDED_MOS
        elif "cholesky_decompose_failed" in result_dict:
            return self.exit_codes.ERROR_ILL_CONDITIONED_MATRIX
        elif "bad_condition_number" in result_dict:
            return self.exit_codes.ERROR_BAD_CONDITION_NUMBER
        elif result_dict.get("exceeded_walltime", False) or self.node.exit_status == 120:
            return self.exit_codes.ERROR_OUT_OF_WALLTIME
        elif "aborted" in result_dict:
            return self.exit_codes.ERROR_OUTPUT_CONTAINS_ABORT
        elif "interrupted" in result_dict:
            return self.exit_codes.ERROR_INTERRUPTED
        elif "incompatible_code" in result_dict:
            return self.exit_codes.ERROR_INCOMPATIBLE_CODE_VERSION
        elif "incomplete" in result_dict:
            return self.exit_codes.ERROR_OUTPUT_INCOMPLETE
        elif "io_error" in result_dict:
            return self.exit_codes.ERROR_READING_OUTPUT_FILE
        else:
            return ExitCode(0)

    def _parse_stdout(self):
        """Parse main CP2K output file."""
        fname = self.node.get_option("output_filename")

        try:
            result_dict = read_cp2k_stdout(
                self.retrieved.get_object_content(fname),
                parser_type=self.parser_type,
                raise_error=False,
            )
        # TODO distinguish different exceptions.
        except IOError:
            result_dict = {"io_error": True}

        if result_dict is None:
            raise OutputParsingError("CP2K version is not supported.")

        return result_dict

    def _parse_output_structure(self):
        """Parse final structure."""
        fname = (
            self.node.process_class._PROJECT_NAME + "-1.restart"
        )  # pylint: disable=protected-access

        # Check if the restart file is present.
        if fname not in self.retrieved.list_object_names():
            raise exceptions.NotExistent(
                "No restart file available, so the output trajectory can't be extracted."
            )

        # Read the restart file.
        # TODO distinguish different exceptions.
        try:
            structures = read_cp2k_restart_structure(self.retrieved.get_object_content(fname))
        except IOError:
            return self.exit_codes.ERROR_READING_OUTPUT_FILE

        # For now only one structure is supported
        if isinstance(structures, list):
            raise OutputParsingError("Multiple force-evaluations not yet supported.")
        else:
            structure = structures
        structure_node = StructureData(cell=structure["cell"], pbc=structure["pbc"])
        for kind, sym, pos in zip(
            structure["kinds"], structure["elements"], structure["positions"]
        ):
            structure_node.append_atom(position=pos, symbols=sym, name=kind)
        return structure_node

    def _parse_gaussian_cubes(self, retrieved_temporary_folder):
        """Parse cube files."""
        if retrieved_temporary_folder is not None:
            pattern = re.compile(r"^\S*-([A-Za-z]*)?_([A-Za-z0-9]+)?_*(\d*)?-\d+_\d+.cube")
            file_pathes = os.listdir(retrieved_temporary_folder)
            for file_path in file_pathes:
                found_match = pattern.match(file_path)
                if found_match is not None:
                    groups = found_match.groups()
                    label = []
                    for grp in groups:
                        if grp == "":
                            continue
                        elif grp.isdigit():
                            label.append(str(int(grp)))
                        else:
                            label.append(grp.lower())
                    label = "_".join(label)
                    file_path = os.path.join(retrieved_temporary_folder, file_path)
                    with open(file_path, "r") as fobj:
                        g_cube_data = GCubeData.set_from_file(fobj)
                    self.out("output_cubes." + label, g_cube_data)
        return {}

    def _create_output_nodes(self, result_dict, output_structure):
        self.out("output_parameters", Dict(dict=result_dict))

    def _process_kind_info(self, result_dict):
        if "kind_info" in result_dict:
            kind_info = result_dict.pop("kind_info")
            self.out("output_kind_info", List(list=kind_info))


class Cp2kStandardParser(_Cp2kBaseParser):
    """Standard parser for CP2K."""

    parser_type = "standard"
    extra_output_functions = ["_parse_pdos", "_parse_gaussian_cubes"]

    def _parse_pdos(self, retrieved_temporary_folder):
        """PDOS parser."""
        return_dict = {}
        if retrieved_temporary_folder is not None:
            pdos_data = None
            try:
                pdos_data = read_cp2k_proj_dos(retrieved_temporary_folder)
            except ValueError as e:
                if str(e) != "No files with the correct naming scheme found.":
                    raise
            if pdos_data is not None:
                xydata = XyData()
                xydata.set_x(
                    np.array(pdos_data["energy"]),
                    "energy",
                    pdos_data["unit_x"],
                )
                y_labels = ["occupation"]
                y_values = [np.array(pdos_data["occupation"])]
                y_units = [""]
                for pdos in pdos_data["pdos"]:
                    for orb_label, density in pdos.items():
                        if orb_label == "kind":
                            continue
                        y_values.append(np.array(density))
                        y_labels.append(pdos["kind"] + "_" + orb_label)
                        y_units.append("states/" + pdos_data["unit_x"])
                xydata.set_y(y_values, y_labels, y_units)
                xydata.set_attribute("e_fermi", pdos_data["e_fermi"])  # TODO this to pDOS output..
                self.out("output_pdos", xydata)
                return_dict["e_fermi"] = pdos_data["e_fermi"]
        return return_dict

    def _create_output_nodes(self, result_dict, output_structure):
        if "kpoint_data" in result_dict:
            bnds = BandsData()
            bnds.set_kpoints(result_dict["kpoint_data"]["kpoints"])
            bnds.labels = result_dict["kpoint_data"]["labels"]
            bnds.set_bands(
                result_dict["kpoint_data"]["bands"],
                units=result_dict["kpoint_data"]["bands_unit"],
                occupations=result_dict["kpoint_data"]["occupations"],
            )
            self.out("output_bands", bnds)
            del result_dict["kpoint_data"]
        if "eigenvalues_info" in result_dict:
            ev_info = result_dict.pop("eigenvalues_info")
            self.out("output_eigenvalues", Dict(dict=ev_info))
        self.out("output_parameters", Dict(dict=result_dict))


class Cp2kPartialChargesParser(_Cp2kBaseParser):
    """
    Parser specifically designed to parse partial charges from CP2K output.
    """

    parser_type = "partial_charges"
    extra_output_functions = ["_parse_gaussian_cubes"]

    def _create_output_nodes(self, result_dict, output_structure):
        for pc_label in ["mulliken", "hirshfeld"]:
            if pc_label in result_dict:
                self.out(f"output_{pc_label}_populations", List(list=result_dict.pop(pc_label)))
        if "kind_info" in result_dict:
            self.out("output_kind_info", List(list=result_dict.pop("kind_info")))
        self.out("output_parameters", Dict(dict=result_dict))


class Cp2kTrajectoryParser(_Cp2kBaseParser):
    """
    Parser which includes information on each motion step and (to-do) the trajectory.
    """

    parser_type = "trajectory"

    def _create_output_nodes(self, result_dict, output_structure):
        self.out("output_motion_step_info", List(list=result_dict.pop("motion_step_info")))
        if "kind_info" in result_dict:
            self.out("output_kind_info", List(list=result_dict.pop("kind_info")))
        self.out("output_parameters", Dict(dict=result_dict))
