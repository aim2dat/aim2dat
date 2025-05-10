"""
Parsers for the critic2 software package.
"""

# Third party library imports
import numpy as np
from aiida.parsers import Parser
from aiida.orm import List, Dict, ArrayData

# Internal library imports
from aim2dat.io import read_critic2_stdout, read_critic2_plane


class Critic2Parser(Parser):
    """
    Output parser for the critic2 software package.
    """

    def parse(self, **kwargs):
        """Receives in input a dictionary of retrieved nodes. Does all the logic here."""
        try:
            result_dict = read_critic2_stdout(
                self.retrieved.get_object_content(self.node.get_option("output_filename"))
            )
        except OSError:
            return self.exit_codes.ERROR_READING_OUTPUT_FILE
        except ValueError:
            return self.exit_codes.ERROR_INVALID_OUTPUT

        planes = {}
        for file_path in result_dict.pop("plane_files"):
            try:
                planes[file_path.split(".")[0]] = read_critic2_plane(
                    self.retrieved.get_object_content(file_path)
                )
            except OSError:
                return self.exit_codes.ERROR_READING_OUTPUT_FILE
            except ValueError:
                return self.exit_codes.ERROR_INVALID_OUTPUT

        if result_dict.get("aborted", False):
            return self.exit_codes.ERROR_ABORT

        self._create_output_nodes(result_dict, planes)

    def _create_output_nodes(self, result_dict, planes):
        """Create output nodes."""
        if "method" in result_dict and result_dict["method"] in [
            "Yu-Trinkle integration",
            "Henkelmann et al. integration",
        ]:
            pc_list = result_dict.pop("partial_charges")
            if "kind_info" in self.node.inputs:
                kind_info_list = self.node.inputs.kind_info.get_list()
                for population, kind_info in zip(pc_list, kind_info_list):
                    population["charge"] = (
                        kind_info["valence_electrons"] - population["population"]
                    )
            self.out("output_bader_populations", List(list=pc_list))
        if len(planes) > 0:
            for plane_label, plane in planes.items():
                array_data = ArrayData()
                array_data.set_array("coordinates", np.array(plane["coordinates"]))
                array_data.set_array("values", np.array(plane["values"]))
                array_data.set_attribute("coordinates_unit", plane["coordinates_unit"])
                self.out("output_planes." + plane_label, array_data)
        self.out("output_parameters", Dict(dict=result_dict))
