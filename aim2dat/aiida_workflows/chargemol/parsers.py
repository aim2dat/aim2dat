"""
Parsers for the chargemol software package.
"""

# Third party library imports
from aiida.parsers import Parser
from aiida.orm import List, Dict


class ChargemolParser(Parser):
    """
    Output parser for the chargemol software package.
    """

    def parse(self, **kwargs):
        """Receives in input a dictionary of retrieved nodes. Does all the logic here."""
        try:
            with self.retrieved.open(self.node.get_option("output_filename"), "r") as handle:
                result_dict = self._parse_stdout(handle)
        except OSError:
            return self.exit_codes.ERROR_READING_OUTPUT_FILE
        except ValueError:
            return self.exit_codes.ERROR_INVALID_OUTPUT

        if result_dict.get("insufficient_accuracy", False):
            return self.exit_codes.ERROR_INSUFFICIENT_ACCURACY

        if result_dict.get("aborted", False):
            return self.exit_codes.ERROR_ABORT

        for method, apx in {"DDEC3": "", "DDEC6": "_even_tempered"}.items():
            if method + apx + "_net_atomic_charges.xyz" in self.retrieved.list_object_names():
                result_dict["method"] = method
                result_dict[method] = []
                with self.retrieved.open(method + apx + "_net_atomic_charges.xyz", "r") as handle:
                    output_string = handle.read()
                    for line in output_string.splitlines()[2:]:
                        if len(line.strip()) == 0:
                            break
                        result_dict[method].append(
                            {"element": line.split()[0], "charge": float(line.split()[4])}
                        )

        self._create_output_nodes(result_dict)

    def _parse_stdout(self, handle):
        """Parse sandard output file."""
        output_string = handle.read()
        result_dict = {}
        for line in output_string.splitlines():
            if line.startswith(" Starting Chargemol version"):
                result_dict["chargemol_version"] = float(line.split()[3])
            if line.startswith(" Job running using"):
                result_dict["chargemol_branch"] = line.split()[3]
            if line.startswith(" Finished chargemol in"):
                result_dict["aborted"] = False
            if line.startswith(" Integration volumes are not sufficiently accurate."):
                result_dict["insufficient_accuracy"] = True
            if line.startswith(" Finished chargemol in"):
                result_dict["runtime"] = float(line.split()[3])
        if "aborted" not in result_dict and "insufficient_accuracy" not in result_dict:
            result_dict["aborted"] = True
        return result_dict

    def _create_output_nodes(self, result_dict):
        """Create output nodes."""
        for method in ["DDEC3", "DDEC6"]:
            if method in result_dict:
                pc_list = result_dict.pop(method)
                self.out(f"output_{method.lower()}_populations", List(list=pc_list))
        self.out("output_parameters", Dict(dict=result_dict))
