"""Tests for the cp2k main-output parser."""

# Standard library imports
import os

# Third party library imports
import pytest

# from aiida_cp2k.utils.parser import parse_cp2k_output_advanced

# Internal library imports
from aim2dat.io.cp2k.legacy_parser import MainOutputParser
from aim2dat.io.yaml import load_yaml_file


MAIN_PATH = os.path.dirname(__file__) + "/cp2k_stdout/"


class OutputParserTester:
    """Class to compare output-dictionaries."""

    def __init__(self):
        """Initialize class."""
        self.result_dicts = {}

    def add_output_file(self, test_case, print_level, cp2k_version, incl_cp2k_parser=False):
        """Add output-file to the class."""
        file_content = open(
            MAIN_PATH + f"cp2k-{cp2k_version}/{test_case}_{print_level}/aiida.out", "r"
        ).read()
        parser = MainOutputParser(file_content)
        self.result_dicts[test_case + "-" + str(cp2k_version)] = {
            "standard": parser.retrieve_result_dict("standard"),
            "partial_charges": parser.retrieve_result_dict("partial_charges"),
            "trajectory": parser.retrieve_result_dict("trajectory"),
        }
        # if incl_cp2k_parser:
        #     self.result_dicts[test_case + "-" + str(cp2k_version)][
        #         "aiida-cp2k"
        #     ] = parse_cp2k_output_advanced(file_content)

    def check_keywords(self, test_case, cp2k_version, keyword_list, parser_type):
        """
        Check values of the dictionary against reference.

        Parameters
        ----------
        test_case : str
            Label of the test case.
        cp2k_version : float
            Used cp2k version.
        keyword_list : list
            A list of tuples consisting of the keyword (or a list of keywords in case of a nested
            dictionary) and the reference value.
        parser_type : str
            Parser type that is tested.
        """
        result_dict = self.result_dicts[test_case + "-" + str(cp2k_version)][parser_type]
        for item in keyword_list:
            if isinstance(item[0], list):
                value = result_dict[item[0][0]]
                for key in item[0][1:]:
                    value = value[key]
            else:
                value = result_dict[item[0]]
            assert value == item[1], f"Wrong {item[0]} in parser {parser_type}."

    def compare_parser_versions(
        self,
        test_case,
        cp2k_version_1,
        cp2k_version_2,
        parser_type,
        exclude_keys=[],
        tolerance=1.0e-3,
    ):
        """
        Compare the output of two different program versions.
        """
        result_dict1 = self.result_dicts[test_case + "-" + str(cp2k_version_1)][parser_type]
        result_dict2 = self.result_dicts[test_case + "-" + str(cp2k_version_2)][parser_type]
        del result_dict1["cp2k_version"]
        del result_dict2["cp2k_version"]
        for key0 in exclude_keys:
            if isinstance(key0, str):
                del result_dict1[key0]
                del result_dict2[key0]
            else:
                helper_dict1 = result_dict1
                helper_dict2 = result_dict2
                for key_val in key0[:-1]:
                    helper_dict1 = helper_dict1[key_val]
                    helper_dict2 = helper_dict2[key_val]
                del helper_dict1[key0[-1]]
                del helper_dict2[key0[-1]]

        self._check_sub_dict(result_dict1, result_dict2, tolerance)

    def compare_aiidacp2k(self, test_case, cp2k_version, keyword_list, parser_type):
        """
        Compare keywords with the aiida-cp2k advanced parser.
        """
        if self.result_dicts[test_case + "-" + str(cp2k_version)].get("aiida-cp2k"):
            result_dict1 = self.result_dicts[test_case + "-" + str(cp2k_version)][parser_type]
            result_dict2 = self.result_dicts[test_case + "-" + str(cp2k_version)]["aiida-cp2k"]
            for item in keyword_list:
                if isinstance(item["key"], list):
                    value1 = result_dict1[item["key"][0]]
                    value2 = result_dict2[item["key"][0]]
                    for key in item["key"][1:]:
                        value1 = value1[key]
                        value2 = value2[key]
                else:
                    value1 = result_dict1[item["key"]]
                    value2 = result_dict2[item["key"]]
                if isinstance(value1, list):
                    if item.get("excl_aiidacp2k"):
                        list_excl = sorted(item["excl_aiidacp2k"], reverse=True)
                        for excl in list_excl:
                            del value2[excl]
                    if item.get("excl_parser"):
                        list_excl = sorted(item["excl_parser"], reverse=True)
                        for excl in list_excl:
                            del value1[excl]
                    assert len(value1) == len(
                        value2
                    ), f"Different dimension of {item['key']} in comparison to aiida-cp2k parser."
                    for list_value1, list_value2 in zip(value1, value2):
                        self._check_value(list_value1, list_value2, item)
                else:
                    self._check_value(value1, value2, item)
        else:
            assert "Result-dict of CP2K-parser not found."

    def _check_sub_dict(self, dict1, dict2, tolerance):
        available_keys = list(dict2.keys())
        for key, value in dict1.items():
            assert key in dict2, f"{key} is not in parser 2."
            if isinstance(value, dict):
                self._check_sub_dict(value, dict2[key], tolerance)
            else:
                self._check_value(value, dict2[key], {"key": key}, tolerance)
            available_keys.remove(key)
        assert len(available_keys) == 0, f"Keys left in parser 2: {available_keys}."

    def _check_value(self, value1, value2, item, tolerance=1.0e-6):
        if isinstance(value1, tuple) or isinstance(value1, list):
            assert len(value1) == len(value2), f"Different dimension of {item['key']}."
            for list_val1, list_val2 in zip(value1, value2):
                self._check_value(list_val1, list_val2, item, tolerance)
        elif isinstance(value1, dict):
            self._check_sub_dict(value1, value2, tolerance)
        else:
            if isinstance(value1, float):
                assert abs(value1 - value2) < tolerance, f"Different value for {item['key']}."
            else:
                assert value1 == value2, f"Different value for {item['key']}."


@pytest.mark.parametrize(
    "test_case, print_level, cp2k_version",
    [
        ("geo_opt", "low", 8.1),
        ("geo_opt", "medium", 8.1),
        ("cell_opt", "low", 8.1),
        ("cell_opt", "medium", 8.1),
        ("cell_opt_1kpoint", "medium", 8.1),
        ("cell_opt_interrupted", "low", 8.1),
        ("cell_opt_walltime", "low", 8.1),
        ("md_nvt", "low", 8.1),
        ("md_nvt", "medium", 8.1),
        ("eigenvalues", "medium", 8.1),
        ("eigenvalues", "low", 8.1),
        ("eigenvalues_spin_pol", "low", 8.1),
        ("eigenvalues_spin_pol", "medium", 8.1),
        ("eigenvalues_changing_character", "low", 8.1),
        ("eigenvalues_changing_character", "medium", 8.1),
        ("bands", "low", 8.1),
        ("bands", "medium", 8.1),
        ("bands_spin_pol", "low", 8.1),
        ("bands_spin_pol", "medium", 8.1),
        ("smearing_need_added_mos", "medium", 8.1),
        ("need_lsd", "medium", 8.1),
        ("geo_opt", "low", 8.2),
        ("geo_opt", "medium", 8.2),
        ("cell_opt", "low", 8.2),
        ("cell_opt", "medium", 8.2),
        ("cell_opt_1kpoint", "medium", 8.2),
        ("cell_opt_interrupted", "low", 8.2),
        ("cell_opt_spgr", "low", 8.2),
        ("cell_opt_spgr", "medium", 8.2),
        ("md_nvt", "low", 8.2),
        ("md_nvt", "medium", 8.2),
        ("eigenvalues", "medium", 8.2),
        ("eigenvalues", "low", 8.2),
        ("eigenvalues_spin_pol", "low", 8.2),
        ("eigenvalues_spin_pol", "medium", 8.2),
        ("eigenvalues_changing_character", "low", 8.2),
        ("eigenvalues_changing_character", "medium", 8.2),
        ("eigenvalues_no_kpoints", "low", 8.2),
        ("eigenvalues_no_kpoints", "medium", 8.2),
        ("eigenvalues_spin_pol_no_kpoints", "low", 8.2),
        ("eigenvalues_spin_pol_no_kpoints", "medium", 8.2),
        ("bands", "low", 8.2),
        ("bands", "medium", 8.2),
        ("bands_spin_pol", "low", 8.2),
        ("bands_spin_pol", "medium", 8.2),
        ("smearing_need_added_mos", "medium", 8.2),
        ("need_lsd", "medium", 8.2),
        ("cholesky_decompose_failed", "medium", 8.2),
        ("geo_opt", "low", 9.1),
        ("geo_opt", "medium", 9.1),
        ("cell_opt", "low", 9.1),
        ("cell_opt", "medium", 9.1),
        ("cell_opt_1kpoint", "medium", 9.1),
        ("cell_opt_interrupted", "low", 9.1),
        ("cell_opt_spgr", "low", 9.1),
        ("cell_opt_spgr", "medium", 9.1),
        ("cell_opt_cg_spgr", "medium", 9.1),
        ("md_nvt", "low", 9.1),
        ("md_nvt", "medium", 9.1),
        ("eigenvalues", "medium", 9.1),
        ("eigenvalues", "low", 9.1),
        ("eigenvalues_spin_pol", "low", 9.1),
        ("eigenvalues_spin_pol", "medium", 9.1),
        ("eigenvalues_changing_character", "low", 9.1),
        ("eigenvalues_changing_character", "medium", 9.1),
        ("eigenvalues_no_kpoints", "low", 9.1),
        ("eigenvalues_no_kpoints", "medium", 9.1),
        ("eigenvalues_spin_pol_no_kpoints", "low", 9.1),
        ("eigenvalues_spin_pol_no_kpoints", "medium", 9.1),
        ("bands", "low", 9.1),
        ("bands", "medium", 9.1),
        ("bands_spin_pol", "low", 9.1),
        ("bands_spin_pol", "medium", 9.1),
        ("smearing_need_added_mos", "medium", 9.1),
        ("need_lsd", "medium", 9.1),
    ],
)
def test_mainoutput(test_case, print_level, cp2k_version):
    """
    Test the cp2k output-parser.
    """
    parser_tester = OutputParserTester()
    parser_tester.add_output_file(test_case, print_level, cp2k_version, incl_cp2k_parser=False)
    parser_types = ["standard", "partial_charges", "trajectory"]
    for ptype in parser_types:
        reference_values = list(
            load_yaml_file(
                MAIN_PATH + f"cp2k-{cp2k_version}/{test_case}_{print_level}_{ptype}_reference.yaml"
            )
        )
        parser_tester.check_keywords(test_case, cp2k_version, reference_values, ptype)


# @pytest.mark.parametrize(
#     "test_case, print_level, cp2k_version",
#     [("cell_opt", "medium", 8.1), ("cell_opt", "medium", 8.2)],
# )
# def test_comp_cp2k_parser_cell_opt(test_case, print_level, cp2k_version):
#     """
#     Compare cell-opt outputs with the parser of the official cp2k-plugin.
#     """
#     parser_tester = OutputParserTester()
#     parser_tester.add_output_file(test_case, print_level, cp2k_version, incl_cp2k_parser=True)
#
#     items_compare_aiidacp2k = [
#         {"key": "energy"},
#         {"key": "cp2k_version"},
#         {"key": "natoms"},
#         {"key": "energy_scf"},
#         {"key": "dft_type"},
#         {"key": ["motion_step_info", "max_step_au"], "excl_aiidacp2k": [0]},
#         {"key": ["motion_step_info", "rms_step_au"], "excl_aiidacp2k": [0]},
#         {"key": ["motion_step_info", "max_grad_au"], "excl_aiidacp2k": [0]},
#         {"key": ["motion_step_info", "rms_grad_au"], "excl_aiidacp2k": [0]},
#         {"key": ["motion_step_info", "cell_vol_angs3"], "excl_parser": [-1]},
#         {"key": ["motion_step_info", "cell_a_angs"], "excl_parser": [-1]},
#         {"key": ["motion_step_info", "cell_b_angs"], "excl_parser": [-1]},
#         {"key": ["motion_step_info", "cell_c_angs"], "excl_parser": [-1]},
#         {"key": ["motion_step_info", "cell_alp_deg"], "excl_parser": [-1]},
#         {"key": ["motion_step_info", "cell_bet_deg"], "excl_parser": [-1]},
#         {"key": ["motion_step_info", "cell_gam_deg"], "excl_parser": [-1]},
#         {"key": ["motion_step_info", "edens_rspace"], "excl_parser": [-1]},
#         {"key": ["motion_step_info", "energy_au"], "excl_parser": [-1]},
#         {"key": ["motion_step_info", "pressure_bar"]},
#     ]
#     # Compare information with aiida-cp2k parser:
#     parser_tester.compare_aiidacp2k(test_case, cp2k_version, items_compare_aiidacp2k, "standard")


# @pytest.mark.parametrize(
#     "test_case, print_level, cp2k_version",
#     [("geo_opt", "medium", 8.1), ("geo_opt", "medium", 8.2)],
# )
# def test_comp_cp2k_parser_geo_opt(test_case, print_level, cp2k_version):
#     """
#     Compare geo-opt outputs with the parser of the official cp2k-plugin.
#     """
#     parser_tester = OutputParserTester()
#     parser_tester.add_output_file(test_case, print_level, cp2k_version, incl_cp2k_parser=True)
#
#     items_compare_aiidacp2k = [
#         {"key": "energy"},
#         {"key": "cp2k_version"},
#         {"key": "natoms"},
#         {"key": "energy_scf"},
#         {"key": "dft_type"},
#         {"key": ["motion_step_info", "max_step_au"], "excl_aiidacp2k": [0]},
#         {"key": ["motion_step_info", "rms_step_au"], "excl_aiidacp2k": [0]},
#         {"key": ["motion_step_info", "max_grad_au"], "excl_aiidacp2k": [0]},
#         {"key": ["motion_step_info", "rms_grad_au"], "excl_aiidacp2k": [0]},
#         {"key": ["motion_step_info", "edens_rspace"], "excl_parser": [-1]},
#         {"key": ["motion_step_info", "energy_au"], "excl_parser": [-1]},
#     ]
#
#     # Compare information with aiida-cp2k parser:
#     parser_tester.compare_aiidacp2k(test_case, cp2k_version, items_compare_aiidacp2k, "standard")


# @pytest.mark.parametrize(
#     "test_case, print_level, cp2k_version",
#     [
#         ("bands", "medium", 8.1),
#         ("bands_spin_pol", "medium", 8.1),
#         ("bands", "medium", 8.2),
#         ("bands_spin_pol", "medium", 8.2),
#     ],
# )
# def test_comp_cp2k_parser_bands(test_case, print_level, cp2k_version):
#     """
#     Compare band structure outputs with the parser of the official cp2k-plugin.
#     """
#     parser_tester = OutputParserTester()
#     parser_tester.add_output_file(test_case, print_level, cp2k_version, incl_cp2k_parser=True)
#
#     items_compare_aiidacp2k = [
#         {"key": "energy"},
#         {"key": "cp2k_version"},
#         {"key": "natoms"},
#         {"key": "energy_scf"},
#         {"key": "dft_type"},
#         {"key": ["kpoint_data", "kpoints"]},
#         {"key": ["kpoint_data", "bands"]},
#         {"key": ["kpoint_data", "bands_unit"]},
#     ]
#
#     # Compare information with aiida-cp2k parser:
#     parser_tester.compare_aiidacp2k(test_case, cp2k_version, items_compare_aiidacp2k, "standard")


@pytest.mark.parametrize(
    "test_case, print_level, cp2k_version_1, cp2k_version_2, exclude_keys",
    [
        ("geo_opt", "medium", 8.1, 8.2, ["nwarnings", "runtime"]),
        ("cell_opt", "medium", 8.1, 8.2, ["nwarnings", "runtime"]),
        ("md_nvt", "medium", 8.1, 8.2, ["nwarnings", "runtime"]),
        ("eigenvalues", "low", 8.1, 8.2, ["nwarnings", "runtime"]),
        ("eigenvalues", "medium", 8.1, 8.2, ["nwarnings", "runtime"]),
        ("eigenvalues_spin_pol", "medium", 8.1, 8.2, ["nwarnings", "runtime"]),
        ("bands", "medium", 8.1, 8.2, ["nwarnings", "runtime"]),
        ("bands_spin_pol", "medium", 8.1, 8.2, ["nwarnings", "runtime"]),
        ("geo_opt", "medium", 8.2, 9.1, ["runtime"]),
        ("cell_opt", "medium", 8.2, 9.1, ["runtime"]),
        ("md_nvt", "low", 8.2, 9.1, ["runtime"]),
        ("eigenvalues", "low", 8.2, 9.1, ["runtime"]),
        ("eigenvalues", "medium", 8.2, 9.1, ["runtime"]),
        ("eigenvalues_spin_pol", "medium", 8.2, 9.1, ["runtime"]),
        ("bands", "medium", 8.2, 9.1, ["runtime"]),
        ("bands_spin_pol", "medium", 8.2, 9.1, ["runtime"]),
    ],
)
def test_comp_standard_parser_versions(
    test_case, print_level, cp2k_version_1, cp2k_version_2, exclude_keys
):
    """Compare different versions of the output-parser."""
    parser_tester = OutputParserTester()
    parser_tester.add_output_file(test_case, print_level, cp2k_version_1, incl_cp2k_parser=False)
    parser_tester.add_output_file(test_case, print_level, cp2k_version_2, incl_cp2k_parser=False)
    # for parser_type in ["standard"]: #, "partial_charges", "trajectory"]:
    parser_tester.compare_parser_versions(
        test_case, cp2k_version_1, cp2k_version_2, "standard", exclude_keys
    )


@pytest.mark.parametrize(
    "test_case, print_level, cp2k_version_1, cp2k_version_2, exclude_keys",
    [
        ("geo_opt", "medium", 8.1, 8.2, ["nwarnings", "hirshfeld", "runtime"]),
        ("cell_opt", "medium", 8.1, 8.2, ["nwarnings", "runtime"]),
        ("md_nvt", "medium", 8.1, 8.2, ["nwarnings", "runtime"]),
        ("eigenvalues_spin_pol", "medium", 8.1, 8.2, ["nwarnings", "runtime"]),
        ("geo_opt", "medium", 8.2, 9.1, ["runtime"]),
        ("cell_opt_spgr", "medium", 8.2, 9.1, ["runtime"]),
        ("md_nvt", "medium", 8.2, 9.1, ["runtime"]),
        ("eigenvalues_spin_pol", "medium", 8.2, 9.1, ["runtime"]),
    ],
)
def test_comp_partial_charges_parser_versions(
    test_case, print_level, cp2k_version_1, cp2k_version_2, exclude_keys
):
    """Compare different versions of the output-parser."""
    parser_tester = OutputParserTester()
    parser_tester.add_output_file(test_case, print_level, cp2k_version_1, incl_cp2k_parser=False)
    parser_tester.add_output_file(test_case, print_level, cp2k_version_2, incl_cp2k_parser=False)
    parser_tester.compare_parser_versions(
        test_case, cp2k_version_1, cp2k_version_2, "partial_charges", exclude_keys
    )


@pytest.mark.parametrize(
    "test_case, print_level, cp2k_version_1, cp2k_version_2, exclude_keys",
    [
        ("md_nvt", "medium", 8.1, 8.2, ["nwarnings", "runtime"]),
        ("eigenvalues_spin_pol", "medium", 8.2, 9.1, ["runtime"]),
        ("geo_opt", "medium", 8.2, 9.1, ["runtime"]),
        ("cell_opt", "medium", 8.2, 9.1, ["runtime"]),
        ("md_nvt", "medium", 8.2, 9.1, ["runtime"]),
    ],
)
def test_comp_trajectory_parser_versions(
    test_case, print_level, cp2k_version_1, cp2k_version_2, exclude_keys
):
    """Compare different versions of the output-parser."""
    parser_tester = OutputParserTester()
    parser_tester.add_output_file(test_case, print_level, cp2k_version_1, incl_cp2k_parser=False)
    parser_tester.add_output_file(test_case, print_level, cp2k_version_2, incl_cp2k_parser=False)
    parser_tester.compare_parser_versions(
        test_case, cp2k_version_1, cp2k_version_2, "trajectory", exclude_keys
    )
