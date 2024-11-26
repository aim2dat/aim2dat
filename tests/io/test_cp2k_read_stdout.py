"""Tests for the cp2k main-output parser."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.io.cp2k import read_stdout
from aim2dat.io.yaml import load_yaml_file


MAIN_PATH = os.path.dirname(__file__) + "/cp2k_stdout/"


class OutputParserTester:
    """Class to compare output-dictionaries."""

    def __init__(self):
        """Initialize class."""
        self.result_dicts = {}

    def add_output_file(self, test_case, print_level, cp2k_version):
        """Add output-file to the class."""
        self.result_dicts[test_case + "-" + str(cp2k_version)] = {
            "standard": read_stdout(
                MAIN_PATH + f"cp2k-{cp2k_version}/{test_case}_{print_level}/aiida.out", "standard"
            ),
            "partial_charges": read_stdout(
                MAIN_PATH + f"cp2k-{cp2k_version}/{test_case}_{print_level}/aiida.out",
                "partial_charges",
            ),
            "trajectory": read_stdout(
                MAIN_PATH + f"cp2k-{cp2k_version}/{test_case}_{print_level}/aiida.out",
                "trajectory",
            ),
        }

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
        ("eigenvalues", "low", 8.1),
        ("eigenvalues_changing_character", "low", 8.1),
        ("cholesky_decompose_failed", "medium", 8.2),
        ("cell_opt_interrupted", "low", 9.1),
        ("cell_opt_max_steps", "low", 9.1),
        ("geo_opt", "low", 2024.1),
        ("geo_opt", "medium", 2024.1),
        ("cell_opt", "low", 2024.1),
        ("cell_opt", "medium", 2024.1),
        ("cell_opt_1kpoint", "medium", 2024.1),
        ("cell_opt_interrupted", "low", 2024.1),
        ("cell_opt_spgr", "low", 2024.1),
        ("cell_opt_spgr", "medium", 2024.1),
        ("cell_opt_cg_spgr", "medium", 2024.1),
        ("cell_opt_max_steps", "low", 2024.1),
        ("cell_opt_walltime", "low", 2024.1),
        ("md_nvt", "low", 2024.1),
        ("md_nvt", "medium", 2024.1),
        ("eigenvalues", "low", 2024.1),
        ("eigenvalues", "medium", 2024.1),
        ("eigenvalues_spin_pol", "low", 2024.1),
        ("eigenvalues_spin_pol", "medium", 2024.1),
        ("eigenvalues_changing_character", "low", 2024.1),
        ("eigenvalues_no_kpoints", "low", 2024.1),
        ("eigenvalues_spin_pol_no_kpoints", "low", 2024.1),
        ("bands", "low", 2024.1),
        ("bands", "medium", 2024.1),
        ("bands_spin_pol", "low", 2024.1),
        ("bands_spin_pol", "medium", 2024.1),
        ("smearing_need_added_mos", "medium", 2024.1),
        ("need_lsd", "medium", 2024.1),
        ("unconverged_scf", "medium", 2024.1),
    ],
)
def test_mainoutput(test_case, print_level, cp2k_version):
    """
    Test the cp2k output-parser.
    """
    parser_tester = OutputParserTester()
    parser_tester.add_output_file(test_case, print_level, cp2k_version)
    parser_types = ["standard", "partial_charges", "trajectory"]
    for ptype in parser_types:
        reference_values = list(
            load_yaml_file(
                MAIN_PATH + f"cp2k-{cp2k_version}/{test_case}_{print_level}_{ptype}_reference.yaml"
            )
        )
        parser_tester.check_keywords(test_case, cp2k_version, reference_values, ptype)


@pytest.mark.parametrize(
    "test_case, print_level, cp2k_version_1, cp2k_version_2, exclude_keys",
    [
        ("eigenvalues_spin_pol", "medium", 8.1, 8.2, ["nwarnings", "runtime"]),
        ("geo_opt", "medium", 9.1, 2024.1, ["runtime"]),
        ("cell_opt", "medium", 8.2, 9.1, ["runtime"]),
        ("md_nvt", "low", 9.1, 2024.1, ["runtime"]),
        ("eigenvalues", "low", 9.1, 2024.1, ["runtime"]),
        ("eigenvalues", "medium", 9.1, 2024.1, ["runtime"]),
        ("eigenvalues_spin_pol", "medium", 9.1, 2024.1, ["runtime"]),
        ("bands", "medium", 9.1, 2024.1, ["runtime", "xc", "nwarnings"]),
        ("bands_spin_pol", "medium", 9.1, 2024.1, ["runtime"]),
    ],
)
def test_comp_standard_parser_versions(
    test_case, print_level, cp2k_version_1, cp2k_version_2, exclude_keys
):
    """Compare different versions of the output-parser."""
    parser_tester = OutputParserTester()
    parser_tester.add_output_file(test_case, print_level, cp2k_version_1)
    parser_tester.add_output_file(test_case, print_level, cp2k_version_2)
    parser_tester.compare_parser_versions(
        test_case, cp2k_version_1, cp2k_version_2, "standard", exclude_keys
    )


@pytest.mark.parametrize(
    "test_case, print_level, cp2k_version_1, cp2k_version_2, exclude_keys",
    [
        ("geo_opt", "medium", 9.1, 2024.1, ["runtime"]),
        ("cell_opt_spgr", "medium", 8.2, 9.1, ["runtime"]),
        ("md_nvt", "medium", 9.1, 2024.1, ["runtime"]),
        ("eigenvalues_spin_pol", "medium", 9.1, 2024.1, ["runtime"]),
    ],
)
def test_comp_partial_charges_parser_versions(
    test_case, print_level, cp2k_version_1, cp2k_version_2, exclude_keys
):
    """Compare different versions of the output-parser."""
    parser_tester = OutputParserTester()
    parser_tester.add_output_file(test_case, print_level, cp2k_version_1)
    parser_tester.add_output_file(test_case, print_level, cp2k_version_2)
    parser_tester.compare_parser_versions(
        test_case, cp2k_version_1, cp2k_version_2, "partial_charges", exclude_keys
    )


@pytest.mark.parametrize(
    "test_case, print_level, cp2k_version_1, cp2k_version_2, exclude_keys",
    [
        ("eigenvalues_spin_pol", "medium", 9.1, 2024.1, ["runtime"]),
        ("geo_opt", "medium", 9.1, 2024.1, ["runtime"]),
        ("cell_opt", "medium", 8.2, 9.1, ["runtime"]),
        ("md_nvt", "medium", 9.1, 2024.1, ["runtime"]),
    ],
)
def test_comp_trajectory_parser_versions(
    test_case, print_level, cp2k_version_1, cp2k_version_2, exclude_keys
):
    """Compare different versions of the output-parser."""
    parser_tester = OutputParserTester()
    parser_tester.add_output_file(test_case, print_level, cp2k_version_1)
    parser_tester.add_output_file(test_case, print_level, cp2k_version_2)
    parser_tester.compare_parser_versions(
        test_case, cp2k_version_1, cp2k_version_2, "trajectory", exclude_keys
    )
