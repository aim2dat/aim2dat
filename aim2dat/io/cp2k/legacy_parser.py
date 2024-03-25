"""Containing all functions and classes that parse output-files of cp2k."""

# Standard library imports
import re
import os

# Internal library imports
from aim2dat.utils.units import energy
from aim2dat.io.yaml import load_yaml_file

cwd = os.path.dirname(__file__)


class Parser:
    """Main class to parse CP2K output."""

    def __init__(self, fstring):
        """Initialize class."""
        self.output_dict = {}

        # Lines and current line index:
        self._line_idx = 0
        self.lines = fstring.splitlines()

        # Patterns to parse sections (filled with the yaml-file):
        self._pattern_dict = None

        # Current blocks and general information to be parsed:
        self._current_patterns = None
        self._current_blocks = None
        self._current_block = None
        self._in_block = False
        self._block_idx = 0
        self._block_length = 0
        self._block_key = None
        self._block_ldict = None

    def _check_general_patterns(self):
        if self._current_patterns and self._line_idx < len(self.lines):
            for pattern in self._current_patterns:
                value = self._parse_pattern(pattern)
                if value:
                    self._update_dict(self.output_dict, pattern, value)

    def _check_blocks(self):
        if self._line_idx < len(self.lines):
            # Check if there are blocks:
            if self._current_blocks:
                # Check if the end of a block is reached:
                if self._in_block:
                    self._check_block_end()

                # Check if a block could start:
                if not self._in_block:
                    self._check_block_start()

                if (
                    self._in_block
                    and self._current_block
                    and self._block_idx < len(self._current_block["patterns"])
                ):
                    self._check_block_pattern()
                self._block_length += 1

    def _check_block_start(self):
        for block in self._current_blocks:
            should_start = False
            regex = False
            if block.get("regex"):
                if block["regex"]:
                    regex = True

            # Check if start pattern matches
            if not isinstance(block["start"], list):
                start_patterns = [block["start"]]
            else:
                start_patterns = block["start"]
            for start_pattern in start_patterns:
                if regex:
                    value = re.findall(start_pattern, self.lines[self._line_idx])
                    if len(value) > 0:
                        should_start = True
                else:
                    if self.lines[self._line_idx].startswith(start_pattern):
                        should_start = True

            if should_start:
                self._current_block = block
                self._in_block = True
                self._block_idx = 0
                self._block_length = 0
                self._block_dict = {}

    def _check_block_pattern(self):
        # Check if pattern is found in line
        pattern = self._current_block["patterns"][self._block_idx]
        value = self._parse_pattern(pattern)

        # If yes, add value to output_dict
        if value:
            if (
                "store_as_block_list" in self._current_block
                and self._current_block["store_as_block_list"]
            ):
                self._update_dict(self._block_dict, pattern, value)
            else:
                self._update_dict(self.output_dict, pattern, value)

            if pattern.get("repeat"):
                if not pattern["repeat"]:
                    self._block_idx += 1
            else:
                self._block_idx += 1

    def _check_block_end(self):
        should_end = False
        regex = False
        if self._current_block.get("regex"):
            if self._current_block["regex"]:
                regex = True

        # Check if maximum lines are exceeded
        if self._current_block["max_length"] > -1:
            if self._block_length > self._current_block["max_length"]:
                should_end = True

        # Check if end pattern matches
        if not isinstance(self._current_block["end"], list):
            end_patterns = [self._current_block["end"]]
        else:
            end_patterns = self._current_block["end"]
        for end_pattern in end_patterns:
            if regex:
                value = re.findall(end_pattern, self.lines[self._line_idx])
                if len(value) > 0:
                    should_end = True
            else:
                if self.lines[self._line_idx].startswith(end_pattern):
                    should_end = True

        # Set in_block to false
        if should_end:
            if (
                "store_as_block_list" in self._current_block
                and self._current_block["store_as_block_list"]
            ):
                # Does not work for nested dictionaries
                for key, value in self._block_dict.items():
                    pattern = {"key": [[key]], "append": [True], "type": "list"}
                    self._update_dict(self.output_dict, pattern, [value])
            self._current_block = None
            self._in_block = False

    def _parse_pattern(self, pattern):
        value = None
        if pattern["regex"]:
            value = re.findall(pattern["pattern"], self.lines[self._line_idx])
            if len(value) == 0:
                value = None
            else:
                if isinstance(value[0], tuple):
                    value = value[0]
                else:
                    value = [value[0]]
        else:
            if self.lines[self._line_idx].startswith(pattern["pattern"]):
                line_splitted = self.lines[self._line_idx].split()
                value = [line_splitted[pos] for pos in pattern["position"]]

        value_new = None
        if value:
            types = pattern["type"]
            if isinstance(pattern["type"], str):
                types = [pattern["type"]] * len(value)
            if all(type0 == "str" for type0 in types):
                if pattern.get("concatenate_str"):
                    value_new = [" ".join([pos for pos in value])]
                else:
                    value_new = [pos for pos in value]
            else:
                value_new = []
                for val, type0 in zip(value, types):
                    if type0 == "str":
                        value_new.append(val)
                    elif type0 == "int":
                        try:
                            value_new.append(int(val))
                        except ValueError:
                            value_new.append(None)
                    elif type0 == "float":
                        try:
                            value_new.append(float(val))
                        except ValueError:
                            value_new.append(None)
        return value_new

    @staticmethod
    def _update_dict(dictionary, pattern, value):
        # create keys in case they don't exist:
        for key_idx, key_list in enumerate(pattern["key"]):
            dict_helper = dictionary
            for key in key_list[:-1]:
                dict_helper = dict_helper.setdefault(key, {})

            # Update output_dict:
            if pattern["append"][key_idx]:
                if key_list[-1] not in dict_helper:
                    dict_helper[key_list[-1]] = value[key_idx]
                else:
                    if pattern["type"] == "str" and pattern.get("append_connector"):
                        dict_helper[key_list[-1]] = (
                            dict_helper[key_list[-1]]
                            + pattern["append_connector"][key_idx]
                            + value[key_idx]
                        )
                    elif pattern["type"] == "list":
                        if isinstance(dict_helper[key_list[-1]], list):
                            dict_helper[key_list[-1]] = [dict_helper[key_list[-1]], value[key_idx]]
                        else:
                            dict_helper[key_list[-1]] = [
                                [dict_helper[key_list[-1]]],
                                value[key_idx],
                            ]
                    else:
                        if isinstance(dict_helper[key_list[-1]], list):
                            dict_helper[key_list[-1]].append(value[key_idx])
                        else:
                            dict_helper[key_list[-1]] = [dict_helper[key_list[-1]], value[key_idx]]
            else:
                dict_helper[key_list[-1]] = value[key_idx]


class MainOutputParser(Parser):
    """Parse CP2K output into a dictionary."""

    def __init__(self, fstring):
        """Initialize class."""
        super().__init__(fstring)

        # errors and warnings
        self._errors = None
        self._warnings = None

    def retrieve_result_dict(self, parser_type):
        """Retrieve the result dictionary."""
        self.output_dict = {"exceeded_walltime": False, "warnings": []}
        program_version = self._get_cp2k_version()
        if program_version is None:
            self.output_dict["incomplete"] = True
            return self.output_dict
        if program_version < 8.1:
            self.output_dict["incompatible_code"] = True
            return self.output_dict
        self.output_dict["cp2k_version"] = program_version
        self._line_idx = 0
        self._read_patterns_from_yaml(parser_type, program_version)
        self._parse_header()
        self._parse_middle_part()
        if parser_type == "standard":
            if self.output_dict.get("bands_data"):
                self._process_bands_v81()
            self._process_eigenvalues()
        if parser_type == "partial_charges":
            self._process_kind()
            self._process_partial_charges()
        if parser_type == "trajectory":
            self._process_kind()
            self._process_motion_step_info()
        self._process_electron_numbers()
        self._process_scf_convergence()
        self._parse_footer()
        self.output_dict.update(self._pattern_dict.get("extra_entries"))

        # The number of warnings is the very last information that is parsed.
        # If it is not in the dictionary the run didn't finish properly:
        if "nwarnings" not in self.output_dict:
            self.output_dict["interrupted"] = True

        return self.output_dict

    def _read_patterns_from_yaml(self, parser_type, program_version):
        prg_v_str = str(program_version).replace(".", "")
        file_path = cwd + f"/parameter_files/parser_main_base_v{prg_v_str}.yaml"
        self._pattern_dict = dict(load_yaml_file(file_path))
        file_path = cwd + f"/parameter_files/parser_main_{parser_type}_v{prg_v_str}.yaml"
        extra_patterns = dict(load_yaml_file(file_path))
        for section in [
            "header_general",
            "header_blocks",
            "middle_general",
            "middle_blocks",
            "footer_general",
            "footer_blocks",
        ]:
            if section in extra_patterns:
                self._pattern_dict[section] += extra_patterns[section]
        file_path = cwd + "/parameter_files/parser_main_errors_and_warnings.yaml"
        pattern_dict = dict(load_yaml_file(file_path))
        self._errors = pattern_dict["errors"]
        self._warnings = pattern_dict["warnings"]

    def _parse_header(self):
        header_end = [" Number of electrons:"]
        self._current_patterns = self._pattern_dict["header_general"]
        self._current_blocks = self._pattern_dict["header_blocks"]
        self._current_block = None
        self._in_block = False
        self._block_idx = 0
        self._block_length = 0

        # Loop until the end of the header is reached:
        while all([end0 not in self.lines[self._line_idx] for end0 in header_end]):
            # Check for warnings and errors at each line:
            self._parse_warnings_and_errors()

            # Check general information that should be part of the output file in any case:
            self._check_general_patterns()

            # Check special blocks (not always part of output):
            self._check_blocks()

            self._line_idx += 1
            if self._line_idx >= len(self.lines):
                break

    def _parse_middle_part(self):
        middle_end = "DBCSR STATISTICS"
        self._current_patterns = self._pattern_dict["middle_general"]
        self._current_blocks = self._pattern_dict["middle_blocks"]
        self._current_block = None
        self._in_block = False
        self._block_idx = 0
        self._block_length = 0

        while (self._line_idx < len(self.lines)) and (
            middle_end not in self.lines[self._line_idx]
        ):
            # Check for warnings and errors at each line:
            self._parse_warnings_and_errors()

            # Check general information that should be part of the output file in any case:
            self._check_general_patterns()

            # Check blocks:
            self._check_blocks()

            self._line_idx += 1

    def _parse_footer(self):
        self._current_patterns = self._pattern_dict["footer_general"]
        self._current_blocks = self._pattern_dict["footer_blocks"]
        self._current_block = None
        self._in_block = False
        self._block_idx = 0
        self._block_length = 0

        while self._line_idx < len(self.lines):
            # Check for warnings and errors at each line:
            self._parse_warnings_and_errors()

            # Check general information that should be part of the output file in any case:
            self._check_general_patterns()

            # Check blocks:
            self._check_blocks()

            self._line_idx += 1
            if self._line_idx >= len(self.lines):
                break

    def _parse_warnings_and_errors(self):
        """Search line for errors and warnings."""
        if self._line_idx < len(self.lines):
            line = self.lines[self._line_idx]
            # Warnings
            for warning_str in self._warnings:
                if warning_str[0] in line and not warning_str[1] in self.output_dict["warnings"]:
                    self.output_dict["warnings"].append(warning_str[1])
            if "The number of warnings for this run is" in line:
                self.output_dict["nwarnings"] = int(line.split()[-1])
            # Errors
            for error_str in self._errors:
                if error_str[0] in line:
                    self.output_dict[error_str[1]] = True

    def _get_cp2k_version(self):
        program_version = None
        for line in self.lines:
            if line.startswith(" CP2K| version string:"):
                program_version = float(line.split()[5])
                break
        return program_version

    def _process_bands_v81(self):
        bands_data = self.output_dict["bands_data"]

        # Convert labels
        label_positions = []
        for idx, nr_points in enumerate(bands_data["nr_points_in_set"]):
            if len(label_positions) == 0:
                label_positions.append(0)
            else:
                label_positions.append(label_positions[-1] + 1)
            label_positions.append(label_positions[-1] + nr_points - 1)
        labels = []
        for label, label_pos in zip(bands_data["labels"], label_positions):
            if label == "not":
                label = None
            labels.append([label_pos, label])

        # Convert k-points:
        kpoints = []
        for spin, kpoint_x, kpoint_y, kpoint_z in zip(
            bands_data["spin"], bands_data["pos_x"], bands_data["pos_y"], bands_data["pos_z"]
        ):
            if spin == 1:
                kpoints.append([kpoint_x, kpoint_y, kpoint_z])

        # Convert energies and occupations:
        bands = []
        occupations = []
        last_bnd_nr = 0
        bands0 = []
        occupations0 = []
        for bnd_nr, energy0, occupation in zip(
            bands_data["band_nr"], bands_data["energies"], bands_data["occ"]
        ):
            if bnd_nr <= last_bnd_nr:
                bands.append(bands0)
                occupations.append(occupations0)
                bands0 = []
                occupations0 = []
            bands0.append(energy0)
            occupations0.append(occupation)
            last_bnd_nr = bnd_nr
        bands.append(bands0)
        occupations.append(occupations0)

        # Split bands for spin-polarized calculation:
        if max(bands_data["spin"]) > 1:
            bands_spin = [[] for idx in range(max(bands_data["spin"]))]
            occupations_spin = [[] for idx in range(max(bands_data["spin"]))]
            for idx, spin in enumerate(bands_data["spin"]):
                bands_spin[spin - 1].append(bands[idx])
                occupations_spin[spin - 1].append(occupations[idx])
            bands = bands_spin
            occupations = occupations_spin

        # Update output_dict
        self.output_dict["kpoint_data"] = {
            "labels": labels,
            "kpoints": kpoints,
            "bands": bands,
            "occupations": occupations,
            "bands_unit": "eV",
        }
        del self.output_dict["bands_data"]

    def _process_eigenvalues(self):
        if "eigenvalues_raw" in self.output_dict:
            ev_raw = self.output_dict.pop("eigenvalues_raw")
            ev_list = []
            for label, value in ev_raw.items():
                if not isinstance(value, list):
                    ev_raw[label] = [value]
            if "kpoint_nrs" in ev_raw:
                nr_states = max(ev_raw["mo_indices"])
                gaps = []
                vbms = []
                cbms = []
                for kpoint_nr, spin in zip(ev_raw["kpoint_nrs"], ev_raw["spin"]):
                    if kpoint_nr is None:
                        kpoint_nr = 1
                    occs = ev_raw["occupations"][
                        (kpoint_nr - 1) * nr_states : kpoint_nr * nr_states
                    ]
                    evs = ev_raw["eigenvalues"][
                        (kpoint_nr - 1) * nr_states : kpoint_nr * nr_states
                    ]
                    gap = None
                    for state_idx, (occ0, ev0) in enumerate(zip(occs, evs)):
                        if occ0 < 0.5 and state_idx > 0:
                            gap = ev0 - evs[state_idx - 1]
                            vbms.append(evs[state_idx - 1])
                            cbms.append(ev0)
                            gaps.append(gap)
                            break
                    if spin == "BETA":
                        ev_list[-1]["occupations"] = [ev_list[-1]["occupations"], occs]
                        ev_list[-1]["energies"] = [ev_list[-1]["energies"], evs]
                        ev_list[-1]["gap"] = [ev_list[-1]["gap"], gap]
                    else:
                        ev_item = {
                            "occupations": occs,
                            "energies": evs,
                            "gap": gap,
                        }
                        if "bz_kpoint_nrs" in ev_raw:
                            bz_idx = ev_raw["bz_kpoint_nrs"].index(kpoint_nr)
                            ev_item["kpoint"] = [
                                ev_raw["x"][bz_idx],
                                ev_raw["y"][bz_idx],
                                ev_raw["z"][bz_idx],
                            ]
                            ev_item["weight"] = ev_raw["weight"][bz_idx]
                        ev_list.append(ev_item)
                self.output_dict["eigenvalues_info"] = {
                    "eigenvalues": ev_list,
                    "direct_gap": max(min(gaps), 0.0),
                    "gap": max(min(cbms) - max(vbms), 0.0),
                }

    def _process_electron_numbers(self):
        if "nr_spins" in self.output_dict:
            del self.output_dict["nr_spins"]
            if "added_mos_up" in self.output_dict:
                added_mos = [
                    self.output_dict["added_mos_up"],
                    self.output_dict["added_mos_down"],
                ]
                self.output_dict["nr_unocc_orbitals"] = added_mos
                del self.output_dict["added_mos_up"]
                del self.output_dict["added_mos_down"]
            for label in ["nelectrons", "nr_occ_orbitals"]:
                if label in self.output_dict and isinstance(self.output_dict[label], list):
                    self.output_dict[label] = self.output_dict[label][-2:]
        else:
            if "added_mos_up" in self.output_dict:
                self.output_dict["nr_unocc_orbitals"] = self.output_dict["added_mos_up"]
                del self.output_dict["added_mos_up"]
                del self.output_dict["added_mos_down"]
            for label in ["nelectrons", "nr_occ_orbitals"]:
                if label in self.output_dict and isinstance(self.output_dict[label], list):
                    self.output_dict[label] = self.output_dict[label][-1]

    def _process_scf_convergence(self):
        if "scf_converged" in self.output_dict:
            if isinstance(self.output_dict["scf_converged"], int):
                self.output_dict["scf_converged"] = True
            else:
                self.output_dict["scf_converged"] = False

    def _process_kind(self):
        if "kind_info" in self.output_dict:
            kind_info = self.output_dict.pop("kind_info")
            for key0 in ["kind", "element", "atomic_nr", "valence_e", "mass"]:
                if not isinstance(kind_info[key0], list):
                    kind_info[key0] = [kind_info[key0]]
            self.output_dict["kind_info"] = []
            for kind, el, at_nr, val_el, mass in zip(
                kind_info["kind"],
                kind_info["element"],
                kind_info["atomic_nr"],
                kind_info["valence_e"],
                kind_info["mass"],
            ):
                self.output_dict["kind_info"].append(
                    {
                        "kind": kind,
                        "element": el,
                        "atomic_nr": at_nr,
                        "core_electrons": int(at_nr - val_el),
                        "valence_electrons": int(val_el),
                        "mass": mass,
                    }
                )

    def _process_partial_charges(self):
        if "natoms" in self.output_dict:
            nat = self.output_dict["natoms"]
            for pc_label in ["mulliken", "hirshfeld"]:
                if pc_label in self.output_dict:
                    pc_dict = self.output_dict[pc_label]
                    for label, value in pc_dict.items():
                        if not isinstance(value, list):
                            pc_dict[label] = [value]
                    self.output_dict[pc_label] = []
                    if "population_alpha" in pc_dict:
                        population = [
                            [pop_a, pop_b]
                            for pop_a, pop_b in zip(
                                pc_dict["population_alpha"], pc_dict["population_beta"]
                            )
                        ]
                    else:
                        population = pc_dict["population"]
                    for kind, element, pop, charge in zip(
                        pc_dict["kind"][-nat:],
                        pc_dict["element"][-nat:],
                        population[-nat:],
                        pc_dict["charge"][-nat:],
                    ):
                        self.output_dict[pc_label].append(
                            {
                                "kind": kind,
                                "element": element,
                                "population": pop,
                                "charge": charge,
                            }
                        )

    def _process_motion_step_info(self):
        # TODO: Fix for CG optimization...

        labels_step2 = ["max_step", "rms_step", "max_grad", "rms_grad"]
        labels_step1 = ["pressure", "energy_drift_p_atom"]
        if "motion_step_info" in self.output_dict:
            m_step_info = []
            m_step_info_raw = self.output_dict.pop("motion_step_info")
            # scf_converged is obsolete in case of outer SCF loop:
            if "outer_scf_converged" in m_step_info_raw:
                del m_step_info_raw["scf_converged"]
            # Add step nr and time for 0th step.
            if "step_nr" in m_step_info_raw:
                m_step_info_raw["step_nr"].insert(0, m_step_info_raw["step_nr"][0] - 1)
            if "time_fs" in m_step_info_raw:
                m_step_info_raw["time_fs"].insert(
                    0, m_step_info_raw["time_fs"][0] - m_step_info_raw["time_step"]
                )
            if "time_step" in m_step_info_raw:
                del m_step_info_raw["time_step"]

            nat = self.output_dict.get("natoms", 0)
            # Make sure that all items are lists:
            for pc_label in ["mulliken", "hirshfeld"]:
                if pc_label in self.output_dict:
                    for label, value in self.output_dict[pc_label].items():
                        if not isinstance(value, list):
                            self.output_dict[pc_label][label] = [value]
            for label, value in m_step_info_raw.items():
                if not isinstance(value, list):
                    m_step_info_raw[label] = [value]
            for step_idx in range(len(m_step_info_raw["energy"])):
                m_step_item = {}
                for label, list0 in m_step_info_raw.items():
                    if label == "scf_converged" or label == "outer_scf_converged":
                        if isinstance(list0[step_idx], int):
                            m_step_item["scf_converged"] = True
                            m_step_item["nr_scf_steps"] = list0[step_idx]
                        else:
                            m_step_item["scf_converged"] = False
                            m_step_item["nr_scf_steps"] = None
                    elif label in labels_step1:
                        if step_idx > 0 and step_idx - 1 < len(list0):
                            m_step_item[label] = list0[step_idx - 1]
                    elif label in labels_step2:
                        if step_idx > 1 and step_idx - 2 < len(list0):
                            m_step_item[label] = list0[step_idx - 2]
                    else:
                        m_step_item[label] = list0[step_idx]
                if nat > 0:
                    for pc_label in ["mulliken", "hirshfeld"]:
                        pc_dict = self.output_dict.get(pc_label, None)
                        if pc_dict is not None:
                            pc_list = []
                            if "population_alpha" in pc_dict:
                                population = [
                                    [pop_a, pop_b]
                                    for pop_a, pop_b in zip(
                                        pc_dict["population_alpha"], pc_dict["population_beta"]
                                    )
                                ]
                            else:
                                population = pc_dict["population"]
                            for kind, element, pop, charge in zip(
                                pc_dict["kind"][step_idx * nat : (step_idx + 1) * nat],
                                pc_dict["element"][step_idx * nat : (step_idx + 1) * nat],
                                population[step_idx * nat : (step_idx + 1) * nat],
                                pc_dict["charge"][step_idx * nat : (step_idx + 1) * nat],
                            ):
                                pc_list.append(
                                    {
                                        "kind": kind,
                                        "element": element,
                                        "population": pop,
                                        "charge": charge,
                                    }
                                )
                            m_step_item[pc_label] = pc_list
                m_step_info.append(m_step_item)
            for pc_label in ["mulliken", "hirshfeld"]:
                if pc_label in self.output_dict:
                    del self.output_dict[pc_label]
            self.output_dict["motion_step_info"] = m_step_info


class BandStructureParser(MainOutputParser):
    """Band structure parser (work in progress)."""

    pass


class PDOSParser(Parser):
    """Parser for the projected density of states."""

    _orbital_labels = [
        "s",
        "px",
        "py",
        "pz",
        "d-2",
        "d-1",
        "d0",
        "d+1",
        "d+2",
        "f-3",
        "f-2",
        "f-1",
        "f0",
        "f+1",
        "f+2",
        "f+3",
    ]

    def __init__(self):
        """Initialize class."""
        super().__init__("")
        self._read_patterns_from_yaml()
        self.used_kinds = {}
        self.pdos = {"unit_x": "eV", "energy": [], "pdos": []}

    def parse_pdos(self, file_content, spin):
        """Parse pDOS from file."""
        self.lines = file_content.splitlines()
        self._line_idx = 0
        self.output_dict = {}
        self._current_patterns = self._pattern_dict["header"]
        self._current_blocks = self._pattern_dict["blocks"]
        self._current_block = None
        self._in_block = False
        self._block_idx = 0
        self._block_length = 0
        while self._line_idx < len(self.lines):
            self._check_general_patterns()
            self._check_blocks()
            self._line_idx += 1
        if self.output_dict["kind"] not in self.used_kinds:
            self.used_kinds[self.output_dict["kind"]] = len(self.used_kinds)
            self.pdos["pdos"].append({"kind": self.output_dict["kind"]})
        for orbital in self._orbital_labels:
            if orbital in self.output_dict:
                orbital_l = orbital
                if spin == "ALPHA":
                    orbital_l += "_alpha"
                elif spin == "BETA":
                    orbital_l += "_beta"
                self.pdos["pdos"][self.used_kinds[self.output_dict["kind"]]][orbital_l] = (
                    self.output_dict[orbital]
                )
        self.pdos["occupation"] = self.output_dict["occupation"]
        self.pdos["e_fermi"] = self.output_dict["e_fermi"] * energy.Hartree
        self.pdos["energy"] = [ev0 * energy.Hartree for ev0 in self.output_dict["eigenvalue"]]

    def _read_patterns_from_yaml(self):
        file_path = cwd + "/parameter_files/parser_pdos.yaml"
        self._pattern_dict = dict(load_yaml_file(file_path))


class RestartStructureParser(Parser):
    """Parser for the cp2k-restart file to obtain the latest structure."""

    def __init__(self, fstring):
        """Initialize class."""
        super().__init__(fstring)
        self.structures = []

    def retrieve_output_structure(self):
        """Retrieve output structure."""
        self._line_idx = 0
        self._read_patterns_from_yaml()

        self._current_blocks = self._pattern_dict["blocks"]
        self._current_block = None
        self._in_block = False
        self._block_idx = 0
        self._block_length = 0

        while self._line_idx < len(self.lines):
            if "&SUBSYS" in self.lines[self._line_idx]:
                self._line_idx += 1
                self._check_subsys()
            else:
                self._line_idx += 1
        return self.structures

    def _check_subsys(self):
        structure = {}
        kinds = {}
        while self._line_idx < len(self.lines):
            self._check_blocks()
            if "&KIND" in self.lines[self._line_idx]:
                kind, element = self._check_kind_blocks()
                kinds[kind] = element
            if "&SUBSYS" in self.lines[self._line_idx]:
                break
            self._line_idx += 1
        structure["cell"] = [self.output_dict["a"], self.output_dict["b"], self.output_dict["c"]]
        structure["pbc"] = [True, True, True]  # In case keyword is not set: Default in cp2k is XYZ
        if "pbc" in self.output_dict:
            structure["pbc"] = [(dir_st in self.output_dict["pbc"]) for dir_st in ["X", "Y", "Z"]]
        structure["symbols"] = []
        structure["kinds"] = []
        structure["positions"] = []
        if isinstance(self.output_dict["kind"], list):
            for kind, pos_x, pos_y, pos_z in zip(
                self.output_dict["kind"],
                self.output_dict["x"],
                self.output_dict["y"],
                self.output_dict["z"],
            ):
                # Bugfix for incomplete numbers in cp2k restart file:
                pos_x = 0.0 if pos_x is None else pos_x
                pos_y = 0.0 if pos_y is None else pos_y
                pos_z = 0.0 if pos_z is None else pos_z

                structure["symbols"].append(kinds[kind])
                structure["kinds"].append(kind)
                structure["positions"].append([pos_x, pos_y, pos_z])
        else:
            # Bugfix for incomplete numbers in cp2k restart file:
            pos_x = 0.0 if self.output_dict["x"] is None else self.output_dict["x"]
            pos_y = 0.0 if self.output_dict["y"] is None else self.output_dict["y"]
            pos_z = 0.0 if self.output_dict["z"] is None else self.output_dict["z"]

            structure["symbols"].append(kinds[self.output_dict["kind"]])
            structure["kinds"].append(self.output_dict["kind"])
            structure["positions"].append(
                [self.output_dict["x"], self.output_dict["y"], self.output_dict["z"]]
            )
        self.structures.append(structure)
        self.output_dict = {}

    def _check_kind_blocks(self):
        line_splitted = self.lines[self._line_idx].split()
        kind = line_splitted[-1]
        element = line_splitted[-1]

        nr_section_start = 1
        nr_section_end = 0
        while nr_section_start > nr_section_end:
            self._line_idx += 1
            line_splitted = self.lines[self._line_idx].split()
            if line_splitted[0] == "&END":
                nr_section_end += 1
            elif line_splitted[0].startswith("&"):
                nr_section_start += 1
            if line_splitted[0] == "ELEMENT":
                element = line_splitted[-1]
        return kind, element

    def _read_patterns_from_yaml(self):
        file_path = cwd + "/parameter_files/parser_structure.yaml"
        self._pattern_dict = dict(load_yaml_file(file_path))
