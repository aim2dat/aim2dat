"""Functions to read the standard output file of CP2K."""

# Standard library imports
from typing import List
import re

# Internal library imports
from aim2dat.io.base_parser import parse_function, _BasePattern, FLOAT


class _CP2KPattern(_BasePattern):
    _pattern = r"^\s*CP2K\|\sversion\sstring:\s+CP2K\sversion\s(?P<cp2k_version>\S+)$"


class _GlobalPattern(_BasePattern):
    _pattern = r"^\s*GLOBAL\|\sRun\stype\s+(?P<run_type>\S+)$"


class _BrillouinPattern(_BasePattern):
    _pattern = rf"""
        ^\s*BRILLOUIN\|\s+(
            (K-point\sscheme\s*(?P<scheme>\S+))
            |(K-Point\sgrid\s*(?P<kx>\d+)\s+(?P<ky>\d+)\s+(?P<kz>\d+))
            |(\s*(?P<kpt_nr>\d+)\s+(?P<kpt_w>{FLOAT})\s+(?P<kpt_x>{FLOAT})\s+(?P<kpt_y>{FLOAT})\s+(?P<kpt_z>{FLOAT}))
        )$
    """

    def process_data(self, output: dict, matches: List[re.Match]):
        for m in matches:
            data = m.groupdict()
            if data["scheme"] is not None:
                output["kpoint_scheme"] = data["scheme"]
            elif data["kx"] is not None:
                output["kpoint_grid"] = [int(data["kx"]), int(data["ky"]), int(data["kz"])]
            else:
                kpts = output.setdefault("kpoints", [])
                kpts.append(
                    (
                        float(data["kpt_w"]),
                        float(data["kpt_x"]),
                        float(data["kpt_y"]),
                        float(data["kpt_z"]),
                    )
                )


class _DFTPattern(_BasePattern):
    _pattern = r"""
        ^\s*DFT\|[\S\s]+\s+(?P<dft_type>\S+)\n
        \s*DFT\|\sMultiplicity\s+(?P<multiplicity>\d+)\n
        \s*DFT\|[\s\S]+\n
        \s*DFT\|\sCharge\s+(?P<charge>[+-]*\d+)$
    """

    def process_data(self, output: dict, matches: List[re.Match]):
        data = matches[-1].groupdict()
        output["dft_type"] = data["dft_type"]
        output["multiplicity"] = int(data["multiplicity"])
        output["charge"] = float(data["charge"])


class _FunctionalPattern(_BasePattern):
    _pattern = r"""
        ^(^\s*FUNCTIONAL\|\s(?P<functional>.+):$)|
        (^\s*vdW\sPOTENTIAL\|\s*(?P<vdw_type>[\w\s]+)\n
         \s*vdW\sPOTENTIAL\|\s*(?P<vdw_method>[\w-]+).*$)
    """

    def process_data(self, output: dict, matches: List[re.Match]):
        output["xc"] = {}
        for m in matches:
            data = m.groupdict()
            fct = data.get("functional", None)
            if fct is None:
                for k0 in ["vdw_type", "vdw_method"]:
                    output["xc"][k0] = data[k0]
                break

            output["xc"]["functional"] = (
                [output["xc"]["functional"], fct] if "functional" in output["xc"] else fct
            )


class _MDPattern(_BasePattern):
    _pattern = r"^\s*MD_PAR\|\sEnsemble\stype\s+(?P<md_ensemble>\w+)$"


class _NumbersPattern(_BasePattern):
    _pattern = r"""
        ^\s*TOTAL\sNUMBERS\sAND\sMAXIMUM\sNUMBERS\n\n
        [\S\s]*\n
        \s*-\sAtoms:\s+(?P<natoms>\d+)$
    """

    def process_data(self, output: dict, matches: List[re.Match]):
        output["natoms"] = int(matches[-1].groupdict()["natoms"])


class _KindInfoPattern(_BasePattern):
    _pattern = rf"""
        ^\s*Atom\s+Kind\s+Element\s+X\s+Y\s+Z\s+Z\(eff\)\s+Mass(\n)+
        (\s*(\d+\s+){{2}}\S+\s+\d+\s+({FLOAT}\s*){{5}}\n)+$
    """

    def process_data(self, output: dict, matches: List[re.Match]):
        m = matches[-1]
        output["kind_info"] = []
        for line in m.string[m.start() : m.end()].splitlines():
            line_sp = line.split()
            if len(line_sp) == 9:
                output["kind_info"].append(
                    {
                        "kind": int(line_sp[1]),
                        "element": line_sp[2],
                        "atomic_nr": int(line_sp[3]),
                        "core_electrons": int(int(line_sp[3]) - float(line_sp[7])),
                        "valence_electrons": int(float(line_sp[7])),
                        "mass": float(line_sp[8]),
                    }
                )


class _SPGRPattern(_BasePattern):
    _pattern = r"""
        ^\s*SPGR\|\sSPACE\sGROUP\sNUMBER:\s+(?P<sg_number>\d+)\n
        \s*SPGR\|\sINTERNATIONAL\sSYMBOL:\s+(?P<int_symbol>.+)\n
        \s*SPGR\|\sPOINT\sGROUP\sSYMBOL:\s+(?P<point_group_symbol>.+)\n
        \s*SPGR\|\sSCHOENFLIES\sSYMBOL:\s+(?P<schoenflies_symbol>.+)$
    """

    def process_data(self, output: dict, matches: List[re.Match]):
        data = matches[-1].groupdict()
        data["sg_number"] = int(data["sg_number"])
        output["spgr_info"] = data


class _SCFParametersPattern(_BasePattern):
    _pattern = rf"^\s*added\sMOs\s*(?P<added_mos1>{FLOAT})\s*(?P<added_mos2>{FLOAT})$"

    def process_data(self, output: dict, matches: List[re.Match]):
        data = matches[-1].groupdict()
        if data["added_mos2"] != "0":
            output["nr_unocc_orbitals"] = [int(data["added_mos1"]), int(data["added_mos2"])]
        else:
            output["nr_unocc_orbitals"] = int(data["added_mos1"])


class _SCFPattern(_BasePattern):
    _pattern = rf"""
        (\s*Spin\s1\n\n)?
        \s*Number\sof\selectrons:\s+\d+\n
        \s*Number\sof\soccupied\sorbitals:\s+\d+\n
        (.*\n)*?
        (\s*ENERGY\|\sTotal\sFORCE_EVAL\s
         \(\sQS\s\)\senergy\s\[(?P<energy_units>\S+)\]:\s+(?P<energy>{FLOAT}))
    """

    def process_data(self, output: dict, matches: List[re.Match]):
        output["scf_steps"] = []
        for m in matches:
            data = m.groupdict()
            m_step_dict = {
                "energy": float(data["energy"]),
                "span": (m.start(), m.end()),
            }

            lines = m.string[m.start() : m.end()].splitlines()
            line_idx = 0
            while line_idx < len(lines):
                line = lines[line_idx]
                line_idx = self._extract_n_electrons(line_idx, lines, output)
                self._extract_scf_details(line, m_step_dict)
                self._extract_total_energy(line, output)
                line_idx = self._extract_partial_charges(line_idx, lines, m_step_dict)
                line_idx += 1
            output["scf_steps"].append(m_step_dict)

        output["scf_converged"] = m_step_dict["scf_converged"]
        output["energy_units"] = data["energy_units"]
        output["energy"] = float(data["energy"])

    @staticmethod
    def _extract_n_electrons(start_idx, lines, outp_dict):
        line_indices = [start_idx]
        if "Spin 1" in lines[start_idx]:
            line_indices = [start_idx + 2, start_idx + 8]
        end_idx = line_indices[-1]

        nrs = [[], []]
        for line_idx in line_indices:
            if "Number of electrons:" in lines[line_idx]:
                nrs[0].append(int(float(lines[line_idx].split()[-1])))
                nrs[1].append(int(float(lines[line_idx + 1].split()[-1])))
                end_idx += 1

        if len(nrs[0]) > 0:
            for label, nr in zip(["nelectrons", "nr_occ_orbitals"], nrs):
                outp_dict[label] = nr[0] if len(nr) == 1 else nr
        return end_idx

    @staticmethod
    def _extract_scf_details(line, outp_dict):
        if "SCF run converged in" in line:
            outp_dict["nr_scf_steps"] = int(line.split()[-3])
            outp_dict["scf_converged"] = True
        elif "Leaving inner SCF loop after" in line:
            outp_dict["nr_scf_steps"] = int(line.split()[-2])
            outp_dict["scf_converged"] = False
        elif "outer SCF loop converged in" in line:
            outp_dict["nr_scf_steps"] = int(line.split()[-2])
            outp_dict["scf_converged"] = True
        elif "outer SCF loop FAILED to converge after" in line:
            outp_dict["nr_scf_steps"] = int(line.split()[-2])
            outp_dict["scf_converged"] = False

    @staticmethod
    def _extract_total_energy(line, outp_dict):
        if "Total energy:" in line:
            outp_dict["energy_scf"] = float(line.split()[-1])

    @staticmethod
    def _extract_partial_charges(start_idx, lines, outp_dict):
        if "Mulliken Population Analysis" in lines[start_idx]:
            charge_type = "mulliken"
        elif "Hirshfeld Charges" in lines[start_idx]:
            charge_type = "hirshfeld"
        else:
            return start_idx

        start_idx += 3
        outp_dict[charge_type] = []
        for line_idx, line in enumerate(lines[start_idx:]):
            line_sp = line.split()
            if len(line_sp) > 4 and "#" != line_sp[0]:
                if charge_type == "mulliken" and len(line_sp) == 7:
                    pop = [float(line_sp[-4]), float(line_sp[-3])]
                    chrg = float(line_sp[-2])
                elif charge_type == "hirshfeld" and len(line_sp) == 8:
                    pop = [float(line_sp[-4]), float(line_sp[-3])]
                    chrg = float(line_sp[-1])
                else:
                    pop = float(line_sp[-2])
                    chrg = float(line_sp[-1])

                outp_dict[charge_type].append(
                    {
                        "kind": int(line_sp[2]),
                        "element": line_sp[1],
                        "population": pop,
                        "charge": chrg,
                    }
                )
            if "!---------------" in line:
                break

        return start_idx + line_idx


class _OptStepPattern(_BasePattern):
    _pattern = rf"""
        ^(
            (\s*\*{{3}}\s+STARTING\s+\S+\s+OPTIMIZATION\s+\*{{3}}\n)|
            (\s*OPTIMIZATION\sSTEP:\s+\d+\n)
        )
        (
            (.*\n)*?
            \s*-+\s\sInformations\sat\sstep\s=\s+(?P<nr_steps>\d+)\s*-+\n
            (.*\n)*?
            \s*Total\sEnergy\s+=\s+{FLOAT}\n
            (\s*Internal\sPressure\s\[\S+\]\s+=\s+(?P<pressure>{FLOAT})\n)?
            (.*\n)*?
            \s*Used\stime\s+=\s+{FLOAT}\n
            (
                \n\s*Convergence\scheck\s:\n
                \s*Max.\sstep\ssize\s+=\s+(?P<max_step>{FLOAT})\n
                (.*\n)*?
                \s*RMS\sstep\ssize\s+=\s+(?P<rms_step>{FLOAT})\n
                (.*\n)*?
                \s*Max.\sgradient\s+=\s+(?P<max_grad>{FLOAT})\n
                (.*\n)*?
                \s*RMS\sgradient\s+=\s+(?P<rms_grad>{FLOAT})\n
                (.*\n)*?
            )?
            \s*-{{51}}\n
        )?
        (
            ((.*\n)*?\s*-{{26}})|
            ((.*\n)?\n\s*\*{{79}}\s*\*{{3}}\s+GEOMETRY\sOPTIMIZATION\s(?P<opt_success>\S+)\s+\*{{3}})|
            ((.*\n)?\n\s*\*{{3}}\s+MAXIMUM\sNUMBER\sOF\sOPTIMIZATION\sSTEPS\s(?P<max_steps>\S+)\s+\*{{3}})
        )?
        $
    """

    def process_data(self, output: dict, matches: List[re.Match]):
        output["motion_step_info"] = []
        for m in matches:
            m_step_dict = {"span": (m.start(), m.end())}
            data = m.groupdict()
            opt_success = data.pop("opt_success")
            max_steps_reached = data.pop("max_steps")
            if data["nr_steps"] is not None:
                output["nr_steps"] = int(data.pop("nr_steps"))
            for key, val in data.items():
                if val is not None:
                    m_step_dict[key] = float(val)
            output["motion_step_info"].append(m_step_dict)
            if opt_success == "COMPLETED":
                output["motion_step_info"].append({"span": (m.end(), None)})
            if max_steps_reached == "REACHED":
                output["geo_not_converged"] = True


class _MDStepPattern(_BasePattern):
    _pattern = rf"""
        ^(
            \s*MD\|\sStep\snumber\s*(?P<step_nr>\d+)\n
            \s*MD\|\sTime\s\[fs\]\s*(?P<time_fs>{FLOAT})\n
            (\s*MD\|.*\n)*
            \s*MD\|\sEnergy\sdrift\sper\satom\s\[\S+\]\s+(?P<energy_drift_p_atom>{FLOAT})\s+{FLOAT}\n
        )?
        \s*MD(_INI)?\|\sPotential\senergy\s\[\S+\]\s+(?P<potential_energy>{FLOAT})(\s+{FLOAT})?\n
        \s*MD(_INI)?\|\sKinetic\senergy\s\[\S+\]\s+(?P<kinetic_energy>{FLOAT})(\s+{FLOAT})?\n
        \s*MD(_INI)?\|\sTemperature\s\[\S+\]\s+(?P<temperature>{FLOAT})(\s+{FLOAT})?$
    """

    def process_data(self, output: dict, matches: List[re.Match]):
        output["motion_step_info"] = []
        span_start = None
        for m in matches:
            data = m.groupdict()
            step_nr = data.pop("step_nr")
            time_fs = data.pop("time_fs")
            m_step_dict = {
                "step_nr": 0 if step_nr is None else int(step_nr),
                "time_fs": 0.0 if time_fs is None else float(time_fs),
                "span": [span_start, None],
            }
            for key, val in data.items():
                if val is not None:
                    m_step_dict[key] = float(val)
            output["motion_step_info"].append(m_step_dict)
            span_start = m.start()


class _BandsPattern(_BasePattern):
    _pattern = rf"""
        ^\s*KPOINTS\|\sSpecial\spoint\s+1\s+(?P<label1>(\S+)|(not\sspecifi))
            \s+{FLOAT}\s+{FLOAT}\s+{FLOAT}\n
        \s*KPOINTS\|\sSpecial\spoint\s+2\s+(?P<label2>(\S+)|(not\sspecifi))
            \s+{FLOAT}\s+{FLOAT}\s+{FLOAT}\n
        (.*\n)*?
        \s*KPOINTS\|\sTime\sfor\sk-point\sline\s+{FLOAT}$
    """

    def process_data(self, output: dict, matches: List[re.Match]):
        output["kpoint_data"] = {
            "labels": [],
            "kpoints": [],
            "bands": [[], []],
            "occupations": [[], []],
        }
        for m in matches:
            labels = [None if val == "not specifi" else val for val in m.groupdict().values()]
            spin = 0
            kpoint_counter = 0
            for line in m.string[m.start() : m.end()].splitlines()[2:]:
                line_sp = line.split()
                if "KPOINTS" in line:
                    continue
                elif line.startswith("#"):
                    if line_sp[2] == "Energy":
                        output["kpoint_data"]["bands_unit"] = line_sp[3][1:-1]
                    elif line_sp[4] == "1:":
                        output["kpoint_data"]["kpoints"].append(
                            [float(line_sp[5]), float(line_sp[6]), float(line_sp[7])]
                        )
                        kpoint_counter += 1
                        for idx in range(2):
                            output["kpoint_data"]["bands"][idx].append([])
                            output["kpoint_data"]["occupations"][idx].append([])
                        spin = 0
                    elif line_sp[4] == "2:":
                        spin = 1
                else:
                    output["kpoint_data"]["bands"][spin][-1].append(float(line_sp[1]))
                    output["kpoint_data"]["occupations"][spin][-1].append(float(line_sp[2]))
            output["kpoint_data"]["labels"].append(
                [len(output["kpoint_data"]["kpoints"]) - kpoint_counter, labels[0]]
            )
            output["kpoint_data"]["labels"].append(
                [len(output["kpoint_data"]["kpoints"]) - 1, labels[1]]
            )
        if spin == 0:
            output["kpoint_data"]["bands"] = output["kpoint_data"]["bands"][0]
            output["kpoint_data"]["occupations"] = output["kpoint_data"]["occupations"][0]


class _EigenvaluesPattern(_BasePattern):
    _pattern = rf"""
        ^\s*(MO\|)?(\s(?P<spin>\S+))?(\sMO)?\sEIGENVALUES\sAND(\sMO)?\sOCCUPATION\sNUMBERS.*\n
        (.*\n)+?
        \s*((Fermi\senergy:\s+)|(MO\|\sE\(Fermi\):\s+))(?P<fermi_energy>{FLOAT})
    """

    def process_data(self, output: dict, matches: List[re.Match]):
        output["eigenvalues_info"] = {
            "eigenvalues": [],
        }
        vbms = [[], []]
        cbms = [[], []]
        gaps = [[], []]
        ev_counter = 0
        ev0 = None
        for m in matches:
            data = m.groupdict()
            if data["spin"] == "BETA":
                energies, occs, vbm, cbm = self._extract_eigenvalues(m)
                ev0["energies"] = [ev0["energies"], energies]
                ev0["occupations"] = [ev0["occupations"], occs]
                if cbm is not None:
                    ev0["gap"] = [ev0["gap"], cbm - vbm]
                    cbms[1].append(cbm)
                    gaps[1].append(ev0["gap"][1])
                vbms[1].append(vbm)
            else:
                ev0 = {
                    "energies": [],
                    "occupations": [],
                }
                if "kpoints" in output:
                    ev0["weight"] = output["kpoints"][ev_counter][0]
                    ev0["kpoint"] = list(output["kpoints"][ev_counter][1:])
                ev0["energies"], ev0["occupations"], vbm, cbm = self._extract_eigenvalues(m)
                if cbm is not None:
                    ev0["gap"] = cbm - vbm
                    cbms[0].append(cbm)
                    gaps[0].append(ev0["gap"])
                vbms[0].append(vbm)
                output["eigenvalues_info"]["eigenvalues"].append(ev0)
                ev_counter += 1
            output["fermi_energy"] = float(data["fermi_energy"])

        if len(cbms[0]) > 0:
            if len(vbms[1]) > 0:
                output["eigenvalues_info"]["gap"] = min(
                    [max(min(cbms[idx]) - max(vbms[idx]), 0.0) for idx in range(2)]
                )
                output["eigenvalues_info"]["direct_gap"] = min(
                    [max(min(gaps[idx]), 0.0) for idx in range(2)]
                )
            else:
                output["eigenvalues_info"]["gap"] = max(min(cbms[0]) - max(vbms[0]), 0.0)
                output["eigenvalues_info"]["direct_gap"] = max(min(gaps[0]), 0.0)

    @staticmethod
    def _extract_eigenvalues(m):
        homo_idx = 0
        start_ev = False
        energies = []
        occs = []
        for line in m.string[m.start() : m.end()].splitlines():
            if "# MO index" in line or "Index" in line:
                start_ev = True
            elif "Sum" in line:
                start_ev = False
            elif start_ev:
                line_sp = line.split()
                occ = float(line_sp[-1])
                if occ >= 0.5:
                    homo_idx = len(occs)
                if line_sp[0] == "MO|":
                    energies.append(float(line_sp[2]))
                else:
                    energies.append(float(line_sp[1]))
                occs.append(occ)
        vbm = energies[homo_idx]
        cbm = energies[homo_idx + 1] if len(energies) > homo_idx + 1 else None
        return energies, occs, vbm, cbm


class _WarningsPattern(_BasePattern):
    _pattern = r"""
        ^\s*\*{3}\sWARNING\sin\s(?P<file_name>\S+):(?P<line_number>\d+)\s::\s(?P<message>.*)\*{3}\n
        ((\s\*{3}.*\*{3}\n)*)?$
    """

    def process_data(self, output: dict, matches: List[re.Match]):
        output["nwarnings"] = len(matches)
        output["warnings"] = []
        if output["nwarnings"] > 0:
            for m in matches:
                warn_dict = m.groupdict()
                message = warn_dict["message"].rstrip()
                for line in m.string[m.start() : m.end()].splitlines()[2:]:
                    message += line[4:-4].rstrip()
                output["warnings"].append(
                    (warn_dict["file_name"], int(warn_dict["line_number"]), message)
                )


class _ErrorPattern(_BasePattern):
    _pattern = r"""
        ^\s*\*\s\[ABORT\].*\n
        \s*\*\s{2}\\___\/\s+(?P<message>.+)\s*\*\n
        (\s*\*.*\*\n)*?
        \s*\*\s\/\s\\\s+(?P<file_name>\S+):(?P<line_number>\d+)\s\*$
    """

    def process_data(self, output: dict, matches: List[re.Match]):
        output["aborted"] = True
        output["errors"] = []

        for m in matches:
            data = m.groupdict()
            output["errors"].append(
                (data["file_name"], int(data["line_number"]), data["message"].rstrip())
            )


class _RuntimePattern(_BasePattern):
    _pattern = rf"""
        \s*-\s*T\sI\sM\s*I\s*N\s*G\s*-\n
        (.*\n)*?
        \s*CP2K\s*\d+\s*{FLOAT}\s*{FLOAT}\s*{FLOAT}\s*{FLOAT}\s*(?P<rt_cp2k>{FLOAT})$
    """

    def process_data(self, output: dict, matches: List[re.Match]):
        output["runtime"] = float(matches[-1].groupdict()["rt_cp2k"])


class _WalltimePattern(_BasePattern):
    _pattern = r"^\s*\*{3}\s.*exceeded\srequested\sexecution\stime:.*$"

    def process_data(self, output: dict, matches: List[re.Match]):
        output["exceeded_walltime"] = True


_BLOCKS = {
    "standard": [
        _CP2KPattern,
        _GlobalPattern,
        _BrillouinPattern,
        _DFTPattern,
        _FunctionalPattern,
        _MDPattern,
        _SPGRPattern,
        _NumbersPattern,
        _SCFParametersPattern,
        _SCFPattern,
        _OptStepPattern,
        _BandsPattern,
        _EigenvaluesPattern,
        _WarningsPattern,
        _ErrorPattern,
        _RuntimePattern,
        _WalltimePattern,
    ],
    "trajectory": [
        _CP2KPattern,
        _GlobalPattern,
        _BrillouinPattern,
        _DFTPattern,
        _FunctionalPattern,
        _MDPattern,
        _SPGRPattern,
        _NumbersPattern,
        _KindInfoPattern,
        _SCFParametersPattern,
        _SCFPattern,
        _OptStepPattern,
        _MDStepPattern,
        _WarningsPattern,
        _ErrorPattern,
        _RuntimePattern,
        _WalltimePattern,
    ],
    "partial_charges": [
        _CP2KPattern,
        _GlobalPattern,
        _BrillouinPattern,
        _DFTPattern,
        _FunctionalPattern,
        _MDPattern,
        _SPGRPattern,
        _NumbersPattern,
        _KindInfoPattern,
        _SCFParametersPattern,
        _SCFPattern,
        _OptStepPattern,
        _BandsPattern,
        _EigenvaluesPattern,
        _WarningsPattern,
        _ErrorPattern,
        _RuntimePattern,
        _WalltimePattern,
    ],
}

_WARNINGS = [
    ("Using a non-square number of", "Using a non-square number of MPI ranks."),
    ("SCF run NOT converged", "One or more SCF run did not converge."),
    ("Specific L-BFGS convergence criteria", "LBFGS converged with specific criteria."),
    ("Add more MOs for proper smearing", "Add more MOs for proper smearing."),
]

_ERRORS = [
    ("exceeded requested execution time", "exceeded_walltime"),
    ("Use the LSD option for an odd number of electrons", "odd_nr_electrons"),
    ("Extra MOs (ADDED_MOS) are required for smearing", "need_added_mos"),
    ("exceeded requested execution time", "exceeded_walltime"),
    ("Cholesky decompose failed", "cholesky_decompose_failed"),
    ("Bad condition number R_COND", "bad_condition_number"),
]


def read_stdout(file_name: str, parser_type: str = "standard") -> dict:
    """
    Read standard output file of CP2K.

    Parameters
    ----------
    file_name : str
        Path to the output file.
    parser_type : str
        Defines the quantities that are being parsed. Supported options are ``'standard'``,
        ``'partial_charges'`` and ``'trajectory'``.

    Returns
    -------
    dict
        Dictionary containing the parsed values.
    """
    output = parse_function(file_name, _BLOCKS[parser_type])
    output["cp2k_version"] = float(output["cp2k_version"])
    if "exceeded_walltime" not in output:
        output["exceeded_walltime"] = False
    if "md_ensemble" in output:
        output["run_type"] += "-" + output.pop("md_ensemble")
    if "runtime" not in output:
        output.pop("nwarnings", None)
        output["energy_units"] = "a.u."
        output["interrupted"] = True
    output.pop("kpoints", None)

    warnings = []
    for warn0 in output.get("warnings", []):
        for warn1 in _WARNINGS:
            if warn1[0] in warn0[2]:
                warnings.append(warn1[1])
    output["warnings"] = warnings

    errors = output.pop("errors", [])
    for err0 in errors:
        if err0[2] == "SCF run NOT converged. To continue the calculation regardless,":
            output["scf_converged"] = False
        for err1 in _ERRORS:
            if err1[0] in err0[2]:
                output[err1[1]] = True

    scf_steps = output.pop("scf_steps", None)
    if parser_type == "partial_charges" and scf_steps is not None:
        for pc_type in ["mulliken", "hirshfeld"]:
            if pc_type in scf_steps[-1]:
                output[pc_type] = scf_steps[-1][pc_type]

    if parser_type == "trajectory" and scf_steps is not None:
        motion_steps = output.setdefault("motion_step_info", [])
        if len(motion_steps) > 0:
            scf_idx = 0
            for m_step_idx, m_step in enumerate(motion_steps[1:]):
                motion_steps[m_step_idx]["scf_steps"] = []
                while scf_idx < len(scf_steps):
                    scf_step = scf_steps[scf_idx]
                    if scf_step["span"][1] > m_step["span"][0]:
                        break
                    motion_steps[m_step_idx]["scf_steps"].append(scf_step)
                    scf_idx += 1
            if scf_idx < len(scf_steps):
                motion_steps[-1]["scf_steps"] = scf_steps[scf_idx:]
        else:
            motion_steps.append({"scf_steps": [scf_steps[-1]]})
        del scf_steps

        for m_step in motion_steps:
            m_step.pop("span", None)
            for scf_step in m_step.get("scf_steps", []):
                scf_step.pop("span", None)
            if len(m_step.get("scf_steps", [])) == 1:
                scf_steps = m_step.pop("scf_steps")
                m_step.update(scf_steps[0])
    else:
        output.pop("motion_step_info", None)
    return output
