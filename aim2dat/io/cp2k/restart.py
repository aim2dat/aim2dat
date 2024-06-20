"""
Functions to read the restart file of CP2K.
"""

# Standard library imports
from typing import List, Union
import re

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.io.base_parser import parse_function, _BasePattern
from aim2dat.io.utils import read_multiple


class _CellPattern(_BasePattern):
    _pattern = r"^\s*&CELL\n(.*\n)*?\s*&END\sCELL"
    _dir_mapping = {"A": 0, "B": 1, "C": 2}

    def process_data(self, output: dict, matches: List[re.Match]):
        output["pbc"] = [True, True, True]
        cell = [None, None, None]
        cell_factors = [1.0, 1.0, 1.0]
        m = matches[-1]
        for line in m.string[m.start() : m.end()].splitlines()[1:-1]:
            line_sp = line.split()
            label = line_sp[0].upper()
            if label in self._dir_mapping:
                cell[self._dir_mapping[label]] = np.array([float(val) for val in line_sp[1:]])
            elif label == "PERIODIC":
                output["pbc"] = [(dir_st in line_sp[1]) for dir_st in ["X", "Y", "Z"]]
            elif label == "MULTIPLE_UNIT_CELL":
                cell_factors = [float(val) for val in line_sp[1:]]
        if all(c0 is not None for c0 in cell):
            output["cell"] = [(c_f * v0).tolist() for c_f, v0 in zip(cell_factors, cell)]


class _CoordPattern(_BasePattern):
    _pattern = r"^\s*&COORD\n(.*\n)*?\s*&END\sCOORD"

    def process_data(self, output: dict, matches: List[re.Match]):
        output["kinds"] = []
        output["positions"] = []
        m = matches[-1]
        for line in m.string[m.start() : m.end()].splitlines()[1:-1]:
            line_sp = line.split()
            output["kinds"].append(line_sp[0])
            pos = []
            for val in line_sp[1:]:
                try:
                    pos.append(float(val))
                except ValueError:
                    pos.append(0.0)
            output["positions"].append(pos)


class _KindPattern(_BasePattern):
    _pattern = r"\s*&KIND\s(?P<kind>\S+)\n(.*\n)*?\s*&END\sKIND"

    def process_data(self, output: dict, matches: List[re.Match]):
        output["kind_info"] = {}
        for m in matches:
            kind = m.groupdict()["kind"]
            element = kind
            for line in m.string[m.start() : m.end()].splitlines()[1:-1]:
                line_sp = line.split()
                if line_sp[0].upper() == "ELEMENT":
                    element = line_sp[1]
                    break
            output["kind_info"][kind] = element


@read_multiple(r".*-1\.restart$")
def read_optimized_structure(folder_path: str) -> Union[dict, List[dict]]:
    """
    Read optimized structures from 'restart'-files.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the CP2K ouput-files.

    Returns
    -------
    dict or list
        Dictionary containing the structural information. In case of a farming job a list of
        dictionaries is returned. In case several calculations have been run in the same folder a
        nested dictionary is returned.
    """
    structures = {}
    for file_p, file_n in zip(folder_path["file"], folder_path["file_name"]):
        proj = file_n.rsplit("-", 1)[0]
        output = parse_function(file_p, [_CellPattern, _CoordPattern, _KindPattern])
        kind_info = output.pop("kind_info")
        output["symbols"] = [kind_info[kind] for kind in output["kinds"]]
        if len(output) == 1:
            structures[proj] = output[0]
        elif len(output) > 1:
            structures[proj] = output
    return list(structures.values())[0] if len(structures.values()) == 1 else structures
