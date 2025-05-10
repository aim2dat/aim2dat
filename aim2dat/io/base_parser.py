"""
Parent class for pattern processing and parse function for regex-bases parsers.
"""

# Standard library imports
import re
import abc
from typing import List, Union

# Internal library imports
from aim2dat.io.utils import custom_open
from aim2dat.utils.dict_tools import dict_set_parameter, dict_merge


FLOAT = r"(([+-]?\d+)?\.?\d+([eE][-+]?\d+)?)"

TYPE_PATTERNS = [
    (re.compile(r"^([+-]?[0-9]+)$"), int),
    (re.compile(r"^" + FLOAT + r"(\(\d*\))?$"), float),
]


def transform_str_value(value: str) -> Union[str, int, float, bool]:
    """
    Get python type from str input and transform it.

    Parameters
    ----------
    value : str
        Input value.

    Returns
    -------
    str, int, float bool
        Detected and transformed value.
    """
    if value is None:
        return value

    value = value.strip()
    # print(value)
    for sl in ["'", '"']:
        value = value.strip(sl)
    if any(bv == value.lower() for bv in ["yes", "true"]):
        return True
    elif any(bv == value.lower() for bv in ["no", "false"]):
        return False
    for pattern, t in TYPE_PATTERNS:
        match = pattern.match(value)
        if match and len(match.group(1)) > 0:
            return t(match.group(1))
    return value


class _BasePattern(abc.ABC):
    pattern = None
    flags = re.VERBOSE | re.MULTILINE
    dict_tree = []

    def match_pattern(self, str_block: str, output: dict):
        pattern = re.compile(self.pattern, self.flags)
        matches = [match for match in pattern.finditer(str_block)]
        if len(matches) > 0:
            self.process_data(output, matches)

    def process_data(self, output: dict, matches: List[re.Match]):
        for key, val in matches[-1].groupdict().items():
            if val is not None:
                dict_set_parameter(output, self.dict_tree + [key], val)


def parse_pattern_function(file_path: str, patterns: List[_BasePattern]) -> dict:
    """
    Parse output file using regex patterns.

    Parameters
    ----------
    file_path : str
        Path to the output file.
    patterns : list
        List of child classes of ``_BasePattern``

    Returns
    -------
    dict
        Output dictionary.
    """
    output = {}
    with custom_open(file_path, mode="r") as fobj:
        content = fobj.read()
        for pattern in patterns:
            pattern().match_pattern(content, output)

    return output


class _BaseDataBlock(abc.ABC):
    start_str = None
    end_str = None
    use_once = False
    current_data_type = dict

    def __init__(self):
        self.all_data = []
        self.line_indices = []
        self.current_data = self.current_data_type()
        self.active = False
        self.sealed = False

    def add_line(self, idx, line):
        if self.sealed:
            return None

        if self.active:
            self._parse_line(line)
            if self._check_line(self.end_str, line):
                self._end_block()
        elif self._check_line(self.start_str, line):
            self.active = True
            self.line_indices.append(idx)
            self._parse_line(line)

    def get_output(self):
        if self.active:
            self._end_block()
        return self._process_output()

    def _end_block(self):
        self.active = False
        self.all_data.append(self.current_data)
        self.current_data = self.current_data_type()
        if self.use_once:
            self.sealed = True

    @staticmethod
    def _check_line(pattern, line):
        if pattern is None:
            return line.strip() == ""
        if isinstance(pattern, str):
            pattern = [pattern]
        for p in pattern:
            if p in line:
                return True
        return False

    @abc.abstractmethod
    def _parse_line(self, line):
        pass

    def _process_output(self):
        if len(self.all_data) > 0:
            return self.all_data[-1]


def parse_block_function(file_path: str, block_classes: List[_BaseDataBlock]) -> dict:
    """
    Parse output file using data blocks.

    Parameters
    ----------
    file_path : str
        Path to the output file.
    patterns : list
        List of child classes of ``_BaseDataBlock``

    Returns
    -------
    dict
        Output dictionary.
    """
    blocks = [block() for block in block_classes]
    output = {}
    with custom_open(file_path, mode="r") as fobj:
        for idx, line in enumerate(fobj):
            for block in blocks:
                block.add_line(idx, line)
    for block in blocks:
        outp = block.get_output()
        if outp is not None:
            dict_merge(output, outp)
    return output, idx
