"""
Parent class for pattern processing and parse function for regex-bases parsers.
"""

# Standard library imports
import re
import abc
from typing import List

# Internal library imports
from aim2dat.io.utils import custom_open
from aim2dat.utils.dict_tools import dict_set_parameter


FLOAT = r"(([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)(\(?[0-9]*?\)?)?)"


class _BasePattern(abc.ABC):
    _pattern = None
    _flags = re.VERBOSE | re.MULTILINE
    _dict_tree = []

    def match_pattern(self, str_block: str, output: dict):
        pattern = re.compile(self._pattern, self._flags)
        matches = [match for match in pattern.finditer(str_block)]
        if len(matches) > 0:
            self.process_data(output, matches)

    def process_data(self, output: dict, matches: List[re.Match]):
        for key, val in matches[-1].groupdict().items():
            if val is not None:
                dict_set_parameter(output, self._dict_tree + [key], val)


def parse_function(file_name: str, patterns: List[_BasePattern]) -> dict:
    """
    Parse output file.

    Parameters
    ----------
    file_name : str
        Path to the output file.
    patterns : list
        List of child classes of ``_BasePattern``

    Returns
    -------
    dict
        Output dictionary.
    """
    output = {}
    with custom_open(file_name, mode="r") as fobj:
        content = fobj.read()
        for pattern in patterns:
            pattern().match_pattern(content, output)

    return output
