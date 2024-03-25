"""Functions to read the standard output file of CP2K."""

# Internal library imports
from aim2dat.io.cp2k.legacy_parser import MainOutputParser
from aim2dat.io.utils import custom_open


def read_stdout(file_name, parser_type="standard"):
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
    with custom_open(file_name, mode="r") as fobj:
        content = fobj.read()
    parser = MainOutputParser(content)
    return parser.retrieve_result_dict(parser_type)
