"""
Functions to read the restart file of CP2K.
"""

# Internal library imports
from aim2dat.io.cp2k.legacy_parser import RestartStructureParser
from aim2dat.io.utils import read_multiple, custom_open


@read_multiple(r".*-1\.restart$")
def read_optimized_structure(folder_path):
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
        with custom_open(file_p, "r") as restart_file:
            restart_content = restart_file.read()
        str_parser = RestartStructureParser(restart_content)
        new_structures = str_parser.retrieve_output_structure()
        if len(new_structures) == 1:
            structures[proj] = new_structures[0]
        elif len(new_structures) > 1:
            structures[proj] = new_structures
    return list(structures.values())[0] if len(structures.values()) == 1 else structures
