"""
Parsers for the cubecruncher software package.
"""

# Standard library imports
import pathlib

# Third party library imports
from aiida.parsers import Parser
from aiida.plugins import DataFactory


GCubeData = DataFactory("aim2dat.gaussian_cube")


class CubecruncherParser(Parser):
    """
    Output parser for the cubecruncher software package.
    """

    def parse(self, **kwargs):
        """Receives in input a dictionary of retrieved nodes. Does all the logic here."""
        retrieved_temporary_folder = kwargs.get("retrieved_temporary_folder", None)

        try:
            temporary_folder = pathlib.Path(retrieved_temporary_folder)

            for subpath in temporary_folder.iterdir():
                if subpath.is_file():
                    g_cube_data = GCubeData.set_from_file(subpath)
                self.out("cdd_cube", g_cube_data)
        except OSError:
            return self.exit_codes.ERROR_READING_OUTPUT_FILE
