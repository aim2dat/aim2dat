"""AiiDA data classes for gaussian cube files."""

# Standard library imports
import contextlib
import tempfile
import bz2

# Third party library imports
from aiida.orm import Data

# Internal library imports
from aim2dat.io.gaussian_cube import read_gaussian_cube_file, _get_cube_data
from aim2dat.units import UnitConverter, length


class GaussianCubeData(Data):
    """AiiDA data object to store gaussian cube files."""

    _comp_file_name = "gcube.bz2"

    def __init__(self, **kwargs):
        """Initialize object."""
        super().__init__(**kwargs)

    @property
    def title(self):
        """list : Return the title of the cube file."""
        return self.base.attributes.get("title", None)

    @property
    def comment(self):
        """list : Return the second line of the cube file."""
        return self.base.attributes.get("comment", None)

    @property
    def origin(self):
        """list : Return the origin of the data."""
        return self.base.attributes.get("origin", None)

    @property
    def cell(self):
        """list : Return the cell."""
        return self.base.attributes.get("cell", None)

    @property
    def shape(self):
        """list : Return the number of points in each direction."""
        return self.base.attributes.get("shape", None)

    @property
    def atomic_numbers(self):
        """list : Return the atomic numbers."""
        return self.base.attributes.get("atomic_numbers", None)

    @property
    def atomic_charges(self):
        """list : Return the atomic charges."""
        return self.base.attributes.get("atomic_charges", None)

    @property
    def atomic_positions(self):
        """list : Return the atomic positions (in bohr)."""
        return self.base.attributes.get("atomic_positions", None)

    @property
    def dset_ids(self):
        """list : Return the data set identifiers."""
        return self.base.attributes.get("dset_ids", None)

    @contextlib.contextmanager
    def open_cube(self, path=None):
        """
        Open cube file.

        Parameters
        ----------
        path : str or None
            Path to the cube file.

        Returns
        -------
        A file handle.
        """
        if path is None:
            path = self._comp_file_name

        with self.base.repository.open(path, mode="rb") as handle:
            with bz2.open(handle, "rt", compresslevel=9) as bz2_handle:
                yield bz2_handle

    def get_content(self):
        """
        Get content of the cube file.

        Returns
        -------
        str
            File conent as string.
        """
        with self.open_cube() as handle:
            return handle.read()

    def get_structure(self, unit="angstrom"):
        """
        Get underlying structure.

        Parameters
        ----------
        unit : str
            Length unit.
        """
        if unit not in length.available_units:
            raise ValueError(f"Unit '{unit}' is not supported.")
        cell = [
            [UnitConverter.convert_units(val, "bohr", unit) for val in cellv]
            for cellv in self.cell
        ]
        positions = [
            [
                UnitConverter.convert_units(val - org, "bohr", unit)
                for val, org in zip(at, self.origin)
            ]
            for at in self.atomic_positions
        ]
        return {
            "elements": self.atomic_numbers,
            "positions": positions,
            "cell": cell,
            "is_cartesian": True,
        }

    def get_cube_data(self):
        """
        Get cube data points.

        Returns
        -------
        np.array
            Data of the cube file.
        """
        with self.open_cube() as handle:
            handle.seek(self.base.attributes.get("data_start"))
            data = _get_cube_data(handle, self.shape, self.base.attributes.get("n_values"))
        return data

    @classmethod
    def set_from_file(cls, file_path):
        """
        Set information from existing cube file.

        Returns
        -------
        GaussianCubeData
            Data object containing the information from the file.
        """
        gdata = GaussianCubeData()
        strct_dict = read_gaussian_cube_file(file_path, unit="bohr", get_data=False)
        cube_dict = strct_dict["attributes"]["cube"]
        for key in ["title", "comment", "origin", "n_values", "shape", "dset_ids", "data_start"]:
            gdata.base.attributes.set(key, cube_dict[key])

        gdata.base.attributes.set("cell", strct_dict["cell"])
        gdata.base.attributes.set("atomic_numbers", strct_dict["elements"])
        gdata.base.attributes.set(
            "atomic_charges", strct_dict["site_attributes"]["atomic_charges"]
        )
        gdata.base.attributes.set(
            "atomic_positions",
            [[v + o for v, o in zip(val, cube_dict["origin"])] for val in strct_dict["positions"]],
        )
        with open(file_path, "r") as file_obj:
            file_obj.seek(0)
            with tempfile.NamedTemporaryFile() as handle:
                with bz2.open(handle, "wt", compresslevel=9) as bz2_handle:
                    for line in file_obj:
                        bz2_handle.write(line)
                handle.flush()
                handle.seek(0)
                gdata.base.repository.put_object_from_filelike(handle, gdata._comp_file_name)
        return gdata
