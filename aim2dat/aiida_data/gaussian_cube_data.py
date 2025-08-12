"""AiiDA data classes for gaussian cube files."""

# Standard library imports
import contextlib
import tempfile
import bz2

# Third party library imports
import numpy as np
from aiida.orm import Data

# Internal library imports
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
            [val * UnitConverter.convert_units(1.0, "bohr", unit) for val in cellv]
            for cellv in self.cell
        ]
        positions = [
            [
                val * UnitConverter.convert_units(1.0, "bohr", unit) - org
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
        data = np.zeros(
            self.shape[0] * self.shape[1] * self.shape[2] * self.base.attributes.get("n_values")
        )
        counter = 0
        with self.open_cube() as handle:
            handle.seek(self.base.attributes.get("data_start"))
            # Read data:
            for line in handle:
                line = line.split()
                data[counter : counter + len(line)] = [float(val) for val in line]
                counter += len(line)
        if self.base.attributes.get("n_values") > 1:
            data = data.reshape(self.shape + [self.base.attributes.get("n_values")])
        else:
            data = data.reshape(self.shape)
        return data

    @classmethod
    def set_from_file(cls, file_obj):
        """
        Set information from existing cube file.

        Returns
        -------
        GaussianCubeData
            Data object containing the information from the file.
        """
        gdata = GaussianCubeData()
        # Read header:
        gdata.base.attributes.set("title", file_obj.readline().strip())
        gdata.base.attributes.set("comment", file_obj.readline().strip())

        # Read origin:
        line = file_obj.readline().split()
        natoms = int(line[0])
        gdata.base.attributes.set("origin", [float(val) for val in line[1:4]])
        has_dset = False
        if natoms < 0:
            has_dset = True
            natoms = abs(natoms)
        n_val = 1
        if len(line) == 5:
            n_val = int(line[4])
        gdata.base.attributes.set("n_values", n_val)

        # Read cell:
        cell = []
        shape = []
        for _ in range(3):
            line = file_obj.readline().split()
            shape.append(int(line[0]))
            cell.append([float(val) * int(line[0]) for val in line[1:4]])
        gdata.base.attributes.set("cell", cell)
        gdata.base.attributes.set("shape", shape)

        # Read atoms:
        atomic_numbers = []
        atomic_charges = []
        atomic_positions = []
        for _ in range(natoms):
            line = file_obj.readline().split()
            atomic_numbers.append(int(line[0]))
            atomic_charges.append(float(line[1]))
            atomic_positions.append([float(val) for val in line[2:5]])
        gdata.base.attributes.set("atomic_numbers", atomic_numbers)
        gdata.base.attributes.set("atomic_charges", atomic_charges)
        gdata.base.attributes.set("atomic_positions", atomic_positions)

        # Read dset-ids:
        if has_dset:
            dset_ids = []
            line = file_obj.readline().split()
            n_dset = int(line[0])
            n_val *= n_dset
            dset_ids += [int(val) for val in line[1:]]
            while len(dset_ids) < n_dset:
                line = file_obj.readline().split()
                dset_ids += [int(val) for val in line[1:]]
            gdata.base.attributes.set("dset_ids", dset_ids)

        # Store file with bz2 compression:
        gdata.base.attributes.set("data_start", file_obj.tell())
        file_obj.seek(0)
        with tempfile.NamedTemporaryFile() as handle:
            with bz2.open(handle, "wt", compresslevel=9) as bz2_handle:
                for line in file_obj:
                    bz2_handle.write(line)
            handle.flush()
            handle.seek(0)
            gdata.base.repository.put_object_from_filelike(handle, gdata._comp_file_name)
        return gdata
