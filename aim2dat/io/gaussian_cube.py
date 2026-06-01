"""Module of functions to read/write Gaussian cube files."""

# Standard library imports
from typing import TYPE_CHECKING, Union

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.io.utils import (
    read_structure,
    write_structure,
    custom_open,
)
from aim2dat.units import UnitConverter, length

if TYPE_CHECKING:
    from aim2dat.strct import Structure, StructureCollection


@read_structure(r".*\.cube", preset_kwargs={"unit": "angstrom"})
def read_gaussian_cube_file(file_path: str, unit: str = "angstrom", get_data: bool = True) -> dict:
    """
    Read Gaussian cube file.

    Parameters
    ----------
    file_path : str
        File path or file content.
    unit : str
        Length unit of the returned cell parameters and positions.
    get_data : bool
        Whether to return the cube data in addition to the structural data.

    Returns
    -------
    dict
        Dictionary containing the structural data.
    """
    if unit not in length.available_units:
        raise ValueError(
            f"Invalid unit '{unit}'. Supported values are "
            + ", ".join(length.available_units)
            + "."
        )
    strct_dict = {
        "elements": [],
        "positions": [],
        "pbc": True,
        "site_attributes": {"atomic_charges": []},
    }
    cube_dict = {}
    with custom_open(file_path, "r") as f_obj:
        cube_dict["title"] = f_obj.readline().strip()
        cube_dict["comment"] = f_obj.readline().strip()

        # Read origin:
        line = f_obj.readline().split()
        natoms = int(line[0])
        cube_dict["origin"] = [float(val) for val in line[1:4]]
        has_dset = False
        if natoms < 0:
            has_dset = True
            natoms = abs(natoms)
        n_val = 1
        if len(line) == 5:
            n_val = int(line[4])

        # Read cell:
        cell = []
        shape = []
        for _ in range(3):
            line = f_obj.readline().split()
            shape.append(int(line[0]))
            cell.append(
                [
                    UnitConverter.convert_units(float(val), "bohr", unit) * int(line[0])
                    for val in line[1:4]
                ]
            )
        strct_dict["cell"] = cell
        cube_dict["shape"] = shape

        # Read atoms:
        for _ in range(natoms):
            line = f_obj.readline().split()
            strct_dict["elements"].append(int(line[0]))
            strct_dict["site_attributes"]["atomic_charges"].append(float(line[1]))
            strct_dict["positions"].append(
                [
                    UnitConverter.convert_units(float(val) - org, "bohr", unit)
                    for val, org in zip(line[2:5], cube_dict["origin"])
                ]
            )

        # Read dset-ids:
        cube_dict["dset_ids"] = None
        if has_dset:
            dset_ids = []
            line = f_obj.readline().split()
            n_dset = int(line[0])
            n_val *= n_dset
            dset_ids += [int(val) for val in line[1:]]
            while len(dset_ids) < n_dset:
                line = f_obj.readline().split()
                dset_ids += [int(val) for val in line[1:]]
            cube_dict["dset_ids"] = dset_ids
        cube_dict["n_values"] = n_val
        cube_dict["data_start"] = f_obj.tell()
        if get_data:
            cube_dict["data"] = _get_cube_data(
                f_obj, cube_dict["shape"], cube_dict["n_values"]
            ).tolist()
    strct_dict["attributes"] = {"cube": cube_dict}
    return strct_dict


@write_structure(r".*\.cube", preset_kwargs=None, writes_site_attributes=False)
def write_gaussian_cube_file(
    file_path: str,
    structure: Union["Structure", "StructureCollection", list],
):
    """
    Write Gaussian cube file. Cube data needs to present in the `attributes` dictionary.

    Parameters
    ----------
    file_path : str
        Path to Gaussian cube file.
    structure : aim2dat.strct.Structure, aim2dat.strct.StructureCollection, list
        Structure object, StructureCollection object or list of Structure objects. For the latter
        two cases the first item/structure is written to file.
    """
    if isinstance(structure, list) or type(structure).__name__ == "StructureCollection":
        structure = structure[0]

    cube_dict = structure.attributes.get("cube", None)
    if cube_dict is None:
        raise ValueError("Missing 'cube' attribute, cannot write cube file.")
    if structure.cell is None:
        raise ValueError("`Structure` needs to have `cell` defined.")

    val_p_line = 6
    with open(file_path, "w") as f_obj:
        f_obj.write(cube_dict["title"] + "\n")
        f_obj.write(cube_dict["comment"] + "\n")
        n_atoms = len(structure)
        n_values = ""
        if cube_dict["dset_ids"] is not None:
            n_atoms *= -1
        elif cube_dict["n_values"] > 1:
            n_values = f" {cube_dict['n_values']:6d}"
        f_obj.write(
            f"{n_atoms:6d} "
            + " ".join(f"{v:16.8f}" for v in cube_dict["origin"])
            + f"{n_values}\n"
        )
        for cell_v, shape in zip(structure.cell, cube_dict["shape"]):
            line = f"{shape:6d}"
            for v in cell_v:
                v = UnitConverter.convert_units(v, "angstrom", "bohr")
                if abs(v) > 1e-5:
                    v /= shape
                line += f" {v:16.8f}"
            f_obj.write(line + "\n")
        atomic_charges = structure.site_attributes.get("atomic_charges", [0.0] * len(structure))
        for nr, ac, pos in zip(structure.numbers, atomic_charges, structure.positions):
            line = f"{nr:6d} {ac:16.8f}"
            for p in pos:
                p = UnitConverter.convert_units(p, "angstrom", "bohr")
                line += f" {p:16.8f}"
            f_obj.write(line + "\n")
        if cube_dict["dset_ids"] is not None:
            dset_vals = [len(cube_dict["dset_ids"])] + cube_dict["dset_ids"]
            for i in range(0, len(dset_vals), val_p_line):
                f_obj.write(" ".join(f"{v:6d}" for v in dset_vals[i : i + val_p_line]) + "\n")

        data = np.array(cube_dict["data"]).flatten()
        for i in range(0, len(data), val_p_line):
            f_obj.write(" ".join(f"{v:16.8f}" for v in data[i : i + val_p_line]) + "\n")


def _get_cube_data(f_obj, shape, n_values):
    data = np.zeros(shape[0] * shape[1] * shape[2] * n_values)
    counter = 0
    for line in f_obj:
        line = line.split()
        data[counter : counter + len(line)] = [float(val) for val in line]
        counter += len(line)
    if n_values > 1:
        data = data.reshape(shape + [n_values])
    else:
        data = data.reshape(shape)
    return data
