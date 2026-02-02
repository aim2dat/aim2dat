"""
Module of functions to read/write xsf files.

Notes
-----
So far, data blocks and grids are not yet supported and neglected.
"""

# Standard library imports
import warnings
from typing import TYPE_CHECKING, Union, List

# Internal library imports
from aim2dat.io.utils import custom_open, read_structure, write_structure

if TYPE_CHECKING:
    from aim2dat.strct import Structure, StructureCollection

_PBC_SETTINGS = {
    "molecule": False,
    "polymer": [True, False, False],
    "slab": [True, True, False],
    "crystal": True,
}


@read_structure(r".*\.xsf", preset_kwargs=None)
def read_xsf_file(file_path: str) -> List[dict]:
    """
    Read xsf file, as specified in http://www.xcrysden.org/doc/XSF.html.

    Parameters
    ----------
    file_path : str
        File path or file content.

    Returns
    -------
    list
        List of dictionaries containing structural information.
    """
    structures = [{"pbc": False}]
    n_structures = 1
    current_section = ""
    current_strct_indices = [0]
    with custom_open(file_path, "r") as f_obj:
        for line in f_obj:
            if line.strip().startswith("#"):
                continue

            line_sp = line.split()
            if len(line_sp) == 0:
                continue

            sect_val = line_sp[0].lower()
            if sect_val.startswith("begin_"):
                current_section = "data_block"
            elif len(line_sp) == 2 and sect_val == "animsteps":
                n_structures = int(line_sp[1])
                current_strct_indices = range(n_structures)
                structures = [{"pbc": False} for _ in range(n_structures)]
            elif sect_val in _PBC_SETTINGS.keys():
                for strct in structures:
                    strct["pbc"] = _PBC_SETTINGS[sect_val]
            elif sect_val in ["atoms", "primvec", "convvec", "primcoord", "convcoord"]:
                current_section = sect_val
                if len(line_sp) == 2:
                    current_strct_indices = [int(line_sp[1]) - 1]
            elif current_section == "primvec":
                for i in current_strct_indices:
                    structures[i].setdefault("cell", []).append([float(v) for v in line_sp[0:3]])
            elif current_section in ["primcoord", "atoms"]:
                if len(line_sp) == 2:
                    continue
                else:
                    for i in current_strct_indices:
                        structures[i].setdefault("elements", []).append(int(line_sp[0]))
                        structures[i].setdefault("positions", []).append(
                            [float(v) for v in line_sp[1:4]]
                        )
                        if len(line_sp) == 7:
                            site_attrs = structures[i].setdefault("site_attributes", {})
                            site_attrs.setdefault("forces", []).append(
                                [float(v) for v in line_sp[4:7]]
                            )
    return structures


@write_structure(r".*\.xsf", preset_kwargs=None, writes_site_attributes=True)
def write_xsf_file(
    file_path: str,
    structures: Union["Structure", "StructureCollection", list],
    include_site_attributes: list = None,
    exclude_site_attributes: list = None,
):
    """
    Write xsf file.

    Parameters
    ----------
    file_path : str
        Path to xsf file.
    structures : aim2dat.strct.Structure, aim2dat.strct.StructureCollection, list
        Structure object, StructureCollection object or list of Structure objects.
    include_site_attributes : list
        List of site attributes that are written to file.
    exclude_site_attributes : list
        List of site attributes that are not written to file.
    """
    structures = [structures] if type(structures).__name__ == "Structure" else structures
    exclude_site_attributes = [] if exclude_site_attributes is None else exclude_site_attributes
    if (
        include_site_attributes is not None
        and any(attr != "forces" for attr in include_site_attributes)
    ) or any(attr != "forces" for attr in exclude_site_attributes):
        warnings.warn(
            "The current implementation of the 'xsf' file parser only supports "
            + "'forces' as `site_attributes`."
        )

    all_pbc = [strct.pbc for strct in structures]
    if all(pbc == (False, False, False) for pbc in all_pbc):
        structure_type = "ATOMS"
    elif all(pbc == (True, False, False) for pbc in all_pbc):
        structure_type = "POLYMER"
    elif all(pbc == (True, True, False) for pbc in all_pbc):
        structure_type = "SLAB"
    else:
        structure_type = "CRYSTAL"

    is_fixed_cell = False
    if structure_type != "ATOMS":
        all_cells = [strct.cell for strct in structures]
        is_fixed_cell = True
        for cell in all_cells[1:]:
            if any(
                [
                    abs(v0 - v1) > 1e-8
                    for cv0, cv1 in zip(all_cells[0], cell)
                    for v0, v1 in zip(cv0, cv1)
                ]
            ):
                is_fixed_cell = False
                break

    with open(file_path, "w") as f_obj:
        if len(structures) > 1:
            f_obj.write(f"ANIMSTEPS {len(structures)}\n")
        for idx, strct in enumerate(structures):
            strct_idx = "" if len(structures) == 1 else f" {idx + 1}"
            if structure_type == "ATOMS":
                f_obj.write(f"ATOMS{strct_idx}\n")
            else:
                f_obj.write(f"{structure_type}\n")
                if not is_fixed_cell or (is_fixed_cell and idx == 0):
                    prim_vec_str = f"PRIMVEC{strct_idx}\n" if not is_fixed_cell else "PRIMVEC\n"
                    f_obj.write(prim_vec_str)
                    for cell_v in strct.cell:
                        f_obj.write(f"{cell_v[0]:16.8f} {cell_v[1]:16.8f} {cell_v[2]:16.8f}\n")
                f_obj.write(f"PRIMCOORD{strct_idx}\n")
                f_obj.write(f"{len(strct)} 1\n")
            forces = ["" for _ in range(len(strct))]
            add_forces = False
            if "forces" in strct.site_attributes:
                if include_site_attributes is not None:
                    if "forces" in include_site_attributes:
                        add_forces = True
                elif "forces" not in exclude_site_attributes:
                    add_forces = True
            if add_forces:
                for i, force in enumerate(strct.site_attributes["forces"]):
                    try:
                        forces[i] = f" {force[0]:16.8f} {force[1]:16.8f} {force[2]:16.8f}"
                    except (ValueError, IndexError):
                        warnings.warn(f"Cannot parse force of site {i}.")
                        forces = ["" for _ in range(len(strct))]
                        break

            for nr, pos, force in zip(strct.numbers, strct.positions, forces):
                nr = str(nr)
                nr = "".join(" " for _ in range(2 - len(nr))) + nr
                f_obj.write(f"{nr} {pos[0]:16.8f} {pos[1]:16.8f} {pos[2]:16.8f}{force}\n")
