"""Read and write cif files."""

# Standard library imports
from warnings import warn

# Third party library imports
import numpy as np
import re
from scipy.spatial.distance import cdist

# Internal library imports
from aim2dat.io.utils import read_structure, custom_open
from aim2dat.io.base_parser import FLOAT, transform_str_value
from aim2dat.utils.strct import _get_cell_from_lattice_p
from aim2dat.utils.space_groups import get_space_group_details
from aim2dat.elements import get_element_symbol
from aim2dat.chem_f import compare_formulas, transform_list_to_dict, transform_str_to_dict


class _CIFDataBlock:
    """Class to process and store data blocks of cif-files."""

    _string_limiters = ["'", '"']
    _cell_fields = [
        "cell_length_a",
        "cell_length_b",
        "cell_length_c",
        "cell_angle_alpha",
        "cell_angle_beta",
        "cell_angle_gamma",
    ]
    _atomic_site_coord_fields = [
        "atom_site_fract_x",
        "atom_site_fract_y",
        "atom_site_fract_z",
    ]
    _symmetry_fields = ["symmetry_equiv_pos_as_xyz", "space_group_symop_operation_xyz"]
    _space_group_fields = [
        "space_group_name_hall",
        "symmetry_space_group_name_hall",
        "symmetry_space_group_name_h-m",
        "symmetry_space_group_name_h-m",
        "symmetry_int_tables_number",
        "space_group_it_number",
    ]
    _chem_formula_fields = ["chemical_formula_sum"]
    _pred_element_mapping = {"Ow": "O", "Hw": "H"}
    _sym_op_pattern = re.compile(
        rf"(?P<sign>[-+])?(?P<num>({FLOAT}))?(\/(?P<den>{FLOAT}))?(?P<coord>[x-z])?"
    )

    def __init__(self, title_line):
        self.title = "_".join(title_line.split("_")[1:])
        self.fields = {}
        self.loops = []
        self.in_loop = False
        self.in_ml_field = False
        self.current_multi_line_field = None
        self.current_loop = None

    def add_line(self, line_idx, line):
        line_tr = line.strip().lower()

        # Omit comment or empty line:
        if line.startswith("#") or line == "":
            return None

        # Truncate if comment starts somewhere midline and is not part of string value:
        if "#" in line:
            lt = line.split("#")
            if len(lt) == 1:
                line = lt[0]
            else:
                l1 = lt[0]
                l2 = "#".join(lt[1:])
                if not any([str_l in l1 and str_l in l2 for str_l in self._string_limiters]):
                    line = l1

        # Start of loop:
        if line_tr.startswith("loop_"):
            # In some cases multi line fields are not properly limited via semicolons.
            self._finalize_current_loop(line_idx, "")
            self._force_finalize_multi_line()

            self.current_loop = {"keys": [], "values": []}
            line_split = line.split()
            if len(line_split) > 1:
                line = " ".join(line.split()[1:])
            else:
                line = ""
            self._add_line_to_loop(line_idx, line)

        # Process loop content:
        elif self.current_loop is not None:
            self._add_line_to_loop(line_idx, line)

        # Field:
        elif line_tr.startswith("_"):
            # In some cases multi line fields are not properly limited via semicolons.
            self._force_finalize_multi_line()

            line_split = line.split()
            if len(line_split) > 1:
                self.fields[self._transf_key(line_split[0])] = transform_str_value(
                    " ".join(line_split[1:])
                )
            else:
                self.current_multi_line_field = [self._transf_key(line_split[0])]

        # Multi-line field:
        elif self.current_multi_line_field is not None:
            self._extract_multi_line_value(line_idx, line)

    def finalize(self, line_idx):
        self._finalize_current_loop(line_idx, "")
        self._force_finalize_multi_line()

    def get_output(self):
        outp_dict = self.fields.copy()
        outp_dict["loops"] = self.loops
        return outp_dict

    def get_cell(self):
        if all(f0 in self.fields for f0 in self._cell_fields):
            return _get_cell_from_lattice_p(*[self.fields[f0] for f0 in self._cell_fields])

    def get_atomic_sites(self, check_chem_formula, get_sym_op_from_sg):
        # Get original sites:
        kinds = []
        scaled_coords = []
        kind_el_mapping = {}
        prel_site_attributes = {}
        for loop in self.loops:
            if "atom_site_label" not in loop:
                continue

            if all(k0 in loop for k0 in self._atomic_site_coord_fields):
                scaled_coords0 = []
                for key in self._atomic_site_coord_fields:
                    scaled_coords0.append(loop[key])
                scaled_coords += list(zip(*scaled_coords0))
                kinds += loop["atom_site_label"]
            if "atom_site_type_symbol" in loop:
                for kind, el in zip(loop["atom_site_label"], loop["atom_site_type_symbol"]):
                    kind_el_mapping[kind] = self._extract_element(el)
            for key, values in loop.items():
                if (
                    key
                    not in ["atom_site_label", "atom_site_type_symbol"]
                    + self._atomic_site_coord_fields
                ):
                    prel_site_attributes[key] = values.copy()

        # Try to get element symbols from site labels:
        if len(kind_el_mapping) == 0:
            for kind in kinds:
                kind_el_mapping[kind] = self._extract_element(kind)

        # In case site attributes are given, check consistency:
        site_attributes = {}
        for key, val in prel_site_attributes.items():
            if len(val) == len(kinds):
                site_attributes[key] = val

        # Remove atoms that occupy the same site:
        scaled_coords_bf = [[round(p0, 15) % 1 for p0 in pos] for pos in scaled_coords]
        dists = cdist(np.array(scaled_coords_bf), np.array(scaled_coords_bf))
        ind2del = set([j for i, j in zip(*np.where(dists <= 1e-3)) if i < j])
        if len(ind2del) > 0:
            warn(
                f"The sites {ind2del} are omitted as they are duplicate of other sites.",
                UserWarning,
            )
            for idx in reversed(sorted(ind2del)):
                del kinds[idx]
                del scaled_coords[idx]
                del scaled_coords_bf[idx]
                for val in site_attributes.values():
                    del val[idx]

        # Add sites from symmetry operations:
        sym_ops = self.get_symmetry_operations(get_sym_op_from_sg)
        sym_ops = [(np.array(rot), np.array(trans)) for rot, trans in sym_ops]
        n_sites = len(kinds)
        for sym_op in sym_ops:
            for idx in range(n_sites):
                new_pos = np.dot(sym_op[0], np.array(scaled_coords[idx])) + sym_op[1]
                new_pos_bf = [round(val, 15) % 1 for val in new_pos]
                dists = cdist(np.array([new_pos_bf]), np.array(scaled_coords_bf))[0]
                if any(dist < 1e-3 for dist in dists):
                    continue
                kinds.append(kinds[idx])
                scaled_coords.append(tuple(new_pos.tolist()))
                scaled_coords_bf.append(new_pos_bf)
                for val in site_attributes.values():
                    val.append(val[idx])

        # In case elements and kinds coincide only elements is considered:
        elements = [kind_el_mapping[kind] for kind in kinds]
        if all(el == k for el, k in zip(elements, kinds)):
            kinds = None

        # Check sum formula in case given:
        if check_chem_formula:
            chem_formula = self.get_chem_formula()
            if chem_formula is not None:
                if not compare_formulas(
                    chem_formula, transform_list_to_dict(elements), reduce_formulas=True
                ):
                    raise ValueError("Chemical formula doesn't match with number of sites.")

        return elements, kinds, scaled_coords, site_attributes

    def get_symmetry_operations(self, get_sym_op_from_sg):
        sym_op_strings = []
        for loop in self.loops:
            for field_key in self._symmetry_fields:
                if field_key in loop:
                    sym_op_strings += loop[field_key]
        sym_ops = []
        for sym_op in sym_op_strings:
            rot_matrix = np.zeros((3, 3))
            shift = np.zeros(3)
            for coord_idx, sym_str in enumerate(sym_op.split(",")):
                sym_str = sym_str.replace(" ", "").lower()
                for m in self._sym_op_pattern.finditer(sym_str):
                    m = m.groupdict()
                    if all(val is None for val in m.values()):
                        continue
                    pref = -1.0 if m["sign"] == "-" else 1.0
                    pref *= float(m["num"]) if m["num"] is not None else 1.0
                    pref /= float(m["den"]) if m["den"] is not None else 1.0
                    if m["coord"] is None:
                        shift[coord_idx] += pref
                    else:
                        rot_matrix[coord_idx]["xyz".index(m["coord"])] += pref
            sym_ops.append((rot_matrix.tolist(), shift.tolist()))
        if get_sym_op_from_sg and len(sym_ops) == 0:
            sg_details = self.get_space_group_details(return_sym_operations=True)
            if sg_details is not None:
                warn(
                    "Could not determine symmetry operations directly, using space group details.",
                    UserWarning,
                )
                sym_ops += sg_details["symmetry_operations"]
        return sym_ops

    def get_space_group_details(self, return_sym_operations=False):
        for key in self._space_group_fields:
            if key in self.fields:
                return get_space_group_details(
                    self.fields[key], return_sym_operations=return_sym_operations
                )

    def get_chem_formula(self):
        for key in self._chem_formula_fields:
            if key in self.fields:
                return transform_str_to_dict(self.fields[key])

    def _add_line_to_loop(self, line_idx, line):
        # Adding loop labels:
        if line.startswith("_"):
            # If line starts with new field label and values have been added we assume that the
            # loop is completed.
            if len(self.current_loop["values"]) > 0:
                self._finalize_current_loop(line_idx, line)
                self.add_line(line_idx, line)
            else:
                self.current_loop["keys"] += [self._transf_key(sp[1:]) for sp in line.split()]

        # Adding loop values:
        else:
            # Multi-line field in loop:
            if line.startswith(";") or self.current_multi_line_field is not None:
                if self.current_multi_line_field is None:
                    self.current_multi_line_field = [""]
                self._extract_multi_line_value(line_idx, line)
            # Single value fields:
            else:
                self._add_loop_values(self._extract_loop_values(line))

    def _add_loop_values(self, loop_values):
        if len(self.current_loop["values"]) == 0:
            start_idx = 0
            self.current_loop["values"] = [[] for _ in self.current_loop["keys"]]
        else:
            val_lengths = [len(val) for val in self.current_loop["values"]]
            start_idx = val_lengths.index(min(val_lengths))
        for val_idx, val in enumerate(loop_values):
            val_idx += start_idx
            if len(self.current_loop["keys"]) > 0:
                val_idx %= len(self.current_loop["keys"])
            self.current_loop["values"][val_idx].append(val)

    def _extract_multi_line_value(self, line_idx, line):
        finalize = False
        add_str = None
        new_str = ""
        line_split = line.split(";")
        if len(line_split) == 1:
            add_str = line
        else:
            if len(line_split) + len(self.current_multi_line_field) > 3:
                finalize = True
            add_str = line_split[0]
            new_str = line_split[-1]
            if line_split[0] == "" and len(self.current_multi_line_field) == 1:
                add_str = line_split[1]
                if len(line_split) == 2:
                    new_str = ""
        if add_str is not None:
            if len(self.current_multi_line_field) == 1:
                self.current_multi_line_field.append(add_str)
            else:
                self.current_multi_line_field[1] += "\n" + add_str

        if finalize:
            self._force_finalize_multi_line()
        self.add_line(line_idx, new_str)

    def _force_finalize_multi_line(self):
        if self.current_multi_line_field is None:
            return None

        val = ""
        if len(self.current_multi_line_field) == 2:
            val = self.current_multi_line_field[1]
        if self.current_loop is None:
            self.fields[self.current_multi_line_field[0]] = val
        else:
            self._add_loop_values([val])
        self.current_multi_line_field = None

    def _extract_loop_values(self, line):
        line_split = line.split()
        loop_values = []
        str_val_limiter = None
        for val in line_split:
            if val[0] in self._string_limiters and str_val_limiter is None:
                str_val_limiter = val[0]
                val = val[1:] if len(val) > 1 else ""
                loop_values.append("")
            if str_val_limiter is not None:
                if val.endswith(str_val_limiter):
                    val = val[:-1]
                    str_val_limiter = None
                loop_values[-1] += " " + val
            else:
                loop_values.append(val)
        return [transform_str_value(val) for val in loop_values]

    def _finalize_current_loop(self, line_idx, line):
        if self.current_loop is None:
            return None

        self._force_finalize_multi_line()
        if any(
            len(self.current_loop["values"][0]) != len(self.current_loop["values"][idx])
            for idx in range(len(self.current_loop["values"]))
        ):
            raise ValueError(f"Number of values differ for loop finishing on line {line_idx}.")
        self.loops.append(
            {
                key: self.current_loop["values"][idx]
                for idx, key in enumerate(self.current_loop["keys"])
            }
        )
        self.current_loop = None

    @staticmethod
    def _transf_key(key):
        return key.strip("_").lower()

    def _extract_element(self, value):
        value = re.split(r"(\d)|(_)|(-)|(\+)", str(value))[0]
        el = self._pred_element_mapping.get(value, None)
        if el is None:
            try:
                el = get_element_symbol(value)
            except ValueError:
                raise ValueError(f"Could not determine element of '{value}'.")
        return el


@read_structure(r".*\.cif", preset_kwargs={"extract_structures": True})
def read_cif_file(
    file_path: str,
    extract_structures: bool = False,
    strct_check_chem_formula: bool = True,
    strct_get_sym_op_from_sg: bool = True,
    strct_wrap: bool = False,
) -> dict:
    """
    Read cif file.

    Parameters
    ----------
    file_path : str
        Path to the cif file.
    extract_structures : bool (optional)
        Whether to extract alls crystal structures and add them to the output dictionary with the
        key ``'structures'``.
    strct_check_chem_formula : bool (optional)
        Check the chemical formula given by field matches with the structure.
    strct_get_sym_op_from_sg : bool (optional)
        Add symmetry operations based on the space group to add symmetry equivalent sites to the
        structures.

    Returns
    -------
    dict
        Output dictionary.
    """
    cif_blocks = []
    current_block = None
    with custom_open(file_path, "r") as f_obj:
        for line_idx, line in enumerate(f_obj):
            line = line.strip()
            if line.startswith("data_"):
                if current_block:
                    current_block.finalize(line_idx)
                    cif_blocks.append(current_block)
                current_block = _CIFDataBlock(line)
            elif current_block:
                current_block.add_line(line_idx, line)
        current_block.finalize(line_idx)
        cif_blocks.append(current_block)

    output_dict = {}
    structures = []
    for block in cif_blocks:
        if block.title in output_dict:
            warn(f"Two data bloocks have the same title: '{block.title}'.", UserWarning)
        if extract_structures:
            if block.title == "structures":
                warn("Data block 'structures' is overwritten.", UserWarning)
            cell = block.get_cell()
            if cell is not None:
                elements, kinds, positions, site_attributes = block.get_atomic_sites(
                    strct_check_chem_formula, strct_get_sym_op_from_sg
                )
                structures.append(
                    {
                        "cell": cell,
                        "label": block.title,
                        "elements": elements,
                        "kinds": kinds,
                        "site_attributes": site_attributes,
                        "positions": positions,
                        "pbc": True,
                        "is_cartesian": False,
                        "wrap": strct_wrap,
                    }
                )
        output_dict[block.title] = block.get_output()
    if extract_structures:
        output_dict["structures"] = structures
    return output_dict


@read_structure(r".*\.cif", preset_kwargs={"extract_structures": True})
def read_file(
    file_path: str,
    extract_structures: bool = False,
    strct_check_chem_formula: bool = True,
    strct_get_sym_op_from_sg: bool = True,
    strct_wrap: bool = False,
) -> dict:
    """
    Read cif file.

    Notes
    -----
        This function is deprecated and will be removed, please use `aim2dat.io.read_cif_file`
        instead.

    Parameters
    ----------
    file_path : str
        Path to the cif file.
    extract_structures : bool (optional)
        Whether to extract alls crystal structures and add them to the output dictionary with the
        key ``'structures'``.
    strct_check_chem_formula : bool (optional)
        Check the chemical formula given by field matches with the structure.
    strct_get_sym_op_from_sg : bool (optional)
        Add symmetry operations based on the space group to add symmetry equivalent sites to the
        structures.

    Returns
    -------
    dict
        Output dictionary.
    """
    from warnings import warn

    warn(
        "This function will be removed, please use `aim2dat.io.read_cif_file` instead.",
        DeprecationWarning,
        2,
    )
    return read_cif_file(
        file_path=file_path,
        extract_structures=extract_structures,
        strct_check_chem_formula=strct_check_chem_formula,
        strct_get_sym_op_from_sg=strct_get_sym_op_from_sg,
        strct_wrap=strct_wrap,
    )
