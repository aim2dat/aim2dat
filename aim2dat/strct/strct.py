"""Module implementing a Structure class."""

# Standard library imports
import copy
from typing import List, Union

# Third party library imports
import numpy as np
from ase import Atoms

try:
    import aiida
except ImportError:
    aiida = None

try:
    import pymatgen
except ImportError:
    pymatgen = None

# Internal library imports
from aim2dat.ext_interfaces import _return_ext_interface_modules
from aim2dat.strct.strct_validation import (
    _structure_validate_cell,
    _structure_validate_elements,
    _structure_validate_positions,
)
from aim2dat.strct.mixin import AnalysisMixin, ManipulationMixin
import aim2dat.utils.chem_formula as utils_cf
import aim2dat.utils.print as utils_pr
from aim2dat.utils.maths import calc_angle


def _compare_function_args(args1, args2):
    """Compare function arguments to check if a property needs to be recalculated."""
    for kwarg, value1 in args1.items():
        if value1 != args2[kwarg]:
            return False
    return True


def _create_index_dict(value):
    index_dict = {}
    for idx, val in enumerate(value):
        if val in index_dict:
            index_dict[val].append(idx)
        else:
            index_dict[val] = [idx]
    return index_dict


def _check_calculated_properties(structure, func, func_args):
    property_name = "_".join(func.__name__.split("_")[1:])
    if structure.store_calculated_properties and property_name in structure._function_args:
        if _compare_function_args(structure._function_args[property_name], func_args):
            return structure.extras[property_name]
    calc_attr, calc_extra = func(structure, **func_args)
    if calc_attr is not None:
        structure.set_attribute(property_name, calc_attr)
    if structure.store_calculated_properties:
        if calc_extra is not None:
            structure._extras[property_name] = calc_extra
        structure._function_args[property_name] = func_args
    return calc_extra


def import_method(func):
    """Mark function as import function."""
    func._is_import_method = True
    return func


def export_method(func):
    """Mark function as export function."""
    func._is_export_method = True
    return func


class Structure(AnalysisMixin, ManipulationMixin):
    """
    Represents a structure and contains methods to calculate properties of a structure
    (molecule or crystal) or to manipulate a structure.
    """

    def __init__(
        self,
        elements: List[str],
        positions: List[List[float]],
        pbc: List[bool],
        is_cartesian: bool = True,
        wrap: bool = False,
        cell: List[List[float]] = None,
        kinds: List[str] = None,
        label: str = None,
        store_calculated_properties: bool = True,
        attributes: dict = None,
        extras: dict = None,
        function_args: dict = None,
    ):
        """Initialize object."""
        self._inverse_cell = None

        self.elements = elements
        self.kinds = kinds
        self.cell = cell
        self.pbc = pbc
        self.label = label
        self.store_calculated_properties = store_calculated_properties

        self._attributes = {} if attributes is None else attributes
        self._extras = {} if extras is None else extras
        self._function_args = {} if function_args is None else function_args

        self.set_positions(positions, is_cartesian=is_cartesian, wrap=wrap)

    def __str__(self):
        """Represent object as string."""

        def _parse_vector(vector):
            vector = ["{0:.4f}".format(val) for val in vector]
            return "[" + " ".join([" ".join([""] * (9 - len(val))) + val for val in vector]) + "]"

        output_str = utils_pr._print_title(f"Structure: {self.label}") + "\n\n"
        output_str += " Formula: " + utils_cf.transform_dict_to_str(self.chem_formula) + "\n"
        output_str += " PBC: [" + " ".join(str(val) for val in self.pbc) + "]\n\n"

        if self.cell is not None:
            output_str += utils_pr._print_subtitle("Cell") + "\n"
            # output_str += utils_pr._print_subtitle("Cell")
            output_str += utils_pr._print_list(
                "Vectors:", [_parse_vector(val) for val in self.cell]
            )
            output_str += " Lengths: " + _parse_vector(self.cell_lengths) + "\n"
            output_str += " Angles: " + _parse_vector(self.cell_angles) + "\n"
            output_str += " Volume: {0:.4f}\n\n".format(self.cell_volume)

        output_str += utils_pr._print_subtitle("Sites") + "\n"
        sites_list = []
        for el, kind, cart_pos, scaled_pos in self.iter_sites(
            get_kind=True, get_scaled_pos=True, get_cart_pos=True
        ):
            site_str = f"{el} " + " ".join([""] * (3 - len(el)))
            site_str += (
                f"{kind} " + " ".join([""] * (6 - len(str(kind)))) + _parse_vector(cart_pos)
            )
            if scaled_pos is not None:
                site_str += " " + _parse_vector(scaled_pos)
            sites_list.append(site_str)
        output_str += utils_pr._print_list("", sites_list)
        output_str += utils_pr._print_hline()
        return output_str

    def __len__(self):
        """int: Get number of sites."""
        return len(self.elements)

    def __iter__(self):
        """Iterate through element and cartesian position."""
        for el, pos in zip(self.elements, self.positions):
            yield el, pos

    def __contains__(self, key: str):
        """Check whether Structure contains the key."""
        keys_to_check = [
            d for d in dir(self) if not callable(getattr(self, d)) and not d.startswith("_")
        ]
        return key in keys_to_check  # key in self.__dict__.keys()

    def __getitem__(self, key: str):
        """Return structure property by key or list of keys."""
        if isinstance(key, list):
            try:
                return {k: getattr(self, k) for k in key}
            except AttributeError:
                raise KeyError(f"Key `{key} is not present.")
        elif isinstance(key, str):
            try:
                return getattr(self, key)
            except AttributeError:
                raise KeyError(f"Key `{key} is not present.")

    def __deepcopy__(self, memo):
        """Create a deepcopy of the object."""
        copy = Structure(
            elements=self.elements,
            positions=self.positions,
            cell=self.cell,
            pbc=self.pbc,
            is_cartesian=True,
            kinds=self.kinds,
            attributes=self.attributes,
            extras=self.extras,
            function_args=self.function_args,
            label=self.label,
            store_calculated_properties=self.store_calculated_properties,
        )
        memo[id(self)] = copy
        return copy

    def keys(self) -> list:
        """Return property names to create the structure."""
        return [
            "elements",
            "positions",
            "pbc",
            "cell",
            "kinds",
            "attributes",
            "extras",
        ]

    def copy(self) -> "Structure":
        """Return copy of `Structure` object."""
        return copy.deepcopy(self)

    def get(self, key, value=None):
        """Get attribute by key and return default if not present."""
        try:
            if self[key] is None:
                return value
            else:
                return self[key]
        except KeyError:
            return value

    @property
    def label(self) -> Union[str, None]:
        """Return label of the structure (especially relevant in StructureCollection)."""
        return self._label

    @label.setter
    def label(self, value):
        if value is not None and not isinstance(value, str):
            raise TypeError("`label` needs to be of type str.")
        self._label = value

    @property
    def elements(self) -> tuple:
        """Return the elements of the structure."""
        return self._elements

    @elements.setter
    def elements(self, value: Union[tuple, list, np.ndarray]):
        elements = _structure_validate_elements(value)
        if self.positions is not None and len(self.positions) != len(elements):
            raise ValueError("Length of `elements` is unequal to length of `positions`.")
        self._elements = elements
        self._element_dict = _create_index_dict(elements)
        self._chem_formula = utils_cf.transform_list_to_dict(elements)

    @property
    def chem_formula(self) -> dict:
        """
        Return chemical formula.
        """
        return self._chem_formula

    @property
    def positions(self) -> tuple:
        """tuple: Return the cartesian positions of the structure."""
        return getattr(self, "_positions", None)

    @property
    def scaled_positions(self) -> Union[tuple, None]:
        """tuple or None: Return the scaled positions of the structure."""
        return getattr(self, "_scaled_positions", None)

    @property
    def pbc(self) -> tuple:
        """Return the pbc of the structure."""
        return self._pbc

    @pbc.setter
    def pbc(self, value: Union[tuple, list, np.ndarray, bool]):
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 3 and all(isinstance(pbc0, (bool, np.bool_)) for pbc0 in value):
                value = tuple([bool(pbc0) for pbc0 in value])
            else:
                raise ValueError("`pbc` must have a length of 3 and consist of boolean variables.")
        else:
            if isinstance(value, (bool, np.bool_)):
                value = tuple([bool(value), bool(value), bool(value)])
            else:
                raise TypeError("`pbc` must be a list, tuple or a boolean.")
        if any(val for val in value) and self.cell is None:
            raise ValueError(
                "`cell` must be set if `pbc` is set to true for one or more direction."
            )
        self._pbc = value

    @property
    def cell(self) -> Union[tuple, None]:
        """Return the cell of the structure."""
        return getattr(self, "_cell", None)

    @cell.setter
    def cell(self, value: Union[tuple, list, np.ndarray]):
        if value is not None:
            self._cell, self._inverse_cell = _structure_validate_cell(value)
            self._cell_volume = abs(np.dot(np.cross(self._cell[0], self._cell[1]), self._cell[2]))
            self._cell_lengths = tuple([float(np.linalg.norm(vec)) for vec in self._cell])
            self._cell_angles = tuple(
                [
                    float(calc_angle(self._cell[i1], self._cell[i2]) * 180.0 / np.pi)
                    for i1, i2 in [(1, 2), (0, 2), (0, 1)]
                ]
            )
            # if hasattr(self, "_positions"):
            #     self.set_positions(self.positions, is_cartesian=True)

    @property
    def cell_volume(self) -> Union[float, None]:
        """tuple: cell volume."""
        return getattr(self, "_cell_volume", None)

    @property
    def cell_lengths(self) -> Union[tuple, None]:
        """tuple: cell lengths."""
        return getattr(self, "_cell_lengths", None)

    @property
    def cell_angles(self) -> Union[tuple, None]:
        """tuple: Cell angles."""
        return getattr(self, "_cell_angles", None)

    @property
    def kinds(self) -> Union[tuple, None]:
        """Return the kinds of the structure."""
        return self._kinds

    @kinds.setter
    def kinds(self, value: Union[tuple, list]):
        if value is not None:
            if not isinstance(value, (list, tuple)):
                raise TypeError("`kinds` must be a list or tuple.")
            if len(value) != len(self.elements):
                raise ValueError("`kinds` must have the same length as `elements`.")
            self._kind_dict = _create_index_dict(value)
            value = tuple(value)
        self._kinds = value

    @property
    def function_args(self) -> dict:
        """Return function arguments for stored extras."""
        return copy.deepcopy(self._function_args)

    @property
    def attributes(self) -> dict:
        """Return the specified attributes."""
        return copy.deepcopy(self._attributes)

    @property
    def extras(self) -> dict:
        """
        Return the specified extras.
        """
        return copy.deepcopy(self._extras)

    @property
    def store_calculated_properties(self) -> bool:
        """
        Store calculated properties to reuse them later.
        """
        return self._store_calculated_properties

    @store_calculated_properties.setter
    def store_calculated_properties(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("`store_calculated_properties` needs to be of type bool.")
        self._store_calculated_properties = value

    def iter_sites(
        self,
        get_kind: bool = False,
        get_cart_pos: bool = False,
        get_scaled_pos: bool = False,
        wrap: bool = False,
    ):
        """
        Iterate through the sites of the structure.

        Parameters
        ----------
        get_kind : bool (optional)
            Include kind in tuple.
        get_cart_pos : bool (optional)
            Include cartesian position in tuple.
        get_scaled_pos : bool (optional)
            Include scaled position in tuple.
        wrap : bool (optional)
            Wrap atomic positions back into the unit cell.

        Yields
        ------
        str or tuple
            Either element symbol or tuple containing the element symbol, kind string,
            cartesian position or scaled position.
        """
        for idx, el in enumerate(self.elements):
            output = [el]
            if get_kind:
                output.append(None if self.kinds is None else self.kinds[idx])
            pos_cart = self.positions[idx]
            pos_scaled = None if self.scaled_positions is None else self.scaled_positions[idx]
            if (get_cart_pos or get_scaled_pos) and wrap:
                pos_cart, pos_scaled = self._wrap_position(pos_cart, pos_scaled)
            if get_cart_pos:
                output.append(pos_cart)
            if get_scaled_pos:
                output.append(pos_scaled)
            if len(output) == 1:
                yield el
            else:
                yield tuple(output)

    def set_positions(
        self, positions: Union[list, tuple], is_cartesian: bool = True, wrap: bool = False
    ):
        """
        Set postions of atoms.

        Parameters
        ----------
        positions : list or tuple
            Nested list or tuple of the coordinates (n atoms x 3).
        is_cartesian : bool (optional)
            Whether the coordinates are cartesian or scaled.
        wrap : bool (optional)
            Wrap atomic positions into the unit cell.
        """
        if len(self.elements) != len(positions):
            raise ValueError("`elements` and `positions` must have the same length.")
        self._positions, self._scaled_positions = _structure_validate_positions(
            positions, is_cartesian, self.cell, self._inverse_cell, self.pbc
        )
        if wrap:
            new_positions = [
                pos for pos in self.iter_sites(get_cart_pos=True, get_scaled_pos=True, wrap=wrap)
            ]
            _, cart_positions, scaled_positions = zip(*new_positions)
            self._positions = tuple(cart_positions)
            self._scaled_positions = tuple(scaled_positions)

    def get_positions(self, cartesian: bool = True, wrap: bool = False):
        """
        Return positions of atoms.

        Parameters
        ----------
        cartesian : bool (optional)
            Get cartesian positions. If set to ``False`` scaled positions are returned.
        wrap : bool (optional)
            Wrap atomic positions into the unit cell.
        """
        return tuple(
            pos
            for _, pos in self.iter_sites(
                get_cart_pos=cartesian, get_scaled_pos=not cartesian, wrap=wrap
            )
        )

    def set_attribute(self, key: str, value):
        """
        Set attribute.

        Parameters
        ----------
        key : str
            Key of the attribute.
        value :
            Value of the attribute.
        """
        self._attributes[key] = value

    @classmethod
    @property
    def import_methods(cls) -> list:
        """list: Return import methods."""
        import_methods = []
        for name, method in cls.__dict__.items():
            if getattr(method, "_is_import_method", False):
                import_methods.append(name)
        return import_methods

    @classmethod
    @property
    def export_methods(cls) -> list:
        """list: Return export methods."""
        export_methods = []
        for name, method in Structure.__dict__.items():
            if getattr(method, "_is_export_method", False):
                export_methods.append(name)
        return export_methods

    @import_method
    @classmethod
    def from_file(cls, file_path: str, attributes: dict = None, label: str = None) -> "Structure":
        """
        Get structure from file using the ase read-function.

        Parameters
        ----------
        file_path : str
            File path.
        attributes : dict

        label : str
            Label used internally to store the structure in the object.

        Returns
        -------
        aiida_scripst.strct.Structure
            Structure.
        """
        backend_module = _return_ext_interface_modules("ase_atoms")
        structure_dicts = backend_module._load_structure_from_file(file_path)
        if len(structure_dicts) == 1:
            return cls(**structure_dicts[0], attributes=attributes, label=label)
        else:
            # TODO How to deal with label and attributes for multiple structures.
            return [
                cls(**structure_dict, attributes=attributes, label=label + f"_{idx}")
                for idx, structure_dict in enumerate(structure_dicts)
            ]

    @import_method
    @classmethod
    def from_ase_atoms(
        cls, ase_atoms: Atoms, attributes: dict = None, label: str = None
    ) -> "Structure":
        """
        Get structure from ase atoms object.

        Parameters
        ----------
        ase_atoms : ase.Atoms
            ase Atoms object.
        attributes : dict
            Additional information about the structure.
        label : str
            Label used internally to store the structure in the object.

        Returns
        -------
        aiida_scripst.strct.Structure
            Structure.
        """
        backend_module = _return_ext_interface_modules("ase_atoms")
        return cls(
            **backend_module._extract_structure_from_atoms(ase_atoms),
            label=label,
            attributes=attributes,
        )

    @import_method
    @classmethod
    def from_pymatgen_structure(
        cls,
        pymatgen_structure: Union["pymatgen.core.Molecule", "pymatgen.core.Structure"],
        attributes: dict = None,
        label: str = None,
    ) -> "Structure":
        """
        Get structure from pymatgen structure or molecule object.

        Parameters
        ----------
        pymatgen_structure : pymatgen.core.Structure or pymatgen.core.Molecule
            pymatgen structure or molecule object.
        attributes : dict
            Additional information about the structure.
        label : str
            Label used internally to store the structure in the object.

        Returns
        -------
        aiida_scripst.strct.Structure
            Structure.
        """
        backend_module = _return_ext_interface_modules("pymatgen")
        return cls(
            **backend_module._extract_structure_from_pymatgen(pymatgen_structure),
            label=label,
            attributes=attributes,
        )

    @import_method
    @classmethod
    def from_aiida_structuredata(
        cls,
        structure_node: Union[int, str, "aiida.orm.StructureData"],
        use_uuid: bool = False,
        label: str = None,
    ) -> "Structure":
        """
        Append structure from AiiDA structure node.

        Parameters
        ----------
        label : str
            Label used internally to store the structure in the object.
        structure_node : int, str or aiida.orm.nodes.data.structure.StructureData
            Primary key, UUID or AiiDA structure node.
        use_uuid : bool (optional)
            Whether to use the uuid (str) to represent AiiDA nodes instead of the primary key
            (int).

        Returns
        -------
        aiida_scripst.strct.Structure
            Structure.
        """
        backend_module = _return_ext_interface_modules("aiida")
        structure_dict = backend_module._extract_dict_from_aiida_structure_node(
            structure_node, use_uuid
        )
        if label is not None:
            structure_dict["label"] = label
        return cls(**structure_dict)

    @export_method
    def to_file(self, file_path: str) -> None:
        """
        Export structure to file using the ase interface.
        """
        backend_module = _return_ext_interface_modules("ase_atoms")
        backend_module._write_structure_to_file(self, file_path)

    @export_method
    def to_ase_atoms(self) -> Atoms:
        """
        Create ase Atoms object.

        Returns
        -------
        ase.Atoms
            ase Atoms object of the structure.
        """
        backend_module = _return_ext_interface_modules("ase_atoms")
        return backend_module._create_atoms_from_structure(self)

    @export_method
    def to_pymatgen_structure(self) -> Union["pymatgen.core.Molecule", "pymatgen.core.Structure"]:
        """
        Create pymatgen Structure (if cell is not `None`) or Molecule (if cell is `None`) object.

        Returns
        -------
        pymatgen.core.Structure or pymatgen.core.Molecule
            pymatgen structure or molecule object.
        """
        backend_module = _return_ext_interface_modules("pymatgen")
        return backend_module._create_pymatgen_obj(self)

    @export_method
    def to_aiida_structuredata(self, label=None):
        """
        Create AiiDA structuredata.

        Returns
        -------
        aiida.orm.StructureData
            AiiDA structure node.
        """
        backend_module = _return_ext_interface_modules("aiida")
        return backend_module._create_structure_node(self)

    def _wrap_position(self, cart_position, scaled_position):
        """Wrap position back into the unit cell."""
        if self.cell is None:
            return cart_position, scaled_position

        if cart_position is not None:
            cart_position = np.array(cart_position)
        if scaled_position is not None:
            scaled_position = np.array(scaled_position)

        if scaled_position is None:
            scaled_position = np.transpose(np.array(self._inverse_cell)).dot(cart_position)
        for direction in range(3):
            if self.pbc[direction]:
                scaled_position[direction] = round(scaled_position[direction], 15) % 1
        cart_position = np.transpose(np.array(self.cell)).dot(scaled_position)
        return tuple(float(p) for p in cart_position), tuple(float(p) for p in scaled_position)

    def _perform_strct_analysis(self, _, method, kwargs):
        return _check_calculated_properties(self, method, kwargs)

    def _perform_strct_manipulation(self, _, method, kwargs):
        new_strct = method(structure=self, **kwargs)
        if isinstance(new_strct, dict):
            return Structure(**new_strct)
        return self
