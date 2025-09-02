"""Module implementing a Structure class."""

# Standard library imports
import copy
from typing import TYPE_CHECKING, Any, List, Union

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.ext_interfaces import _return_ext_interface_modules
from aim2dat.strct.named_structures import molecules, functional_groups
from aim2dat.strct.io_interface import get_structures_from_file
from aim2dat.io import write_zeo_file
from aim2dat.strct.validation import (
    _structure_validate_cell,
    _structure_validate_elements,
    _structure_validate_positions,
)
from aim2dat.strct.analysis_mixin import AnalysisMixin
from aim2dat.strct.manipulation_mixin import ManipulationMixin
from aim2dat.strct.import_export_mixin import ImportExportMixin, import_method, export_method
from aim2dat.chem_f import transform_dict_to_str, transform_list_to_dict
import aim2dat.utils.print as utils_pr
from aim2dat.utils.maths import calc_angle
from aim2dat.elements import get_atomic_number
from aim2dat.utils.dict_tools import dict_retrieve_parameter


if TYPE_CHECKING:
    import aiida
    import ase
    import openmm
    import pymatgen


def _compare_function_args(args1, args2):
    """Compare function arguments to check if a property needs to be recalculated."""
    if len(args1) != len(args2):
        return False

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


def _check_calculated_properties(structure, func, func_args, mapping):
    property_name = "_".join(func.__name__.split("_")[1:])
    output = None
    if structure.store_calculated_properties and property_name in structure._function_args:
        if _compare_function_args(structure._function_args[property_name], func_args):
            output = structure.extras[property_name]
    if output is None:
        output = func(structure, **func_args)
    if mapping is not None:
        for key, attr_tree in mapping.items():
            structure.set_attribute(key, dict_retrieve_parameter(output, attr_tree))
    if structure.store_calculated_properties:
        structure._extras[property_name] = output
        structure._function_args[property_name] = func_args
    return output


def _update_label_attributes_extras(strct_dict, label, attributes, site_attributes, extras):
    # TODO handle deepcopy.
    if label is not None:
        strct_dict["label"] = label
    if attributes is not None:
        strct_dict.setdefault("attributes", {}).update(attributes)
    if site_attributes is not None:
        strct_dict.setdefault("site_attributes", {}).update(site_attributes)
    if extras is not None:
        strct_dict.setdefault("extras", {}).update(extras)


class Structure(AnalysisMixin, ManipulationMixin, ImportExportMixin):
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
        site_attributes: dict = None,
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
        self.set_positions(positions, is_cartesian=is_cartesian, wrap=wrap)

        self.label = label
        self.attributes = attributes
        self.site_attributes = site_attributes
        self.store_calculated_properties = store_calculated_properties

        self._extras = {} if extras is None else extras
        self._function_args = {} if function_args is None else function_args

    def __str__(self):
        """Represent object as string."""

        def _parse_vector(vector):
            vector = ["{0:.4f}".format(val) for val in vector]
            return "[" + " ".join([" ".join([""] * (9 - len(val))) + val for val in vector]) + "]"

        output_str = utils_pr._print_title(f"Structure: {self.label}") + "\n\n"
        output_str += " Formula: " + transform_dict_to_str(self.chem_formula) + "\n"
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
        return key in keys_to_check

    def __getitem__(self, key: str):
        """Return structure attribute by key or list of keys."""
        if isinstance(key, list):
            try:
                return {k: getattr(self, k) for k in key}
            except AttributeError:
                raise KeyError(f"Key {key} is not present.")
        elif isinstance(key, str):
            try:
                return getattr(self, key)
            except AttributeError:
                raise KeyError(f"Key {key} is not present.")

    def __setitem__(self, key: str, value: Any):
        """Set structure attribute."""
        valid_keys = [
            attr
            for attr, v in vars(self.__class__).items()
            if isinstance(v, property) and v.fset is not None
        ]
        if key in valid_keys:
            setattr(self, key, value)
        else:
            raise ValueError(
                f"'{key}' is not a supported property, valid options are {valid_keys}."
            )

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
        """Return attribute keys to create the structure."""
        return [
            "label",
            "elements",
            "positions",
            "pbc",
            "cell",
            "kinds",
            "site_attributes",
            "attributes",
            "extras",
            "function_args",
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
        self._chem_formula = transform_list_to_dict(elements)
        self._numbers = tuple(get_atomic_number(el) for el in elements)
        self.reset_calculated_properties()

    @property
    def chem_formula(self) -> dict:
        """
        Return chemical formula.
        """
        return self._chem_formula

    @property
    def numbers(self) -> tuple:
        """Return the atomic numbers of the structure."""
        return self._numbers

    @property
    def positions(self) -> tuple:
        """tuple: Return the cartesian positions of the structure."""
        return getattr(self, "_positions", None)

    @positions.setter
    def positions(self, value: Union[list, tuple]):
        self.set_positions(
            value,
            is_cartesian=True,
        )

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
        self.reset_calculated_properties()

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
            if self.scaled_positions is not None:
                self.set_positions(self.scaled_positions, is_cartesian=False)
            elif self.positions is not None:
                self.set_positions(self.positions, is_cartesian=True)

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
        """tuple: Kinds of the structure."""
        return self._kinds

    @kinds.setter
    def kinds(self, value: Union[tuple, list]):
        if value is None:
            value = [None] * len(self.elements)
        if not isinstance(value, (list, tuple)):
            raise TypeError("`kinds` must be a list or tuple.")
        if len(value) != len(self.elements):
            raise ValueError("`kinds` must have the same length as `elements`.")
        self._kind_dict = _create_index_dict(value)
        self._kinds = tuple(value)
        self.reset_calculated_properties()

    @property
    def site_attributes(self) -> Union[dict, None]:
        """
        dict :
            Copy of ``site_attributes``: Dictionary containing the label of a site attribute as key
            and a tuple/list of values having the same length as the ``Structure`` object itself
            (number of sites) containing site specific properties or attributes (e.g. charges,
            magnetic moments, forces, ...).
        """
        return copy.deepcopy(self._site_attributes)

    @site_attributes.setter
    def site_attributes(self, value: dict):
        if value is None:
            value = {}
        self._site_attributes = {}
        for key, val in value.items():
            self.set_site_attribute(key, val)

    @property
    def function_args(self) -> dict:
        """Return function arguments for stored extras."""
        return copy.deepcopy(self._function_args)

    @property
    def attributes(self) -> dict:
        """Return attributes."""
        return self._attributes

    @attributes.setter
    def attributes(self, attributes: dict):
        if attributes is None:
            self._attributes = {}
        elif isinstance(attributes, dict):
            self._attributes = attributes
        else:
            raise TypeError("`attributes` must be a dictionary.")

    @property
    def extras(self) -> dict:
        """
        Return extras.
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

    def reset_calculated_properties(self):
        """
        Reset all previously calculated properties that are stored within the
        ``Structure`` object.
        """
        self._extras = {}
        self._function_args = {}

    def iter_sites(
        self,
        get_kind: bool = False,
        get_cart_pos: bool = False,
        get_scaled_pos: bool = False,
        wrap: bool = False,
        site_attributes: Union[str, list] = None,
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
        site_attributes : list (optional)
            Include site attributes defined by their label.

        Yields
        ------
        str or tuple
            Either element symbol or tuple containing the element symbol, kind string,
            cartesian position, scaled position or specified site attributes.
        """
        if site_attributes is None:
            site_attributes = []
        elif isinstance(site_attributes, str):
            site_attributes = [site_attributes]
        for idx, el in enumerate(self.elements):
            output = [el]
            if get_kind:
                output.append(self.kinds[idx])

            if get_cart_pos or get_scaled_pos:
                pos_cart, pos_scaled = self._get_position(idx, wrap)
                if get_cart_pos:
                    output.append(pos_cart)
                if get_scaled_pos:
                    output.append(pos_scaled)
            for site_attr in site_attributes:
                output.append(self._site_attributes[site_attr][idx])

            if len(output) == 1:
                yield el
            else:
                yield tuple(output)

    def set_positions(
        self, positions: Union[list, tuple], is_cartesian: bool = True, wrap: bool = False
    ):
        """
        Set postions of all sites.

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
        self.reset_calculated_properties()

    def get_position(self, index: int, cartesian: bool = True, wrap: bool = False):
        """
        Return position of one site.

        Parameters
        ----------
        index : int
            Site index.
        cartesian : bool (optional)
            Get cartesian position. If set to ``False``, the scaled position is returned.
        wrap : bool (optional)
            Wrap atomic position into the unit cell.
        """
        pos_cart, pos_scaled = self._get_position(index=index, wrap=wrap)
        if cartesian:
            return pos_cart
        return pos_scaled

    def get_positions(self, cartesian: bool = True, wrap: bool = False):
        """
        Return positions of all sites.

        Parameters
        ----------
        cartesian : bool (optional)
            Get cartesian positions. If set to ``False``, scaled positions are returned.
        wrap : bool (optional)
            Wrap atomic positions into the unit cell.
        """
        return tuple(
            self.get_position(index=idx, cartesian=cartesian, wrap=wrap)
            for idx in range(len(self))
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

    def set_site_attribute(self, key: str, values: Union[list, tuple]):
        """
        Set site attribute.

        Parameters
        ----------
        key : str
            Key of the site attribute.
        values :
            Values of the attribute, need to have the same length as the ``Structure`` object
            itself (number of sites).
        """
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"Value of site property `{key}` must be a list or tuple.")
        if len(values) != len(self.elements):
            raise ValueError(
                f"Value of site property `{key}` must have the same length as `elements`."
            )
        self._site_attributes[key] = tuple(values)

    @classmethod
    def list_named_structures(cls) -> list:
        """List of all named structures which can be generated via the ``from_str`` classmethod."""
        return list(molecules.keys()) + list(functional_groups.keys())

    @import_method
    @classmethod
    def from_str(
        cls,
        name: str,
        attributes: dict = None,
        site_attributes: dict = None,
        extras: dict = None,
        label: str = None,
    ) -> "Structure":
        """
        Generate a pre-defined structure by name.

        Parameters
        ----------
        name : str
            Name of the structure.
        attributes : dict
            Attributes stored within the structure object(s).
        site_attributes : dict
            Site attributes stored within the structure object(s).
        extras : dict
            Extras stored within the structure object(s).
        label : str
            Label used internally to store the structure in the object.

        """
        for pred_dict in [molecules, functional_groups]:
            if name in pred_dict:
                strct_dict = pred_dict[name]
                if isinstance(strct_dict, str):
                    strct_dict = pred_dict[strct_dict]
                _update_label_attributes_extras(
                    strct_dict, label, attributes, site_attributes, extras
                )
                return cls(**strct_dict)
        raise ValueError(
            f"Name '{name}' is not supported. Valid options are: {cls.list_named_structures()}."
        )

    @import_method
    @classmethod
    def from_file(
        cls,
        file_path: str,
        attributes: dict = None,
        site_attributes: dict = None,
        extras: dict = None,
        label: str = None,
        index: int = 0,
        backend: str = "ase",
        file_format: str = None,
        backend_kwargs: dict = None,
    ) -> "Structure":
        """
        Get structure from file using the ase read-function.

        Parameters
        ----------
        file_path : str
            File path.
        attributes : dict
            Attributes stored within the structure object(s).
        site_attributes : dict
            Site attributes stored within the structure object(s).
        extras : dict
            Extras stored within the structure object(s).
        label : str
            Label used internally to store the structure in the object.
        index : int
            Index of the structure in case the file contains several structures.
        backend : str (optional)
            Backend to be used to parse the structure file. Supported options are ``'ase'``
            and ``'internal'``.
        file_format : str or None (optional)
            File format of the backend. For ``'ase'``, please refer to the documentation of the
            package for a complete list. For ``'internal'``, the format translates from
            ``io.{module}.read_structure`` to ``'{module}'`` or from
            ``{module}.read_{specification}_structure`` to ``'module-specification'``. If set to
            ``None`` the corresponding function is searched based on the file name and suffix.
        backend_kwargs : dict (optional)
            Arguments passed to the backend function.

        Returns
        -------
        aim2dat.strct.Structure
            Structure.
        """
        strct_dict = get_structures_from_file(file_path, backend, file_format, backend_kwargs)[
            index
        ]
        _update_label_attributes_extras(strct_dict, label, attributes, site_attributes, extras)
        return cls(**strct_dict)

    @import_method
    @classmethod
    def from_ase_atoms(
        cls,
        ase_atoms: "ase.Atoms",
        attributes: dict = None,
        site_attributes: dict = None,
        extras: dict = None,
        label: str = None,
    ) -> "Structure":
        """
        Get structure from ase atoms object. Attributes and site attributes
        are obtained from the ``info`` and ``arrays`` properties, respectively.

        Parameters
        ----------
        ase_atoms : ase.Atoms
            ase Atoms object.
        attributes : dict
            Attributes stored within the structure object.
        site_attributes : dict
            Site attributes stored within the structure object.
        extras : dict
            Extras stored within the structure object.
        label : str
            Label used internally to store the structure in the object.

        Returns
        -------
        aim2dat.strct.Structure
            Structure.
        """
        backend_module = _return_ext_interface_modules("ase_atoms")
        strct_dict = backend_module._extract_structure_from_atoms(ase_atoms)
        _update_label_attributes_extras(strct_dict, label, attributes, site_attributes, extras)
        return cls(**strct_dict)

    @import_method
    @classmethod
    def from_pymatgen_structure(
        cls,
        pymatgen_structure: Union["pymatgen.core.Molecule", "pymatgen.core.Structure"],
        attributes: dict = None,
        site_attributes: dict = None,
        extras: dict = None,
        label: str = None,
    ) -> "Structure":
        """
        Get structure from pymatgen structure or molecule object.

        Parameters
        ----------
        pymatgen_structure : pymatgen.core.Structure or pymatgen.core.Molecule
            pymatgen structure or molecule object.
        attributes : dict
            Attributes stored within the structure object.
        site_attributes : dict
            Site attributes stored within the structure object.
        extras : dict
            Extras stored within the structure object.
        label : str
            Label used internally to store the structure in the object.

        Returns
        -------
        aim2dat.strct.Structure
            Structure.
        """
        backend_module = _return_ext_interface_modules("pymatgen")
        strct_dict = backend_module._extract_structure_from_pymatgen(pymatgen_structure)
        _update_label_attributes_extras(strct_dict, label, attributes, site_attributes, extras)
        return cls(**strct_dict)

    @import_method
    @classmethod
    def from_aiida_structuredata(
        cls,
        structure_node: Union[int, str, "aiida.orm.StructureData"],
        use_uuid: bool = False,
        label: str = None,
    ) -> "Structure":
        """
        Get structure from AiiDA structure node.

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
        aim2dat.strct.Structure
            Structure.
        """
        backend_module = _return_ext_interface_modules("aiida")
        structure_dict = backend_module._extract_dict_from_aiida_structure_node(
            structure_node, use_uuid
        )
        if label is not None:
            structure_dict["label"] = label
        return cls(**structure_dict)

    @import_method
    @classmethod
    def from_openmm_simulation(
        cls,
        simulation,
        attributes: dict = None,
        site_attributes: dict = None,
        extras: dict = None,
        label: str = None,
    ):
        """
        Get structure from openmm simulation using the latest context state.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            openmm simulation.
        attributes : dict
            Attributes stored within the structure object.
        site_attributes : dict
            Site attributes stored within the structure object.
        extras : dict
            Extras stored within the structure object.
        label : str
            Label used internally to store the structure in the object.

        Returns
        -------
        aim2dat.strct.Structure
            Structure.
        """
        backend_module = _return_ext_interface_modules("openmm")
        strct_dict = backend_module._extract_structure_from_simulation(simulation)
        _update_label_attributes_extras(strct_dict, label, attributes, site_attributes, extras)
        return cls(**strct_dict)

    @export_method
    def to_dict(
        self,
        cartesian: bool = True,
        wrap: bool = False,
        include_calculated_properties: bool = False,
    ) -> dict:
        """
        Export structure to python dictionary.

        Parameters
        ----------
        cartesian : bool (optional)
            Whether cartesian or scaled coordinates are returned.
        wrap : bool (optional)
            Whether the coordinates are wrapped back into the unit cell.
        include_calculated_properties : bool (optional)
            Include ``extras`` and ``function_args`` in the dictionary as well.

        Returns
        -------
        dict
            Dictionary representing the structure. The ``Structure`` object can be retrieved via
            ``Structure(**dict)``.
        """
        # TODO add test:
        calc_prop_keys = ["extras", "function_args"]
        strct_dict = {}
        for key in self.keys():
            if (not include_calculated_properties and key in calc_prop_keys) or key == "positions":
                continue
            strct_dict[key] = (
                copy.deepcopy(getattr(self, key)) if key == "attributes" else getattr(self, key)
            )
        strct_dict["positions"] = self.get_positions(cartesian=cartesian, wrap=wrap)
        if not cartesian:
            strct_dict["is_cartesian"] = False
        return strct_dict

    @export_method
    def to_file(self, file_path: str) -> None:
        """
        Export structure to file using the ase interface or certain file formats for Zeo++.
        """
        if file_path.endswith((".cssr", ".v1", ".cuc")):
            write_zeo_file(file_path, self)
        else:
            backend_module = _return_ext_interface_modules("ase_atoms")
            backend_module._write_structure_to_file(file_path, self)

    @export_method
    def to_ase_atoms(self) -> "ase.Atoms":
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

    @export_method
    def to_openmm_simulation(
        self,
        potential: "openmm.app.Simulation.ForceField",
        integrator: "openmm.Integrator",
        potential_kwargs=None,
        bonds=None,
        device="cpu",
    ) -> "openmm.app.Simulation":
        """
        Create openmm simulation object.

        Parameters
        ----------
        potential
            openmm potential or force field.
        integrator
            openmm integrator.
        potential_kwargs : dict
            Additional keyword argurments for the ``create_system`` function of the
            potential/force field.
        bonds : list
            List of tuples of two site indices that share a chemical bond.
        device : str
            Device/platform used for the simulation.

        Returns
        -------
        openmm.app.Simulation
            openmm simulation object.
        """
        backend_module = _return_ext_interface_modules("openmm")
        return backend_module._create_simulation(
            self, potential, integrator, potential_kwargs, bonds, device
        )

    def _get_position(self, index: int, wrap: bool):
        """Get cartesian and scaled position and (optionally) wrap them back into the unit cell."""
        cart_pos = self.positions[index]
        scaled_pos = None if self.scaled_positions is None else self.scaled_positions[index]

        if wrap and scaled_pos is not None:
            cart_pos = np.array(cart_pos)
            scaled_pos = np.array(scaled_pos)
            for direction in range(3):
                if self.pbc[direction]:
                    scaled_pos[direction] = round(scaled_pos[direction], 15) % 1
            cart_pos = np.transpose(np.array(self.cell)).dot(scaled_pos)
            cart_pos = tuple(float(p) for p in cart_pos)
            scaled_pos = tuple(float(p) for p in scaled_pos)
        return cart_pos, scaled_pos

    def _perform_strct_analysis(self, method, kwargs, mapping=None):
        return _check_calculated_properties(self, method, kwargs, mapping)

    def _perform_strct_manipulation(self, method, kwargs):
        new_strct = method(structure=self, **kwargs)
        if isinstance(new_strct, Structure):
            return new_strct
        elif isinstance(new_strct, dict):
            return Structure(**new_strct)
        return self
