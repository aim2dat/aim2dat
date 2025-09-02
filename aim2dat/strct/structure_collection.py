"""
Module implementing the StructureCollection class to handle a set of molecular or
crystalline structures.
"""

# Standard library imports
import re
import copy
from warnings import warn
from typing import TYPE_CHECKING, Union, List, Tuple, Iterator

# Third party library imports
import pandas as pd
from ase import Atoms

# Internal library imports
from aim2dat.strct.structure import Structure
from aim2dat.strct.import_export_mixin import ImportExportMixin, import_method, export_method
from aim2dat.strct.io_interface import get_structures_from_file
from aim2dat.io import write_hdf5_structure
from aim2dat.ext_interfaces import _return_ext_interface_modules
import aim2dat.utils.print as utils_pr
from aim2dat.chem_f import transform_dict_to_str


if TYPE_CHECKING:
    import aiida
    import pymatgen


def _process_strct_list(structures, indices):
    # TODO handle keys???
    if isinstance(indices, int):
        return [structures[indices]]
    elif isinstance(indices, slice):
        return structures[indices]
    elif isinstance(indices, (tuple, list)):
        return [structures[idx] for idx in indices]
    else:
        raise TypeError("`indices` must be of type int, tuple, list, or slice.")


class StructureCollection(ImportExportMixin):
    """
    The StructureCollection class is a container for one or multiple atomic structures. It
    implements several ``import_*`` and ``append_*`` functions to add new data to the object.

    Parameters
    ----------
    structures : list
        List of ``Structure`` or dict objects.
    """

    def __init__(self, structures: Union[List[Union[Structure, dict]], None] = None):
        """Initialize object."""
        self._structures = []
        if structures is not None:
            for strct in structures:
                if isinstance(strct, Structure):
                    self.append_structure(strct)
                elif isinstance(strct, dict):
                    self.append(**strct)
                else:
                    raise TypeError(
                        "`structures` needs to be a list containing "
                        "dictionary or Structure objects."
                    )

    @property
    def labels(self) -> List[str]:
        """
        Labels assigened to the structures.
        """
        return [strct.label for strct in self._structures]

    def index(self, label: str):
        """
        Return index of label. If the label is not present, ``None`` is returned.

        Parameters
        ----------
        str
            Label of the structure.
        """
        index = None
        for idx, strct in enumerate(self._structures):
            if strct.label == label:
                index = idx
                break
        return index

    def items(self) -> List[Tuple[str, Structure]]:
        """
        Return a list of label, value tuples.
        """
        return [(strct.label, strct) for strct in self._structures]

    def pop(self, key: Union[str, int]) -> Structure:
        """
        Pop structure.

        Parameters
        ----------
        str
            Key of the structure.
        """
        strct, index, label = self.get_structure(key, True)
        del self._structures[index]
        return strct

    def __str__(self) -> str:
        """
        Represent object as string.
        """

        def create_structure_summaries(start, end):
            strct_list = []
            for strct in self[start:end]:
                cf_str = transform_dict_to_str(strct.chem_formula)
                strct_str = (
                    strct.label
                    + " ".join([""] * (20 - len(strct.label)))
                    + " "
                    + cf_str
                    + " ".join([""] * (20 - len(cf_str)))
                )
                strct_str += (
                    " ["
                    + " ".join(
                        str(val) + " ".join([""] * (6 - len(str(val)))) for val in strct.pbc
                    )
                    + "]"
                )
                strct_list.append(strct_str)
            return strct_list

        output_str = utils_pr._print_title("Structure Collection") + "\n\n"
        output_str += " - Number of structures: " + str(len(self)) + "\n"
        output_str += " - Elements: " + "-".join(self.get_all_elements()) + "\n"
        output_str += "\n"
        output_str += utils_pr._print_subtitle("Structures") + "\n"
        if len(self) < 11:
            strct_list = create_structure_summaries(0, len(self))
        else:
            output_str += utils_pr._print_list("", create_structure_summaries(0, 5))
            output_str += "  ...\n"
            strct_list = create_structure_summaries(len(self) - 5, len(self))
        output_str += utils_pr._print_list("", strct_list)
        output_str += utils_pr._print_hline()
        return output_str

    def __len__(self) -> int:
        """
        Return length of the object.
        """
        return len(self._structures)

    def __getitem__(
        self, key: Union[str, int, tuple, list, slice]
    ) -> Union[Structure, "StructureCollection"]:
        """
        Return structure by key. If a slice, tuple or list of keys is given a
        ``StructureCollection`` object of the subset is returned.

        Parameters
        ----------
        str
            Key of the structure(s).

        Returns
        -------
        Structure or StructureCollection
            structure or structures.
        """
        if isinstance(key, (str, int)):
            return self.get_structure(key)
        elif isinstance(key, (slice, tuple, list)):
            new_sc = StructureCollection()
            for key0 in self._process_key(key):
                new_sc.append_structure(self.get_structure(key0))
            return new_sc
        else:
            raise TypeError("`key` needs to be of type: str, int, slice, tuple or list.")

    def __setitem__(self, key: Union[str, int], value: Union[dict, Structure]):
        """
        Set item by index or label.

        Parameters
        ----------
        str
            Key of the structure.
        """
        # TODO update type hints and doc-strings
        for key0, value0 in zip(*self._process_key(key, value=value)):
            if isinstance(value0, dict):
                value0 = Structure(**value0)
            self._add_structure(key0, value0)

    def __delitem__(self, key: Union[str, int, tuple, list, slice]):
        """
        Delete structure by key.

        Parameters
        ----------
        str
            Key of the structure(s).
        """
        for key0 in self._process_key(key):
            self.pop(key0)

    def __iter__(self) -> Iterator[Structure]:
        """
        Iterate through structures.
        """
        for strct in self._structures:
            yield strct

    def __add__(self, other: "StructureCollection") -> "StructureCollection":
        """
        Add two objects.
        """
        if type(other) is type(self):
            new_sc = StructureCollection()
            for sc_obj in [self, other]:
                for struct in sc_obj:
                    new_sc.append_structure(struct.copy())
            return new_sc
        else:
            raise TypeError("Can only add objects of type StructureCollection.")

    def __deepcopy__(self, memo) -> "StructureCollection":
        """Create a deepcopy of the object."""
        copy = StructureCollection()
        for strct in self:
            copy.append_structure(strct.copy())
        memo[id(self)] = copy
        return copy

    def copy(self) -> "StructureCollection":
        """Return copy of ``StructureCollection`` object."""
        return copy.deepcopy(self)

    def append_structure(self, structure: Structure, label: str = None):
        """
        Append ``Structure`` object to collection. The label of the structure needs to be
        either given via the structures's property or as keyword argument.

        Parameters
        ----------
        structure : Structure
            Structure object.
        label : str (optional)
            String used to identify the structure. Overwrites ``label`` property of the structure.
        """
        self._add_structure(label, structure, raise_label_error=True)

    def append(
        self,
        label: str,
        elements: list,
        positions: list,
        pbc: list,
        cell: list = None,
        is_cartesian: bool = True,
        wrap: bool = False,
        kinds: list = None,
        attributes: dict = None,
        site_attributes: dict = None,
        extras: dict = None,
    ):
        """
        Append structure.

        Parameters
        ----------
        label : str
            String used to identify the structure.
        elements : list
            List of element symbols or their atomic numbers.
        positions : list
            List of the atomic positions, either cartesian or scaled coordinates.
        pbc : list or bool
            Periodic boundary conditions.
        cell : list or np.array
            Nested 3x3 list of the cell vectors.
        is_cartesian : bool (optional)
            Whether the coordinates are cartesian or scaled.
        wrap : bool (optional)
            Wrap atomic positions back into the unit cell.
        kinds : list
            List of kind names (this allows custom kinds like Ni0, Ni1, ...). If None,
            the elements will be used as the kind names.
        attributes : dict
            Attributes stored within the structure object.
        site_attributes : dict
            Site attributes stored within the structure object.
        extras : dict
            Extras stored within the structure object.
        """
        structure = Structure(
            label=label,
            elements=elements,
            positions=positions,
            pbc=pbc,
            cell=cell,
            is_cartesian=is_cartesian,
            wrap=wrap,
            kinds=kinds,
            attributes=attributes,
            site_attributes=site_attributes,
            extras=None,
        )
        self.append_structure(structure)

    def append_from_aiida_structuredata(
        self,
        aiida_node: Union[int, str, "aiida.orm.StructureData"],
        use_uuid: bool = False,
        label: str = None,
    ):
        """
        Append structure from aiida structuredata.

        Parameters
        ----------
        aiida_node : int, str or aiida.orm.StructureData
            Primary key, UUID or AiiDA structure node.
        use_uuid : bool (optional)
            Whether to use the uuid (str) to represent AiiDA nodes instead of the primary key
            (int).
        label : str
            String used to identify the structure. Overwrites ``label`` property of the structure.
        """
        self.append_structure(
            Structure.from_aiida_structuredata(aiida_node, label=label, use_uuid=use_uuid)
        )

    def append_from_ase_atoms(self, label: str, ase_atoms: Atoms, attributes: dict = None):
        """
        Append structure from ase atoms object.

        Parameters
        ----------
        label : str
            String used to identify the structure.
        ase_atoms : ase.Atoms
            ase Atoms object.
        attributes : dict
            Additional information about the structure.
        """
        self.append_structure(
            Structure.from_ase_atoms(ase_atoms, label=label, attributes=attributes)
        )

    def append_from_pymatgen_structure(
        self,
        label: str,
        pymatgen_structure: Union["pymatgen.core.Molecule", "pymatgen.core.Structure"],
        attributes: dict = None,
    ):
        """
        Append structure from pymatgen structure or molecule object.

        Parameters
        ----------
        label : str
            String used to identify the structure.
        pymatgen_structure : pymatgen.core.Structure or pymatgen.core.Molecule
            pymatgen structure or molecule object.
        attributes : dict
            Additional information about the structure.
        """
        self.append_structure(
            Structure.from_pymatgen_structure(
                pymatgen_structure, label=label, attributes=attributes
            )
        )

    def append_from_file(
        self,
        label: str,
        file_path: str,
        attributes: dict = None,
        backend: str = "ase",
        backend_kwargs: dict = None,
    ):
        """
        Append structure from file using the ase read-function.

        Parameters
        ----------
        label : str
            String used to identify the structure.
        file_path : str
            File path.
        attributes : dict
            Additional information about the structure.
        """
        structure = Structure.from_file(
            file_path,
            label=label,
            attributes=attributes,
            backend=backend,
            backend_kwargs=backend_kwargs,
        )
        if isinstance(structure, Structure):
            structure = [structure]
        for strct in structure:
            self.append_structure(strct)

    @import_method
    @classmethod
    def from_file(
        cls,
        file_path: str,
        labels: list = None,
        indices: Union[int, list, tuple, slice] = slice(None),
        raise_error: bool = True,
        backend: str = "internal",
        file_format: str = None,
        backend_kwargs: dict = None,
    ):
        """
        Import from hdf5-file. Calculated extras are not yet supported.

        Parameters
        ------------
        file_path : str
            File path.
        labels : list (optional)
            List of labels with the same length as loaded structures, overwriting the structure
            labels contained in the file.
        indices : int, list, tuple, slice
            Indices of a subset of the structures contained in the file.
        raise_error : bool (optional)
            Whether to raise an error if one of the constraints is not met.
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
        aim2dat.strct.StructureCollection
            Structure collection.
        """
        structures = get_structures_from_file(file_path, backend, file_format, backend_kwargs)
        strct_c = cls()
        for idx, structure in enumerate(_process_strct_list(structures, indices)):
            if labels is not None:
                structure["label"] = labels[idx]
            strct_c._add_structure(
                structure.get("label", f"strct_{idx}"),
                Structure(**structure),
                raise_label_warning=True,
                raise_label_error=raise_error,
            )
        return strct_c

    @import_method
    @classmethod
    def from_pandas_df(
        cls,
        data_frame: pd.DataFrame,
        indices: Union[int, list, tuple, slice] = slice(None),
        structure_column: str = "structure",
        exclude_columns: list = None,
        use_uuid: bool = False,
        raise_error: bool = True,
    ):
        """
        Import from pandas data frame.

        Parameters
        ----------
        data_frame : pd.DataFrame
            Pandas data frame containing at least one column with the AiiDA structure nodes.
        indices : int, list, tuple, slice
            Indices of a subset of the structures contained in the file.
        structure_column : str (optional)
            Column containing AiiDA structure nodes used to determine structural and compositional
            properties. The default value is ``'optimized_structure'``.
        exclude_columns : list (optional)
            Columns of the data frame that are excluded. The default value is ``[]``.
        use_uuid : bool (optional)
            Whether to use the uuid (str) to represent AiiDA nodes instead of the primary key
            (int).
        raise_error : bool (optional)
            Whether to raise an error if one of the constraints is not met.

        Returns
        -------
        aim2dat.strct.StructureCollection
            Structure collection.
        """
        exclude_columns = [] if exclude_columns is None else exclude_columns
        label_unit_pattern = re.compile(r"^([\S\s]+)?\s\(([\S\s]+)\)$")
        structures = []
        strct_c = cls()
        for _, row in data_frame.iterrows():
            structure = row.pop(structure_column)
            if structure is None or structure is pd.NA:
                continue
            if not isinstance(structure, Structure):
                backend_module = _return_ext_interface_modules("aiida")
                structure = Structure.from_aiida_structuredata(structure, use_uuid)
                if structure.label is None and "parent_node" in row:
                    structure.label = backend_module._extract_label_from_aiida_node(
                        row["parent_node"]
                    )
            if "label" in row:
                new_label = row.pop("label")
                if structure.label is None:
                    structure.label = new_label
            if structure.label is None:
                structure.label = f"pandas_{len(strct_c)}"

            for label0, value in row.items():
                if "el_conc" in label0 or "nr_atoms" in label0:
                    continue
                if label0 in exclude_columns:
                    continue
                match = label_unit_pattern.match(label0)
                if match:
                    structure.set_attribute(
                        match.groups()[0],
                        {
                            "value": data_frame.dtypes[label0].type(value),
                            "unit": match.groups()[1],
                        },
                    )
                else:
                    try:
                        structure.set_attribute(label0, data_frame.dtypes[label0].type(value))
                    except TypeError:
                        continue
            structures.append(structure)
        for structure in _process_strct_list(structures, indices):
            strct_c._add_structure(
                structure.label, structure, raise_label_warning=True, raise_label_error=raise_error
            )
        return strct_c

    @import_method
    @classmethod
    def from_aiida_db(
        cls,
        indices: Union[int, list, tuple, slice] = slice(None),
        group_label: str = None,
        use_uuid: bool = False,
        raise_error: bool = True,
    ):
        """
        Import from the AiiDA database.

        Parameters
        ----------
        indices : int, list, tuple, slice
            Indices of a subset of the structures contained in the file.
        group_label : str or list (optional)
            Constrains query to structures that are member of the group(s).
        use_uuid : bool (optional)
            Whether to use the uuid (str) to represent AiiDA nodes instead of the primary key
            (int).
        raise_error : bool (optional)
            Whether to raise an error if one of the constraints is not met.
        """
        backend_module = _return_ext_interface_modules("aiida")
        structure_nodes = []
        if not isinstance(group_label, list):
            group_label = [group_label]
        for gl0 in group_label:
            structure_nodes += backend_module._query_structure_nodes(group_label=gl0)

        strct_c = cls()
        for structure_node in _process_strct_list(structure_nodes, indices):
            structure = Structure.from_aiida_structuredata(structure_node, use_uuid)
            if structure.label is None:
                structure.label = f"aiida_{len(strct_c)}"
            strct_c._add_structure(
                key=structure.label,
                structure=structure,
                raise_label_error=raise_error,
                raise_label_warning=True,
            )
        return strct_c

    @export_method
    def to_file(self, file_path: str, keys=slice(None)):
        """
        Store structures in file. If the file suffix is ``'*.h(df)?5'``, the internal hdf5 format
        is used, otherwise the ase Python package is used as backend.

        Parameters
        ------------
        file_path : str
            File path.
        keys : int, str, list, tuple, slice
            Keys for a subset of structures.
        """
        structures = [self[keys]] if isinstance(keys, (str, int)) else self[keys]
        if file_path.endswith(("h5", "hdf5")):
            write_hdf5_structure(file_path, structures)
        else:
            backend_module = _return_ext_interface_modules("ase_atoms")
            backend_module._write_structure_to_file(file_path, self[keys])

    @export_method
    def to_pandas_df(
        self, keys: Union[int, str, list, tuple, slice] = slice(None), exclude_columns: list = None
    ) -> pd.DataFrame:
        """
        Create a pandas data frame of the object.

        Parameters
        ----------
        keys : int, str, list, tuple, slice
            Keys for a subset of structures.
        exclude_columns : list (optional)
            Columns that are not shown in the pandas data frame.

        Returns
        -------
        pandas.DataFrame
            Pandas data frame.
        """
        structures = [self[keys]] if isinstance(keys, (str, int)) else self[keys]
        backend_module = _return_ext_interface_modules("pandas")
        return backend_module._create_strct_c_pandas_df(structures, exclude_columns)

    @export_method
    def to_aiida_db(
        self,
        keys: Union[int, str, list, tuple, slice] = slice(None),
        group_label: str = None,
        group_description: str = None,
    ):
        """
        Store structures into the AiiDA-database.

        Parameters
        ----------
        keys : int, str, list, tuple, slice
            Keys for a subset of structures.
        group_label : str (optional)
            Label of the AiiDA group.
        group_description : str (optional)
            Description of the AiiDA group.

        Returns
        -------
        list
            List containing dictionary of all structure nodes.
        """
        structures = [self[keys]] if isinstance(keys, (str, int)) else self[keys]
        backend_module = _return_ext_interface_modules("aiida")

        if group_label is not None:
            print(f"Storing data as group `{group_label}` in the AiiDA database.")
            if group_description is None:
                group_description = "Structures from StructureCollection."
        return backend_module._store_data_aiida(group_label, group_description, structures)

    def import_from_aiida_db(
        self, group_label: str = None, use_uuid: bool = False, raise_error: bool = True
    ):
        """
        Import from the AiiDA database.

        Parameters
        ----------
        group_label : str or list (optional)
            Constrains query to structures that are member of the group(s).
        use_uuid : bool (optional)
            Whether to use the uuid (str) to represent AiiDA nodes instead of the primary key
            (int).
        raise_error : bool (optional)
            Whether to raise an error if one of the constraints is not met.
        """
        from warnings import warn

        warn(
            "This function will be removed, please use the `from_aiida_db` class method instead.",
            DeprecationWarning,
            2,
        )
        new_structures = self.from_aiida_db(
            group_label=group_label, use_uuid=use_uuid, raise_error=raise_error
        )
        for strct in new_structures:
            self._add_structure(
                strct.label, strct, raise_label_warning=True, raise_label_error=raise_error
            )

    def import_from_pandas_df(
        self,
        data_frame: pd.DataFrame,
        structure_column: str = "optimized_structure",
        exclude_columns: list = [],
        use_uuid: bool = False,
        raise_error: bool = True,
    ):
        """
        Import from pandas data frame.

        Parameters
        ----------
        data_frame : pd.DataFrame
            Pandas data frame containing at least one column with the AiiDA structure nodes.
        structure_column : str (optional)
            Column containing AiiDA structure nodes used to determine structural and compositional
            properties. The default value is ``'optimized_structure'``.
        exclude_columns : list (optional)
            Columns of the data frame that are excluded. The default value is ``[]``.
        use_uuid : bool (optional)
            Whether to use the uuid (str) to represent AiiDA nodes instead of the primary key
            (int).
        raise_error : bool (optional)
            Whether to raise an error if one of the constraints is not met.
        """
        from warnings import warn

        warn(
            "This function will be removed, please use the `from_pandas_df` class method instead.",
            DeprecationWarning,
            2,
        )
        new_structures = self.from_pandas_df(data_frame, raise_error=raise_error)
        for strct in new_structures:
            self._add_structure(
                strct.label, strct, raise_label_warning=True, raise_label_error=raise_error
            )

    def import_from_hdf5_file(self, file_path: str, raise_error: bool = True):
        """
        Import from hdf5-file. Calculated extras are not yet supported.

        Parameters
        ------------
        file_path : str
            File path.
        raise_error : bool (optional)
            Whether to raise an error if one of the constraints is not met.
        """
        from warnings import warn

        warn(
            "This function will be removed, please use the `from_file` class method instead.",
            DeprecationWarning,
            2,
        )
        new_structures = self.from_file(file_path, raise_error=raise_error)
        for strct in new_structures:
            self._add_structure(
                strct.label, strct, raise_label_warning=True, raise_label_error=raise_error
            )

    def duplicate_structure(self, key: Union[str, int], new_label: str):
        """
        Duplicate structure.

        Parameters
        ----------
        key : str or int
            Key of the structure.
        new_label : str
            Label of the copied structure.
        """
        self.append_structure(self.get_structure(key), label=new_label)

    def get_structure(self, key: Union[str, int], return_index_label: bool = False) -> Structure:
        """
        Get structure by key.

        Parameters
        ----------
        key : str or int
            Key of the structure.

        Returns
        ----------
        Structure
            structure.
        """
        index, label = self._get_index_label(key)

        if index is None or label is None:
            structure = None
        else:
            structure = self._structures[index]
        if return_index_label:
            return structure, index, label
        return structure

    def get_all_structures(self) -> List[Structure]:
        """
        Return a list of all structures.

        Returns
        -------
        list
            List of all structures stored in the object.
        """
        return [self.get_structure(label) for label in self.labels]

    def store_in_hdf5_file(self, file_path: str):
        """
        Store structures in hdf5-file. Calculated extras are not yet supported.

        Parameters
        ------------
        file_path : str
            File path.
        """
        from warnings import warn

        warn(
            "This function will be removed, please use the `to_file` method instead.",
            DeprecationWarning,
            2,
        )
        self.to_file(file_path=file_path)

    def store_in_aiida_db(self, group_label: str = None, group_description: str = None):
        """
        Store structures into the AiiDA-database.

        Parameters
        ----------
        group_label : str (optional)
            Label of the AiiDA group.
        group_description : str (optional)
            Description of the AiiDA group.

        Returns
        -------
        list
            List containing dictionary of all structure nodes.
        """
        from warnings import warn

        warn(
            "This function will be removed, please use the `to_aiida_db` method instead.",
            DeprecationWarning,
            2,
        )
        return self.to_aiida_db(group_label=group_label, group_description=group_description)

    def create_pandas_df(self, exclude_columns: list = None) -> pd.DataFrame:
        """
        Create a pandas data frame of the object.

        Parameters
        ----------
        exclude_columns : list (optional)
            Columns that are not shown in the pandas data frame.

        Returns
        -------
        pandas.DataFrame
            Pandas data frame.
        """
        from warnings import warn

        warn(
            "This function will be removed, please use the `to_pandas_df` class method instead.",
            DeprecationWarning,
            2,
        )
        return self.to_pandas_df(exclude_columns=exclude_columns)

    def get_all_elements(self) -> List[str]:
        """
        Get the element symbols of all structures.

        Returns
        -------
        list
            List of all element symbols .
        """
        all_elements = []
        for strct in self:
            all_elements += strct["elements"]
        return sorted(set(all_elements))

    def get_all_kinds(self) -> list:
        """
        Get the kind strings of all structures.

        Returns
        -------
        list
            List of all kinds.
        """
        all_kinds = []
        for strct in self:
            all_kinds += strct.kinds
        return sorted(set(all_kinds))

    def get_all_attribute_keys(self) -> list:
        """
        Get all attribute keys.

        Returns
        -------
        list
            All attribute keys.
        """
        all_attr_keys = []
        for strct in self:
            all_attr_keys += list(strct.attributes.keys())
        return sorted(set(all_attr_keys))

    def _get_index_label(self, key: Union[str, int]) -> Tuple[Union[int, None], Union[str, None]]:
        if isinstance(key, str):
            return self.index(key), key
        elif isinstance(key, int):
            if key < len(self):
                return key, self._structures[key].label
            else:
                return key, None
        return None, None

    def _process_key(self, key, value=None) -> Union[list, tuple, range]:
        if isinstance(key, (str, int)):
            if value is None:
                return [key]
            return [key], [value]
        elif isinstance(key, (slice, tuple, list)):
            if isinstance(key, slice):
                start = key.start if key.start is not None else 0
                if start < 0:
                    start += len(self)
                stop = key.stop if key.stop is not None else len(self)
                if stop < 0:
                    stop += len(self)
                key = range(start, stop)
            if value is None:
                return key
            return key, value
        else:
            raise TypeError("`key` needs to be of type: str, int, slice, tuple or list.")

    def _add_structure(
        self,
        key: Union[str, int],
        structure: Structure,
        raise_label_warning: bool = False,
        raise_label_error: bool = False,
    ):
        if any(id(structure) == id(strct) for strct in self):
            structure = structure.copy()
        if key is None:
            key = structure.label
        if isinstance(key, str):
            index = self.index(key)
            structure.label = key
            if raise_label_warning and index is not None:
                warn(
                    f"Index '{index}' is being overwritten.",
                    UserWarning,
                    2,
                )
        elif isinstance(key, int):
            if key < len(self):
                index = key
                label = self._structures[key].label
                if raise_label_warning:
                    warn(  # TODO untested
                        f"Label '{label}' is being overwritten.",
                        UserWarning,
                        2,
                    )
                if structure.label is None:
                    structure.label = label
                elif structure.label in self.labels and raise_label_error:
                    raise ValueError(f"Label '{structure.label}' already used.")
            else:
                raise ValueError(f"Index out of range ({key} >= {len(self)}).")
        else:
            raise TypeError("`key` needs to be of type int or str.")

        if structure.label in self.labels and raise_label_error:
            raise ValueError(f"Label '{structure.label}' already used.")
        if index is None:
            self._structures.append(structure)
        else:
            self._structures[index] = structure
