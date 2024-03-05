"""Implements the StructureComparison and StructuresOperations classes to
analyze a collection of structures.
"""

# Standard library imports
import itertools
from typing import List, Tuple, Union
import copy
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from collections.abc import Callable


# Third party library imports
import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

# Internal library imports
from aim2dat.strct import StructureCollection
from aim2dat.strct import Structure
from aim2dat.strct.mixin import AnalysisMixin, ManipulationMixin
from aim2dat.strct.strct_comparison import (
    _compare_structures_ffprint,
    _compare_structures_direct_comp,
    _compare_structures_comp_sym,
)
from aim2dat.strct.stability import _calculate_stabilities
from aim2dat.strct.strct_coordination import _coordination_compare_sites
from aim2dat.strct.strct_prdf import _ffingerprint_compare_sites
from aim2dat.strct.strct_space_groups import determine_space_group
from aim2dat.strct.strct_prdf import calculate_ffingerprint


def _create_index_combinations(confined, strct_c_len):
    if confined is None:
        return list(itertools.combinations(range(strct_c_len), 2))
    min_idx = confined[0]
    max_idx = confined[1]
    if min_idx is None or min_idx < 0:
        min_idx = 0
    if max_idx is None or max_idx > strct_c_len:
        max_idx = strct_c_len
    pairs = []
    for idx0 in range(strct_c_len):
        for idx1 in range(min_idx, max_idx):
            if idx0 != idx1 and (idx1, idx0) not in pairs:
                pairs.append((idx0, idx1))
    return pairs


def structure_wrapper(structure, method, kwargs, check_stored):
    """Parallelize structure analysis and manipulation methods via this wrapper function."""
    if check_stored and not structure.store_calculated_properties:
        return None
    if getattr(method, "_is_analysis_method", False) or getattr(
        method, "_manipulates_structure", False
    ):
        return structure, method(structure, **copy.deepcopy(kwargs))
    else:
        return structure, getattr(structure, method.__name__)(**copy.deepcopy(kwargs))


def compare_structures(structures, compare_function, comp_kwargs, threshold):
    """Parallelize structure comparison methods via this wrapper function."""
    if threshold is None:
        return compare_function(structures[0], structures[1], **comp_kwargs)
    return compare_function(structures[0], structures[1], **comp_kwargs) < threshold


class StructureOperations(AnalysisMixin, ManipulationMixin):
    """Serves as a wrapper to make the methods defined on a single
    Structure object accessible for a StructureCollection.
    """

    def __init__(
        self,
        structures: StructureCollection,
        append_to_coll: bool = True,
        output_format: str = "dict",
        n_procs: int = 1,
        chunksize: int = 50,
        verbose: bool = True,
    ):
        """Initialize object."""
        self.structures = structures
        self.append_to_coll = append_to_coll
        self.output_format = output_format
        self.n_procs = n_procs
        self.chunksize = chunksize
        self.verbose = verbose

    def __deepcopy__(self, memo) -> "StructureOperations":
        """Create a deepcopy of the object."""
        copy = StructureOperations(
            structures=self.structures.copy(),
            append_to_coll=self.append_to_coll,
            output_format=self.output_format,
            n_procs=self.n_procs,
            chunksize=self.chunksize,
            verbose=self.verbose,
        )
        memo[id(self)] = copy
        return copy

    def __getitem__(
        self, key: Union[str, int, tuple, list, slice]
    ) -> Union[Structure, "StructureOperations"]:
        """
        Return structure by key. If a slice, tuple or list of keys is given a
        ``StructureOperations`` object of the subset is returned.

        Parameters
        ----------
        str
            Key of the structure(s).

        Returns
        -------
        Structure or StructureOperations
            structure or ``StructureOperations`` object of the structures.
        """
        if isinstance(key, (str, int)):
            return self.structures.get_structure(key)
        elif isinstance(key, (slice, tuple, list)):
            new_sc = StructureCollection()
            if isinstance(key, slice):
                start = key.start if key.start is not None else 0
                if start < 0:
                    start += len(self)
                stop = key.stop if key.stop is not None else len(self)
                if stop < 0:
                    stop += len(self)
                key = range(start, stop)
            for key0 in key:
                new_sc.append_structure(self.get_structure(key0))
            return StructureOperations(new_sc)
        else:
            raise TypeError("key needs to be of type: str, int, slice, tuple or list.")

    def copy(self) -> "StructureOperations":
        """Return copy of ``StructureCollection`` object."""
        return copy.deepcopy(self)

    @property
    def structures(self) -> StructureCollection:
        """Return the internal ``StructureCollection`` object."""
        return self._structures

    @structures.setter
    def structures(self, value: StructureCollection):
        if isinstance(value, StructureCollection):
            self._structures = value
        else:
            raise TypeError("`structures` needs to be of type `StructureCollection`.")

    @property
    def verbose(self) -> bool:
        """
        bool: Print progress bar.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("`verbose` needs to be of type bool.")
        self._verbose = value

    @property
    def n_procs(self) -> int:
        """int: Number of parallel processes."""
        return self._n_procs

    @n_procs.setter
    def n_procs(self, value: int):
        if not isinstance(value, int):
            raise TypeError("`n_procs` needs to be of type int.")
        if value < 1:
            raise TypeError("`n_procs` needs to be larger than 0.")
        self._n_procs = value

    @property
    def chunksize(self) -> int:
        """int: Number of tasks handed to each process at once."""
        return self._chunksize

    @chunksize.setter
    def chunksize(self, value: int):
        if not isinstance(value, int):
            raise TypeError("`chunksize` needs to be of type int.")
        if value < 1:
            raise TypeError("`chunksize` needs to be larger than 0.")
        self._chunksize = value

    @property
    def append_to_coll(self) -> bool:
        """
        bool: Whether to append manipulated structures also to the internal ``StructureCollection``
        object.
        """
        return self._append_to_coll

    @append_to_coll.setter
    def append_to_coll(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("`append_to_coll` needs to be of type bool.")
        self._append_to_coll = value

    @property
    def supported_output_formats(self) -> List[str]:
        """Return the supported output formats."""
        return ["dict", "DataFrame"]

    @property
    def output_format(self) -> str:
        """
        str: Specify the output format of calculation methods. Supported options are ``'dict'``
        and ``'DataFrame'``.
        """
        return self._output_format

    @output_format.setter
    def output_format(self, value: str):
        if not isinstance(value, str):
            raise TypeError("`output_format` needs to be of type str.")
        if value not in self.supported_output_formats:
            raise ValueError(
                f"`output_format` '{value}' is not supported. It has to be "
                f"one of the following options: {self.supported_output_formats}"
            )
        self._output_format = value

    def calculate_stabilities(
        self, unit: str = "eV", exclude_keys: list = []
    ) -> Tuple[list, list]:
        """
        Calculate the formation energies and stabilities of all structures.

        The stabilities are only valid for binary systems.

        Parameters
        ----------
        unit : str (optional)
            Energy unit.
        exclude_keys : list
            List of keys of structures that are not included in the detection of the convex hull.
            This means that the stability of these structures may have a negative sign.

        Returns
        -------
        formation_energies : list
            List of the formation energies of all structures.
        stabilities : list
            List of the stabilities of all structures.
        """
        return _calculate_stabilities(self.structures, output_unit=unit, exclude_keys=exclude_keys)

    def compare_structures_via_ffingerprint(
        self,
        key1: Union[str, int],
        key2: Union[str, int],
        r_max: float = 15.0,
        delta_bin: float = 0.005,
        sigma: float = 0.05,
        use_weights: bool = True,
        use_legacy_smearing: bool = False,
        distinguish_kinds: bool = False,
    ) -> float:
        """
        Calculate similarity of two structures.

        The cosine-distance is used to compare the two structures.

        Parameters
        ----------
        key1 : str or int
            Index or label of the first structure.
        key2 : str or int
            Index or label of the second structure.
        r_max : float (optional)
            Cut-off value for the maximum distance between two atoms in angstrom.
        delta_bin : float (optional)
            Bin size to descritize the function in angstrom.
        sigma : float (optional)
            Smearing parameter for the Gaussian function.
        use_weights : bool (optional)
            Whether to use importance weights for the element pairs.
        use_legacy_smearing : bool
            Use the depreciated smearing method.
        distinguish_kinds: bool (optional)
            Whether different kinds should be distinguished e.g. Ni0 and Ni1 would be considered as
            different elements if ``True``.

        Returns
        -------
        distance : float
            Measure for the similarity of the two structures.
        """
        return _compare_structures_ffprint(
            self.structures[key1],
            self.structures[key2],
            r_max,
            delta_bin,
            sigma,
            use_weights,
            use_legacy_smearing,
            distinguish_kinds,
        )

    def compare_structures_via_comp_sym(
        self,
        key1: Union[str, int],
        key2: Union[str, int],
        symprec: float = 0.005,
        angle_tolerance: float = -1.0,
        hall_number: int = 0,
    ) -> bool:
        """
        Compare two structures merely based on the composition and space group.

        Parameters
        ----------
        key1 : str or int
            Index or label of the first structure.
        key2 : str or int
            Index or label of the second structure.
        symprec : float (optional)
            Tolerance parameter for spglib.
        angle_tolerance : float (optional)
            Tolerance parameter for spglib.
        hall_number : int (optional)
            The argument to constrain the space-group-type search only for the Hall symbol
            corresponding to it.

        Returns
        -------
        bool
            Returns ``True`` if the structures match and otherwise ``False``.
        """
        return _compare_structures_comp_sym(
            self.structures[key1], self.structures[key2], symprec, angle_tolerance, hall_number
        )

    def compare_structures_via_direct_comp(
        self,
        key1: Union[str, int],
        key2: Union[str, int],
        symprec: float = 0.005,
        angle_tolerance: float = -1.0,
        hall_number: float = 0,
        no_idealize: bool = False,
        length_threshold: float = 0.08,
        angle_threshold: float = 0.03,
        position_threshold: float = 0.025,
        distinguish_kinds: bool = False,
    ) -> bool:
        """Compare structures by comparing lattice vectors, angles and scaled positions."""
        return _compare_structures_direct_comp(
            self.structures[key1],
            self.structures[key2],
            symprec,
            angle_tolerance,
            hall_number,
            no_idealize,
            length_threshold,
            angle_threshold,
            position_threshold,
            distinguish_kinds,
        )

    def find_duplicates_via_ffingerprint(
        self,
        confined: list = None,
        remove_structures: bool = False,
        threshold: float = 0.001,
        r_max: float = 15.0,
        delta_bin: float = 0.005,
        sigma: float = 0.05,
        use_weights: bool = True,
        use_legacy_smearing: bool = False,
        distinguish_kinds: bool = False,
    ) -> List[Tuple[str]]:
        """
        Find duplicate structures using the FFingerprint method.

        Parameters
        ----------
        confined : list or None (optional)
            Confine comparison to a subset of the structure collection by giving a minimum and
            maximum index.
        remove_structures : bool (optional)
            Whether to remove the duplicate structures.
        threshold : float (optional)
            Threshold of the FFingerprint to detect duplicate structures.
        r_max : float (optional)
            Maximum distance between two atoms used to construct the super cell.
        delta_bin : float (optional)
            Bin size to discretize the function in angstrom.
        sigma : float (optional)
            Smearing parameter for the Gaussian function.
        use_weights : bool (optional)
            Whether to use importance weights for the element pairs.
        use_legacy_smearing : bool
            Use the depreciated smearing method.
        distinguish_kinds: bool (optional)
            Whether different kinds should be distinguished e.g. Ni0 and Ni1 would be considered as
            different elements if ``True``.

        Returns
        -------
        list
            List of tuples containing the indices of the found duplicate pairs.
        """
        comp_kwargs = {
            "r_max": r_max,
            "delta_bin": delta_bin,
            "sigma": sigma,
            "use_legacy_smearing": use_legacy_smearing,
            "distinguish_kinds": distinguish_kinds,
        }
        self._perform_operation(
            list(range(len(self.structures))), calculate_ffingerprint, comp_kwargs, True
        )
        comp_kwargs["use_weights"] = use_weights
        return self._find_duplicate_structures(
            _compare_structures_ffprint,
            comp_kwargs,
            threshold,
            confined,
            remove_structures,
        )

    def find_duplicates_via_comp_sym(
        self,
        confined: list = None,
        remove_structures: bool = False,
        symprec: float = 0.005,
        angle_tolerance: float = -1.0,
        hall_number: int = 0,
    ) -> List[Tuple[str]]:
        """
        Find duplicate structures coimparing the composition and space group.

        Parameters
        ----------
        confined : list or None (optional)
            Confine comparison to a subset of the structure collection by giving a minimum and
            maximum index.
        remove_structures : bool (optional)
            Whether to remove the duplicate structures.
        symprec : float (optional)
            Tolerance parameter for spglib.
        angle_tolerance : float (optional)
            Tolerance parameter for spglib.
        hall_number : int (optional)
            The argument to constrain the space-group-type search only for the Hall symbol
            corresponding to it.

        Returns
        -------
        list
            List of tuples containing the indices of the found duplicate pairs.
        """
        comp_kwargs = {
            "symprec": symprec,
            "angle_tolerance": angle_tolerance,
            "hall_number": hall_number,
            "return_standardized_structure": True,
        }
        self._perform_operation(
            list(range(len(self.structures))), determine_space_group, comp_kwargs, True
        )
        return self._find_duplicate_structures(
            _compare_structures_comp_sym, comp_kwargs, None, confined, remove_structures
        )

    def find_duplicates_via_direct_comp(
        self,
        confined: list = None,
        remove_structures: bool = False,
        symprec: float = 0.005,
        angle_tolerance: float = -1.0,
        hall_number: int = 0,
        no_idealize: bool = False,
        length_threshold: float = 0.08,
        angle_threshold: float = 0.03,
        position_threshold: float = 0.025,
        distinguish_kinds: bool = False,
    ) -> List[Tuple[str]]:
        """
        Find duplicate structures comparing directly the lattice parameters and positions of the
         standardized structures..

        Parameters
        ----------
        confined : list or None (optional)
            Confine comparison to a subset of the structure collection by giving a minimum and
            maximum index.
        remove_structures : bool (optional)
            Whether to remove the duplicate structures.
        symprec : float (optional)
            Tolerance parameter for spglib.
        angle_tolerance : float (optional)
            Tolerance parameter for spglib.
        hall_number : int (optional)
            The argument to constrain the space-group-type search only for the Hall symbol
            corresponding to it.

        Returns
        -------
        list
            List of tuples containing the indices of the found duplicate pairs.
        """
        comp_kwargs = {
            "symprec": symprec,
            "angle_tolerance": angle_tolerance,
            "hall_number": hall_number,
            "no_idealize": no_idealize,
            "return_standardized_structure": True,
        }
        self._perform_operation(
            list(range(len(self.structures))), determine_space_group, comp_kwargs, True
        )
        comp_kwargs["length_threshold"] = length_threshold
        comp_kwargs["angle_threshold"] = angle_threshold
        comp_kwargs["position_threshold"] = position_threshold
        comp_kwargs["distinguish_kinds"] = distinguish_kinds
        return self._find_duplicate_structures(
            _compare_structures_direct_comp, comp_kwargs, None, confined, remove_structures
        )

    def _find_duplicate_structures(
        self, compare_function, comp_kwargs, threshold, confined, remove_structures
    ):
        duplicate_pairs = []
        structures2del = []
        index_comb = _create_index_combinations(confined, len(self.structures))

        if self.n_procs > 1:
            strct_comb = [
                (self.structures[idx0], self.structures[idx1]) for idx0, idx1 in index_comb
            ]
            if self.verbose:
                output_list = process_map(
                    partial(
                        compare_structures,
                        compare_function=compare_function,
                        comp_kwargs=comp_kwargs,
                        threshold=threshold,
                    ),
                    strct_comb,
                    max_workers=self.n_procs,
                    chunksize=self.chunksize,
                    desc="find_duplicates",
                )
            else:
                exc = ProcessPoolExecutor(max_workers=self.n_procs)
                output_list = exc.map(
                    partial(
                        compare_structures,
                        compare_function=compare_function,
                        comp_kwargs=comp_kwargs,
                        threshold=threshold,
                    ),
                    strct_comb,
                    chunksize=self.chunksize,
                )
                exc.shutdown()
            for strct_pair, is_dup in zip(strct_comb, output_list):
                if strct_pair[1].label in structures2del:
                    continue
                if is_dup:
                    structures2del.append(strct_pair[1].label)
                    duplicate_pairs.append((strct_pair[1].label, strct_pair[0].label))
        else:
            if self.verbose:
                index_comb = tqdm(index_comb, desc="find_duplicates")
            for idx_pair in index_comb:
                structure1 = self.structures[idx_pair[0]]
                structure2 = self.structures[idx_pair[1]]
                if structure2.label in structures2del:
                    continue
                if compare_structures(
                    [structure1, structure2], compare_function, comp_kwargs, threshold
                ):
                    structures2del.append(structure2.label)
                    duplicate_pairs.append((structure2.label, structure1.label))
        if remove_structures:
            for label in set(structures2del):
                self.structures.pop(label)
        return duplicate_pairs

    def _compare_sites(
        self,
        key1,
        key2,
        site_index1,
        site_index2,
        calc_function,
        calc_f_kwargs,
        compare_function,
        compare_f_kwargs,
    ):
        site_indices = (site_index1, site_index2)
        structures = []
        calc_props = []
        for key, site_index in zip([key1, key2], site_indices):
            structure = self.structures.get_structure(key, False)
            if site_index > len(structure["elements"]):
                raise ValueError(f"Site index out of range for structure '{key}'.")
            calc_props.append(calc_function(key, **calc_f_kwargs))
            structures.append(structure)
        return compare_function(structures, site_indices, calc_props, **compare_f_kwargs)

    def _find_equivalent_sites(
        self, key, comp_function, comp_kwargs, threshold, distinguish_kinds
    ):
        structure = self.structures.get_structure(key, False)
        comp_type = "elements"
        if distinguish_kinds:
            comp_type = "kinds"
        eq_sites = {}
        chem_f = {}
        for site_idx, specie in enumerate(structure[comp_type]):
            is_not_eq = True
            for eq_site_indices in eq_sites.values():
                if specie != structure[comp_type][eq_site_indices[0]]:
                    continue
                comp_value = comp_function(key, key, site_idx, eq_site_indices[0], **comp_kwargs)
                if not isinstance(comp_value, bool):
                    comp_value = comp_value < threshold
                if comp_value:
                    eq_site_indices.append(site_idx)
                    is_not_eq = False
                    break
            if is_not_eq:
                if specie in chem_f:
                    chem_f[specie] += 1
                else:
                    chem_f[specie] = 1
                eq_sites[specie + str(chem_f[specie])] = [site_idx]
        return eq_sites

    # Comparison

    # Problem: eq_sites in one structure depend on compare sites
    # (which is also available for multiple structures)
    # Check how to split. Ideally, eq_sites in Structure and
    # compare_sites in SturtucreComparison
    def compare_sites_via_coordination(
        self,
        key1: Union[str, int],
        key2: Union[str, int],
        site_index1: int,
        site_index2: int,
        r_max: float = 10.0,
        cn_method: str = "minimum_distance",
        min_dist_delta: float = 0.1,
        n_nearest_neighbours: int = 5,
        econ_tolerance: float = 0.5,
        econ_conv_threshold: float = 0.001,
        voronoi_weight_type: float = "rel_solid_angle",
        voronoi_weight_threshold: float = 0.5,
        okeeffe_weight_threshold: float = 0.5,
        distinguish_kinds: bool = False,
        threshold: float = 1e-2,
    ):
        """
        Compare two atomic sites based on their coordination and the distances to their neighbour
        atoms.

        Parameters
        ----------
        key1 : str or int
            Index or label of the first structure.
        key2 : str or int
            Index or label of the second structure.
        site_index1 : int
            Index of the site.
        site_index2 : int
            Index of the site.
        r_max : float (optional)
            Cut-off value for the maximum distance between two atoms in angstrom.
        cn_method : str (optional)
            Method used to calculate the coordination environment.
        min_dist_delta : float (optional)
            Tolerance parameter that defines the relative distance from the nearest neighbour atom
            for the ``'minimum_distance'`` method.
        n_nearest_neighbours : int (optional)
            Number of neighbours that are considered coordinated for the ``'n_neighbours'``
            method.
        econ_tolerance : float (optional)
            Tolerance parameter for the econ method.
        econ_conv_threshold : float (optional)
            Convergence threshold for the econ method.
        voronoi_weight_type : str (optional)
            Weight type of the Voronoi facets. Supported options are ``'covalent_atomic_radius'``,
            ``'area'`` and ``'solid_angle'``. The prefix ``'rel_'`` specifies that the relative
            weights with respect to the maximum value of the polyhedron are calculated.
        voronoi_weight_threshold : float (optional)
            Weight threshold to consider a neighbouring atom coordinated.
        okeeffe_weight_threshold : float (optional)
            Threshold parameter to distinguish indirect and direct neighbour atoms for the
            ``'okeeffe'``.
        distinguish_kinds: bool (optional)
            Whether different kinds should be distinguished e.g. Ni0 and Ni1 would be considered as
            different elements if ``True``.
        threshold : float (optional)
            Threshold to consider two sites equivalent.

        Returns
        -------
        bool
            Whether the two sites are equivalent or not.
        """
        calc_f_kwargs = {
            "r_max": r_max,
            "method": cn_method,
            "min_dist_delta": min_dist_delta,
            "econ_tolerance": econ_tolerance,
            "econ_conv_threshold": econ_conv_threshold,
            "voronoi_weight_type": voronoi_weight_type,
            "voronoi_weight_threshold": voronoi_weight_threshold,
            "okeeffe_weight_threshold": okeeffe_weight_threshold,
        }
        compare_f_kwargs = {
            "distinguish_kinds": distinguish_kinds,
            "threshold": threshold,
        }
        return self._compare_sites(
            key1,
            key2,
            site_index1,
            site_index2,
            self.calculate_coordination,
            calc_f_kwargs,
            _coordination_compare_sites,
            compare_f_kwargs,
        )

    def compare_sites_via_ffingerprint(
        self,
        key1: Union[str, int],
        key2: Union[str, int],
        site_index1: int,
        site_index2: int,
        r_max: float = 15.0,
        delta_bin: float = 0.005,
        sigma: float = 10.0,
        use_weights: bool = True,
        use_legacy_smearing: bool = False,
        distinguish_kinds: bool = False,
    ):
        """
        Calculate similarity of two atom sites.

        The cosine-distance is used to compare the two structures.

        Parameters
        ----------
        key1 : str or int
            Index or label of the first structure.
        key2 : str or int
            Index or label of the second structure.
        site_index1 : int
            Index of the site.
        site_index2 : int
            Index of the site.
        r_max : float (optional)
            Cut-off value for the maximum distance between two atoms in angstrom.
        delta_bin : float (optional)
            Bin size to descritize the function in angstrom.
        sigma : float (optional)
            Smearing parameter for the Gaussian function.
        use_weights : bool (optional)
            Whether to use importance weights for the element pairs.
        use_legacy_smearing : bool
            Use the depreciated smearing method.
        distinguish_kinds: bool (optional)
            Whether different kinds should be distinguished e.g. Ni0 and Ni1 would be considered as
            different elements if ``True``.

        Returns
        -------
        distance : float
            Measure for the similarity of the two sites.
        """
        calc_f_kwargs = {
            "r_max": r_max,
            "delta_bin": delta_bin,
            "sigma": sigma,
            "use_legacy_smearing": use_legacy_smearing,
            "distinguish_kinds": distinguish_kinds,
        }
        compare_f_kwargs = {
            "distinguish_kinds": distinguish_kinds,
            "use_weights": use_weights,
        }
        return self._compare_sites(
            key1,
            key2,
            site_index1,
            site_index2,
            self.calculate_ffingerprint,
            calc_f_kwargs,
            _ffingerprint_compare_sites,
            compare_f_kwargs,
        )

    def find_eq_sites_via_coordination(
        self,
        key: Union[str, int],
        r_max: float = 10.0,
        cn_method: str = "minimum_distance",
        min_dist_delta: float = 0.1,
        n_nearest_neighbours: int = 5,
        econ_tolerance: float = 0.5,
        econ_conv_threshold: float = 0.001,
        voronoi_weight_type: float = "rel_solid_angle",
        voronoi_weight_threshold: float = 0.5,
        okeeffe_weight_threshold: float = 0.5,
        distinguish_kinds: bool = False,
        threshold: float = 1e-2,
    ):
        """
        Find equivalent sites by comparing the coordination of each site and its distance to the
        neighbour atoms.

        Parameters
        ----------
        key : str or int
            Index or label of the structure.
        r_max : float (optional)
            Cut-off value for the maximum distance between two atoms in angstrom.
        cn_method : str (optional)
            Method used to calculate the coordination environment.
        min_dist_delta : float (optional)
            Tolerance parameter that defines the relative distance from the nearest neighbour atom
            for the ``'minimum_distance'`` method.
        n_nearest_neighbours : int (optional)
            Number of neighbours that are considered coordinated for the ``'n_neighbours'``
            method.
        econ_tolerance : float (optional)
            Tolerance parameter for the econ method.
        econ_conv_threshold : float (optional)
            Convergence threshold for the econ method.
        okeeffe_weight_threshold : float (optional)
            Threshold parameter to distinguish indirect and direct neighbour atoms for the
            ``'okeeffe'``.
        distinguish_kinds: bool (optional)
            Whether different kinds should be distinguished e.g. Ni0 and Ni1 would be considered as
            different elements if ``True``.
        threshold : float (optional)
            Threshold to consider two sites equivalent.

        Returns
        --------
        dict :
            Dictionary grouping equivalent sites.
        """
        coord_kwargs = {
            "r_max": r_max,
            "cn_method": cn_method,
            "min_dist_delta": min_dist_delta,
            "econ_tolerance": econ_tolerance,
            "econ_conv_threshold": econ_conv_threshold,
            "voronoi_weight_type": voronoi_weight_type,
            "voronoi_weight_threshold": voronoi_weight_threshold,
            "okeeffe_weight_threshold": okeeffe_weight_threshold,
            "threshold": threshold,
        }
        return self._find_equivalent_sites(
            key, self.compare_sites_via_coordination, coord_kwargs, None, distinguish_kinds
        )

    def find_eq_sites_via_ffingerprint(
        self,
        key: Union[str, int],
        r_max: float = 20.0,
        delta_bin: float = 0.005,
        sigma: float = 0.05,
        use_weights: bool = True,
        use_legacy_smearing: bool = False,
        distinguish_kinds: bool = False,
        threshold: float = 1e-3,
    ):
        """
        Find equivalent sites by comparing the F-Fingerprint of each site.

        Parameters
        ----------
        key : str or int
            Index or label of the structure.
        r_max : float (optional)
            Cut-off value for the maximum distance between two atoms in angstrom.
        delta_bin : float (optional)
            Bin size to descritize the function in angstrom.
        sigma : float (optional)
            Smearing parameter for the Gaussian function.
        use_weights : bool
            Whether to use importance weights for the element pairs.
        use_legacy_smearing : bool
            Use the depreciated smearing method.
        distinguish_kinds: bool (optional)
            Whether different kinds should be distinguished e.g. Ni0 and Ni1 would be considered as
            different elements if ``True``.
        threshold : float (optional)
            Threshold to consider two sites equivalent.

        Returns
        --------
        dict :
            Dictionary grouping equivalent sites.
        """
        ffingerprint_kwargs = {
            "r_max": r_max,
            "delta_bin": delta_bin,
            "sigma": sigma,
            "use_weights": use_weights,
            "use_legacy_smearing": use_legacy_smearing,
            "distinguish_kinds": distinguish_kinds,
        }
        return self._find_equivalent_sites(
            key,
            self.compare_sites_via_ffingerprint,
            ffingerprint_kwargs,
            threshold,
            distinguish_kinds,
        )

    def perform_analysis(
        self, key: Union[str, int, tuple, list], method: Callable, kwargs: dict = {}
    ):
        """
        Perform structure analaysis using an external method.

        Parameters
        ----------
        key : str, int, tuple or list
            Only used in the ``StructureOperations`` class. Specifies the key or list/tuple of
            keys of the underlying ``StructureCollection`` object.
        method : function
            Analysis function.
        kwargs : dict
            Arguments to be passed to the function.

        Returns
        ------
        output
            Output of the analysis.
        """
        if not getattr(method, "_is_analysis_method", False):
            raise TypeError("Function is not a structure analysis method.")
        return self._perform_strct_analysis(key, method, kwargs)

    def perform_manipulation(
        self, key: Union[str, int, tuple, list], method: Callable, kwargs: dict = {}
    ):
        """
        Perform structure manipulation using an external method.

        Parameters
        ----------
        key : str, int, tuple or list
            Only used in the ``StructureOperations`` class. Specifies the key or list/tuple of
            keys of the underlying ``StructureCollection`` object.
        method : function
            Function which manipulates the structure(s).
        kwargs : dict
            Arguments to be passed to the function.

        Returns
        ------
        aim2dat.strct.Structure or
        aim2dat.strct.StructureCollection
            Manipulated structure(s).
        """
        if not getattr(method, "_manipulates_structure", False):
            raise TypeError("Function is not a structure analysis method.")
        return self._perform_strct_manipulation(key, method, kwargs)

    def _perform_strct_manipulation(self, key, method, kwargs):
        output = self._perform_operation(key, method, kwargs, False)
        output_strct_c = StructureCollection()
        for label, structure in output.items():
            if self.append_to_coll:
                self._structures[structure.label] = structure
            output_strct_c.append_structure(structure)
        if isinstance(key, (str, int)):
            return list(output.values())[0]
        return output_strct_c

    def _perform_strct_analysis(self, key, method, kwargs):
        output = self._perform_operation(key, method, kwargs, False)
        if isinstance(key, (str, int)):
            return list(output.values())[0]
        elif self.output_format == "dict":
            return output
        elif self.output_format == "DataFrame":
            return pd.DataFrame(output.values(), index=output.keys(), columns=[method])

    def _perform_operation(self, key, method, kwargs, check_stored):
        if key is None:
            raise TypeError("`key` needs to be set to perform structural analysis tasks.")
        keys = key if isinstance(key, (list, tuple)) else [key]
        structure_list = [self.structures.get_structure(key0) for key0 in keys]
        output = {}
        if self.n_procs > 1 and len(keys) > 1:
            if self.verbose:
                output_list = process_map(
                    partial(
                        structure_wrapper,
                        method=method,
                        kwargs=kwargs,
                        check_stored=check_stored,
                    ),
                    structure_list,
                    max_workers=self.n_procs,
                    chunksize=self.chunksize,
                    desc=method.__name__,
                )
            else:
                exc = ProcessPoolExecutor(max_workers=self.n_procs)
                output_list = exc.map(
                    partial(
                        structure_wrapper,
                        method=method,
                        kwargs=kwargs,
                        check_stored=check_stored,
                    ),
                    structure_list,
                    chunksize=self.chunksize,
                )
                exc.shutdown()
            for strct, output0 in output_list:
                self.structures[strct.label] = strct
                output[strct.label] = output0
        else:
            if self.verbose and len(keys) > 1:
                structure_list = tqdm(structure_list, desc=method.__name__)
            for structure in structure_list:
                _, output0 = structure_wrapper(structure, method, kwargs, check_stored)
                output[structure.label] = output0
        return output
