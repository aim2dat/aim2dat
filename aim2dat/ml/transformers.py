"""Scikit learn Transformer classes extracting features from crystals or molecules."""

# Standard library imports
import itertools
import math
from abc import ABC, abstractmethod

# Third party library imports
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Internal library imports
from aim2dat.strct import StructureCollection, StructureOperations
from aim2dat.strct.ext_analysis import (
    calc_warren_cowley_order_p,
    calc_prdf,
    calc_interaction_matrix,
    calc_acsf_descriptor,
    calc_soap_descriptor,
    calc_mbtr_descriptor,
)
from aim2dat.strct.structure import _compare_function_args
from aim2dat.chem_f import transform_list_to_dict
from aim2dat.ml.utils import _get_all_elements


class _BaseStructureTransformer(BaseEstimator, TransformerMixin, ABC):
    """Base class for all structure transformers using the StructureCollection object."""

    _supported_params = []

    def __init__(self, n_procs, chunksize, verbose):
        super().__init__()
        self.n_procs = n_procs
        self.chunksize = chunksize
        self.verbose = verbose

        self._precomp_properties = []

    def __sklearn_clone__(self):
        new_obj = super().__sklearn_clone__()
        for precomp_prop in self.precomputed_properties:
            new_obj.add_precomputed_properties(*precomp_prop)
        return new_obj

    @property
    def precomputed_properties(self):
        """
        list: Precomputed properties given as list of tuples consisting of input parameters and
        StructureOperations object.
        """
        return tuple(self._precomp_properties)

    def clear_precomputed_properties(self):
        """
        Clear all precomputed properties.
        """
        self._precomp_properties = []

    def get_feature_names_out(self, input_features=None):
        """
        Get feature names.
        """
        check_is_fitted(self, "_features_out")
        return self._features_out

    def fit(self, X, y=None):
        """
        Fit function that determines the number of features.

        Parameters
        ----------
        X : list or aim2dat.strct.StructureCollection
            List of structures or StructureCollection.
        y : list (optional)
            list of target property.

        Returns
        -------
        self
            Transformer object.
        """
        self._fit(X, y)
        return self

    def transform(self, X):
        """
        Transform structures to features.

        Parameters
        ----------
        X : list or aim2dat.strct.StructureCollection
            List of structures or StructureCollection.

        Returns
        -------
        numpy.array
            Nested array of features.
        """
        # Logic for precomputed
        input_p = {param: getattr(self, param) for param in self._supported_params}
        strct_op = None
        for input_p0, strct_op0 in self._precomp_properties:
            if _compare_function_args(input_p, input_p0):
                strct_op = strct_op0
                break
        label_list, strct_op = self._create_strct_op(X, strct_op)
        features = self._get_features(label_list, strct_op, input_p)
        return features

    def precompute_parameter_space(self, param_grid, X):
        """
        Precompute and store structural properties to be reused later e.g. for a grid search.

        Parameters
        ----------
        param_grid : list or dict
            Dictionary or list of dictionaries of input parameters.
        X : list or aim2dat.strct.StructureCollection
            List of structures or StructureCollection.
        """
        if len(self._supported_params) == 0:
            # Return None if no expensive structural analysis is needed.
            return None

        label_list, strct_op = self._create_strct_op(X)
        if isinstance(param_grid, dict):
            param_grid = [param_grid]
        for param_grid0 in param_grid:
            complete_grid = []
            for param in self._supported_params:
                if param in param_grid0:
                    complete_grid.append(param_grid0[param])
                else:
                    complete_grid.append([getattr(self, param)])
            for comb in itertools.product(*complete_grid):
                input_p = {p_key: p_val for p_key, p_val in zip(self._supported_params, comb)}
                strct_op_copy = strct_op.copy()
                self._get_strct_op_properties(label_list, strct_op_copy, input_p)
                self.add_precomputed_properties(input_p, strct_op_copy)

    def add_precomputed_properties(self, parameters, structure_operations):
        """
        Add precomputed properties.

        Parameters
        ----------
        parameters : dict
            Dictionary of input parameters.
        structure_operations : StructureOperations
            StructureOperations object storing the properties according to the input parameters.
        """
        parameters = self._complete_parameters(parameters)
        is_added = False
        for idx, (stored_param, _) in enumerate(self._precomp_properties):
            if parameters == stored_param:
                self._precomp_properties[idx] = (parameters, structure_operations)
                is_added = True
                break
        if not is_added:
            self._precomp_properties.append((parameters, structure_operations))

    @abstractmethod
    def _fit(self, X, y):
        pass

    def _get_strct_op_properties(self, label_list, strct_op):
        pass

    @abstractmethod
    def _get_features(self, label_list, strct_op):
        pass

    def _complete_parameters(self, parameters):
        comp_params = {}
        for param in self._supported_params:
            value = getattr(self, param)
            if param in parameters:
                value = parameters[param]
            comp_params[param] = value
        return comp_params

    def _create_strct_op(self, X, strct_op=None):
        if strct_op is None:
            strct_op = StructureOperations(structures=StructureCollection())
        strct_op.n_procs = self.n_procs
        strct_op.chunksize = self.chunksize
        strct_op.verbose = self.verbose
        label_list = []
        for strct in X:
            label_list.append(strct["label"])
            if strct["label"] not in strct_op.structures.labels:
                if isinstance(strct, dict):
                    strct_op.structures.append(**strct)
                else:
                    strct_op.structures.append_structure(strct.copy())
        return label_list, strct_op


class _BaseDscribeTransformer(_BaseStructureTransformer):
    def _fit(self, X, y):
        if self.elements is None:
            self.elements_ = _get_all_elements(X, False)
        else:
            self.elements_ = self.elements

    def _get_features(self, label_list, strct_op, input_p):
        check_is_fitted(self, attributes="elements_")
        input_p["elements"] = self.elements_
        descriptors = self._get_strct_op_properties(label_list, strct_op, input_p)
        features = []
        for label in label_list:
            features.append(descriptors[label])
        return np.array(features)


class StructureCompositionTransformer(_BaseStructureTransformer):
    """
    Extract fractional concentrations of elements or kinds.

    Attributes
    ----------
    distinguish_kinds : bool (optional)
        Whether to use kinds instead of elements.
    n_procs : int (optional)
        Number of parallel processes.
    chunksize : int (optional)
        Number of structures handed to each process at once.
    verbose : bool (optional)
        Whether to print a progress bar.
    """

    def __init__(
        self,
        distinguish_kinds=False,
        n_procs=1,
        chunksize=50,
        verbose=True,
    ):
        """Initialize object."""
        super().__init__(n_procs, chunksize, verbose)
        self.distinguish_kinds = distinguish_kinds

    def _fit(self, X, y):
        self.elements_ = _get_all_elements(X, self.distinguish_kinds)
        self._features_out = np.asarray(["c_" + el for el in self.elements_], dtype=object)

    def _get_features(self, label_list, strct_op, _):
        check_is_fitted(self, attributes="elements_")
        features = np.zeros((len(label_list), len(self.elements_)))
        for idx, label in enumerate(label_list):
            structure = strct_op.structures[label]
            el_type = "elements"
            if self.distinguish_kinds:
                el_type = "kinds"
            chem_f = transform_list_to_dict(structure[el_type])
            n_atoms = sum(chem_f.values())
            for el_idx, el in enumerate(self.elements_):
                if el in chem_f:
                    features[idx][el_idx] = chem_f[el] / n_atoms
        return features


class StructureDensityTransformer(_BaseStructureTransformer):
    """
    Extract density of each element or kind.

    Attributes
    ----------
    distinguish_kinds : bool (optional)
        Whether to use kinds instead of elements.
    n_procs : int (optional)
        Number of parallel processes.
    chunksize : int (optional)
        Number of structures handed to each process at once.
    verbose : bool (optional)
        Whether to print a progress bar.
    """

    def __init__(
        self,
        distinguish_kinds=False,
        n_procs=1,
        chunksize=50,
        verbose=True,
    ):
        """Initialize object."""
        super().__init__(n_procs, chunksize, verbose)
        self.distinguish_kinds = distinguish_kinds

    def _fit(self, X, y):
        self.elements_ = _get_all_elements(X, self.distinguish_kinds)
        self._features_out = np.asarray(["density_" + el for el in self.elements_], dtype=object)

    def _get_features(self, label_list, strct_op, _):
        check_is_fitted(self, attributes="elements_")
        features = np.zeros((len(label_list), len(self.elements_)))
        for idx, label in enumerate(label_list):
            structure = strct_op.structures[label]
            if "cell_volume" not in structure:
                raise ValueError(f"'volume' not available for structure '{label}'.")
            el_type = "elements"
            if self.distinguish_kinds:
                el_type = "kinds"
            chem_f = transform_list_to_dict(structure[el_type])
            for el_idx, el in enumerate(self.elements_):
                if el in chem_f:
                    features[idx][el_idx] = chem_f[el] / structure["cell_volume"]
        return features


class StructureCoordinationTransformer(_BaseStructureTransformer):
    """
    Extract coordination numbers and distances between elements or kinds.

    Attributes
    ----------
    r_max : float (optional)
        Cut-off value for the maximum distance between two atoms in angstrom.
    method : str (optional)
        Method used to calculate the coordination environment. The default value is
        ``'minimum_distance'``.
    min_dist_delta : float (optional)
        Tolerance parameter that defines the relative distance from the nearest neighbour atom
        for the ``'minimum_distance'`` method.
    n_nearest_neighbours : int (optional)
        Number of neighbours that are considered coordinated for the ``'n_neighbours'``
        method.
    radius_type : str (optional)
        Type of the atomic radius used for the ``'atomic_radius'`` method (``'covalent'`` is
        used as fallback in the radius for an element is not defined).
    atomic_radius_delta : float (optional)
        Tolerance relative to the sum of the atomic radii for the ``'atomic_radius'`` method.
        If set to ``0.0`` the maximum threshold is defined by the sum of the atomic radii,
        positive (negative) values increase (decrease) the threshold.
    econ_tolerance : float (optional)
        Tolerance parameter for the econ method.
    econ_conv_threshold : float (optional)
        Convergence threshold for the econ method.
    okeeffe_weight_threshold : float (optional)
        Threshold parameter to distinguish indirect and direct neighbour atoms for the
        ``'okeeffe'``.
    feature_types : tuple or str (optional)
        Tuple of features that are extracted. Supported options are: ``'nrs_avg'``,
        ``'nrs_stdev'``, ``'nrs_max'``, ``'nrs_min'``, ``'distance_avg'``,
        ``'distance_stdev'``, ``'distance_max'`` and ``'distance_min'``.
    n_procs : int (optional)
        Number of parallel processes.
    chunksize : int (optional)
        Number of structures handed to each process at once.
    verbose : bool (optional)
        Whether to print a progress bar.
    """

    # TODO include kinds support
    _supported_features = (
        "nrs_avg",
        "nrs_stdev",
        "nrs_max",
        "nrs_min",
        "distance_avg",
        "distance_stdev",
        "distance_max",
        "distance_min",
        "weight_avg",
        "weight_stdev",
        "weight_max",
        "weight_min",
    )
    _supported_params = (
        "r_max",
        "method",
        "min_dist_delta",
        "n_nearest_neighbours",
        "radius_type",
        "atomic_radius_delta",
        "econ_tolerance",
        "econ_conv_threshold",
        "voronoi_weight_type",
        "voronoi_weight_threshold",
    )

    def __init__(
        self,
        r_max=15.0,
        method="minimum_distance",
        min_dist_delta=0.1,
        n_nearest_neighbours=5,
        radius_type="chen_manz",
        atomic_radius_delta=0.0,
        econ_tolerance=0.5,
        econ_conv_threshold=0.001,
        voronoi_weight_type="rel_solid_angle",
        voronoi_weight_threshold=0.5,
        feature_types=(
            "nrs_avg",
            "nrs_stdev",
            "nrs_max",
            "nrs_min",
            "distance_avg",
            "distance_stdev",
            "distance_max",
            "distance_min",
        ),
        n_procs=1,
        chunksize=50,
        verbose=True,
        # precomputed_properties=None,
    ):
        """Initialize object."""
        super().__init__(n_procs, chunksize, verbose)
        self.r_max = r_max
        self.method = method
        self.min_dist_delta = min_dist_delta
        self.n_nearest_neighbours = n_nearest_neighbours
        self.radius_type = radius_type
        self.atomic_radius_delta = atomic_radius_delta
        self.econ_tolerance = econ_tolerance
        self.econ_conv_threshold = econ_conv_threshold
        self.voronoi_weight_type = voronoi_weight_type
        self.voronoi_weight_threshold = voronoi_weight_threshold
        self.feature_types = feature_types
        # self.precomputed_properties = precomputed_properties

    #        if precomputed_properties:
    #            for prec_prop in precomputed_properties:
    #                self.add_precomputed_properties(*prec_prop)

    @property
    def feature_types(self):
        """
        tuple or str : Feature types that are included.
        """
        return self._feature_types if hasattr(self, "_feature_types") else None

    @feature_types.setter
    def feature_types(self, value):
        if isinstance(value, str):
            value = (value,)
        elif isinstance(value, tuple):
            pass
        else:
            raise TypeError("`features` need to be a tuple of strings or a string.")
        for val in value:
            if val not in self._supported_features:
                raise ValueError(f"Feature '{val}' is not supported.")
        self._feature_types = value

    def _fit(self, X, y=None):
        all_elements = sorted(_get_all_elements(X, False))
        self.el_pairs_ = list(itertools.product(all_elements, repeat=2))
        f_labels = []
        for feature in self.feature_types:
            f_labels += [feature + "_" + "-".join(el_pair) for el_pair in self.el_pairs_]
        self._features_out = np.asarray(f_labels, dtype=object)

    def _get_strct_op_properties(self, label_list, strct_op, input_p):
        return strct_op[label_list].calc_coordination(**input_p)

    def _get_features(self, label_list, strct_op, input_p):
        check_is_fitted(self, attributes="el_pairs_")
        features = np.zeros((len(label_list), len(self.feature_types) * len(self.el_pairs_)))
        all_coord_nrs = self._get_strct_op_properties(label_list, strct_op, input_p)
        for idx, label in enumerate(label_list):
            coord_nrs = all_coord_nrs[label]
            for el_pair_idx, el_pair in enumerate(self.el_pairs_):
                for f_idx, f_label in enumerate(self.feature_types):
                    if el_pair in coord_nrs[f_label]:
                        features[idx][f_idx * len(self.el_pairs_) + el_pair_idx] = coord_nrs[
                            f_label
                        ][el_pair]
        return features


class StructureChemOrderTransformer(_BaseStructureTransformer):
    # TODO extend to other coordination methods --> introduce weight as parameter.
    """
    Extract Warren Cowley like order parameters for each element as defined in
    :doi:`10.1103/PhysRevB.96.024104`.

    Attributes
    ----------
    r_max : float (optional)
        Cut-off value for the maximum distance between two atoms in angstrom.
    max_shells : int (optional)
        Number of neighbour shells that are evaluated.
    n_procs : int (optional)
        Number of parallel processes.
    chunksize : int (optional)
        Number of structures handed to each process at once.
    verbose : bool (optional)
        Whether to print a progress bar.
    """

    _supported_params = ("r_max", "max_shells")

    def __init__(
        self,
        r_max=15.0,
        max_shells=3,
        n_procs=1,
        chunksize=50,
        verbose=True,
    ):
        """Initialize object."""
        super().__init__(n_procs, chunksize, verbose)
        self.r_max = r_max
        self.max_shells = max_shells

    def _fit(self, X, y=None):
        self.elements_ = sorted(_get_all_elements(X, False))
        f_labels = []
        for el in self.elements_:
            for shell in range(self.max_shells):
                f_labels.append(el + "_" + str(shell + 1))
        self._features_out = np.asarray(f_labels, dtype=object)

    def _get_strct_op_properties(self, label_list, strct_op, input_p):
        return strct_op[label_list].perform_analysis(
            method=calc_warren_cowley_order_p, kwargs=input_p
        )

    def _get_features(self, label_list, strct_op, input_p):
        check_is_fitted(self, attributes="elements_")
        features = np.zeros((len(label_list), len(self.elements_) * self.max_shells))
        all_order_p = self._get_strct_op_properties(label_list, strct_op, input_p)
        for idx, label in enumerate(label_list):
            order_p = all_order_p[label]
            for el_idx, el in enumerate(self.elements_):
                if el in order_p["order_p"].keys():
                    for shell_idx in range(self.max_shells):
                        features[idx][el_idx * self.max_shells + shell_idx] = order_p["order_p"][
                            el
                        ][shell_idx]
        return features


class StructureFFPrintTransformer(_BaseStructureTransformer):
    """
    Extract the F-fingerprint for each element-pair as defined in
    :doi:`10.1103/PhysRevB.96.024104`.

    Attributes
    ----------
    r_max : float (optional)
        Cut-off value for the maximum distance between two atoms in angstrom.
    delta_bin : float (optional)
            Bin size to descritize the function in angstrom.
    sigma : float (optional)
        Smearing parameter for the Gaussian function.
    distinguish_kinds: bool (optional)
        Whether different kinds should be distinguished e.g. Ni0 and Ni1 would be considered as
        different elements if ``True``.
    add_header : bool
        Add leading entries that describe the weights and composition for the ffprint kernels.
    n_procs : int (optional)
        Number of parallel processes.
    chunksize : int (optional)
        Number of structures handed to each process at once.
    verbose : bool (optional)
        Whether to print a progress bar.
    """

    _supported_params = ("r_max", "delta_bin", "sigma", "distinguish_kinds")

    def __init__(
        self,
        r_max=15.0,
        delta_bin=0.005,
        sigma=10.0,
        distinguish_kinds=False,
        add_header=False,
        use_weights=False,
        n_procs=1,
        chunksize=50,
        verbose=True,
    ):
        """Initialize object."""
        super().__init__(n_procs, chunksize, verbose)
        self.r_max = r_max
        self.delta_bin = delta_bin
        self.sigma = sigma
        self.distinguish_kinds = distinguish_kinds
        self.add_header = add_header
        self.use_weights = use_weights

    def _fit(self, X, y=None):
        self.elements_ = _get_all_elements(X, self.distinguish_kinds)
        self.el_pairs_ = list(itertools.combinations_with_replacement(self.elements_, 2))

    def _get_strct_op_properties(self, label_list, strct_op, input_p):
        return strct_op[label_list].calc_ffingerprint(**input_p)

    def _get_features(self, label_list, strct_op, input_p):
        check_is_fitted(self, attributes="el_pairs_")
        len_ffprint = math.ceil(self.r_max / self.delta_bin)
        header_size = 0
        if self.add_header:
            header_size = len(self.elements_) + 3
        features = np.zeros((len(label_list), header_size + len(self.el_pairs_) * len_ffprint))
        all_fprints = self._get_strct_op_properties(label_list, strct_op, input_p)
        for idx, label in enumerate(label_list):
            fprints = all_fprints[label]
            features[idx][0] = header_size
            features[idx][1] = len_ffprint
            features[idx][2] = len(self.elements_)
            if not self.use_weights:
                features[idx][2] *= -1
            el_dict = (
                transform_list_to_dict(strct_op.structures[label]["kinds"])
                if self.distinguish_kinds
                else transform_list_to_dict(strct_op.structures[label]["elements"])
            )
            for el_idx, el in enumerate(self.elements_):
                features[idx][3 + el_idx] = el_dict.get(el, 0.0)
            for el_pair, fprint in fprints[0]["fingerprints"].items():
                if el_pair in self.el_pairs_:
                    start_idx = header_size + self.el_pairs_.index(el_pair) * len_ffprint
                    end_idx = start_idx + len_ffprint
                    features[idx][start_idx:end_idx] = fprint
        return features


class StructurePRDFTransformer(_BaseStructureTransformer):
    """
    Extract the partial radial distribution function for each element-pair as defined in
    :doi:`10.1103/PhysRevB.89.205118`.

    Attributes
    ----------
    r_max : float (optional)
        Cut-off value for the maximum distance between two atoms in angstrom.
    delta_bin : float (optional)
            Bin size to descritize the function in angstrom.
    distinguish_kinds: bool (optional)
        Whether different kinds should be distinguished e.g. Ni0 and Ni1 would be considered as
        different elements if ``True``.
    n_procs : int (optional)
        Number of parallel processes.
    chunksize : int (optional)
        Number of structures handed to each process at once.
    verbose : bool (optional)
        Whether to print a progress bar.
    """

    _supported_params = ("r_max", "delta_bin", "distinguish_kinds")

    def __init__(
        self,
        r_max=15.0,
        delta_bin=0.005,
        distinguish_kinds=False,
        n_procs=1,
        chunksize=50,
        verbose=True,
    ):
        """Initialize object."""
        super().__init__(n_procs, chunksize, verbose)
        self.r_max = r_max
        self.delta_bin = delta_bin
        self.distinguish_kinds = distinguish_kinds

    def _fit(self, X, y=None):
        all_elements = _get_all_elements(X, self.distinguish_kinds)
        self.el_pairs_ = list(itertools.product(all_elements, repeat=2))

    def _get_strct_op_properties(self, label_list, strct_op, input_p):
        return strct_op[label_list].perform_analysis(method=calc_prdf, kwargs=input_p)

    def _get_features(self, label_list, strct_op, input_p):
        check_is_fitted(self, attributes="el_pairs_")
        len_prdf = math.ceil(self.r_max / self.delta_bin)
        features = np.zeros((len(label_list), len(self.el_pairs_) * len_prdf))
        all_prdfs = self._get_strct_op_properties(label_list, strct_op, input_p)
        for idx, label in enumerate(label_list):
            prdfs = all_prdfs[label][0]
            for el_pair, prdf in prdfs.items():
                if el_pair in self.el_pairs_:
                    start_idx = self.el_pairs_.index(el_pair) * len_prdf
                    end_idx = start_idx + len_prdf
                    features[idx][start_idx:end_idx] = prdf
        return features


class StructureMatrixTransformer(_BaseStructureTransformer):
    """
    Extract features based on interaction matrices as defined in :doi:`10.1002/qua.24917`.
    This transformer class is based on the implementations of the dscribe python package.

    Attributes
    ----------
    matrix_type : str
        Matrix type. Supported options are ``'coulomb'``, ``'ewald_sum'`` or ``'sine'``.
    permutation : str
        Defines the output format. Options are: ``'none'``, ``'sorted_l2'``, ``'eigenspectrum'``
        or ``'random'``.
    sigma : float
        Standar deviation of the Gaussian distributed noise when using ``'random'`` for
        ``permutation``.
    seed : int
        Seed for the random numbers in case ``'random'`` is chosen for the ``permutation``
        attibute.
    sparse : bool
        Whether to return a sparse matrix or a dense 1D array.
    ewald_accuracy : float
        Accuracy threshold for the Ewald sum.
    ewald_w : int
        Weight parameter.
    ewald_r_cut : float or None
        Real space cutoff parameter.
    ewald_g_cut : float or None
        Reciprocal space cutoff parameter.
    ewald_a : float or None
        Parameter controlling the width of the Gaussian functions.
    dscribe_n_jobs : int
        Number of jobs used by dscribe to calculate the interaction matrix.
    dscribe_only_physical_cores : bool
        Whether to only use physicsl cores.
    n_procs : int (optional)
        Number of parallel processes.
    chunksize : int (optional)
        Number of structures handed to each process at once.
    verbose : bool (optional)
        Whether to print a progress bar.
    """

    _supported_params = (
        "matrix_type",
        "n_atoms_max",
        "enforce_real",
        "permutation",
        "sigma",
        "seed",
        "sparse",
        "ewald_accuracy",
        "ewald_w",
        "ewald_r_cut",
        "ewald_g_cut",
        "ewald_a",
        "dscribe_n_jobs",
        "dscribe_only_physical_cores",
    )

    def __init__(
        self,
        matrix_type="coulomb",
        n_atoms_max=None,
        enforce_real=False,
        permutation="eigenspectrum",
        sigma=None,
        seed=None,
        sparse=False,
        ewald_accuracy=1.0e-5,
        ewald_w=1,
        ewald_r_cut=None,
        ewald_g_cut=None,
        ewald_a=None,
        dscribe_n_jobs=1,
        dscribe_only_physical_cores=False,
        n_procs=1,
        chunksize=50,
        verbose=True,
    ):
        """Initialize object."""
        super().__init__(n_procs, chunksize, verbose)
        self.matrix_type = matrix_type
        self.n_atoms_max = n_atoms_max
        self.enforce_real = enforce_real
        self.permutation = permutation
        self.sigma = sigma
        self.seed = seed
        self.sparse = sparse
        self.ewald_accuracy = ewald_accuracy
        self.ewald_w = ewald_w
        self.ewald_r_cut = ewald_r_cut
        self.ewald_g_cut = ewald_g_cut
        self.ewald_a = ewald_a
        self.dscribe_n_jobs = dscribe_n_jobs
        self.dscribe_only_physical_cores = dscribe_only_physical_cores

    def _fit(self, X, y):
        if self.n_atoms_max is None:
            self.n_atoms_max_ = max([len(strct["elements"]) for strct in X])
        else:
            self.n_atoms_max_ = self.n_atoms_max

    def _get_strct_op_properties(self, label_list, strct_op, input_p):
        return strct_op[label_list].perform_analysis(
            method=calc_interaction_matrix, kwargs=input_p
        )

    def _get_features(self, label_list, strct_op, input_p):
        check_is_fitted(self, attributes="n_atoms_max_")
        input_p["n_atoms_max"] = self.n_atoms_max_
        matrices = self._get_strct_op_properties(label_list, strct_op, input_p)
        features = []
        for label in label_list:
            features.append(matrices[label])
        return np.array(features)


class StructureACSFTransformer(_BaseDscribeTransformer):
    """
    Extract ACSF descriptor as defined in :doi:`10.1063/1.3553717`. This transformer class is
    based on the implementations of the dscribe python package.

    Attributes
    ----------
    r_cut : float
        Cutoff value.
    g2_params : np.array
        List of pairs of eta and R_s values for the G^2 functions.
    g3_params : np.array
        List of kappa values for the G^3 functions.
    g4_params : np.array
        List of triplets of eta, zeta and lambda values for G^4 functions.
    g5_params : np.array
        List of triplets of eta, zeta and lambda values for G^5 functions.
    elements : list
        List of atomic numbers or chemical symbols.
    periodic : bool
        Whether to consider periodic boundary conditions.
    sparse : bool
        Whether to return a sparse matrix or a dense array.
    dscribe_n_jobs : int
        Number of jobs used by dscribe to calculate the interaction matrix.
    dscribe_only_physical_cores : bool
        Whether to only use physicsl cores.
    n_procs : int (optional)
        Number of parallel processes.
    chunksize : int (optional)
        Number of structures handed to each process at once.
    verbose : bool (optional)
        Whether to print a progress bar.
    """

    _supported_params = (
        "r_cut",
        "g2_params",
        "g3_params",
        "g4_params",
        "g5_params",
        "elements",
        "periodic",
        "sparse",
        "dscribe_n_jobs",
        "dscribe_only_physical_cores",
    )

    def __init__(
        self,
        r_cut=7.5,
        g2_params=None,
        g3_params=None,
        g4_params=None,
        g5_params=None,
        elements=None,
        periodic=False,
        sparse=False,
        dscribe_n_jobs=1,
        dscribe_only_physical_cores=False,
        n_procs=1,
        chunksize=50,
        verbose=True,
    ):
        """Initialize object."""
        _BaseStructureTransformer.__init__(self, n_procs, chunksize, verbose)
        self.r_cut = r_cut
        self.g2_params = g2_params
        self.g3_params = g3_params
        self.g4_params = g4_params
        self.g5_params = g5_params
        self.elements = elements
        self.periodic = periodic
        self.sparse = sparse
        self.dscribe_n_jobs = dscribe_n_jobs
        self.dscribe_only_physical_cores = dscribe_only_physical_cores

    def _get_strct_op_properties(self, label_list, strct_op, input_p):
        return strct_op[label_list].perform_analysis(method=calc_acsf_descriptor, kwargs=input_p)


class StructureSOAPTransformer(_BaseDscribeTransformer):
    """
    Extract SOAP descriptor as defined in :doi:`10.1103/PhysRevB.87.184115`. This transformer
    class is based on the implementations of the dscribe python package.

    Attributes
    ----------
    r_cut : float
        Cutoff value.
    n_max : int
        The number of radial basis functions.
    l_max : int
        The maximum degree of spherical harmonics.
    sigma : float
        The standard deviation of the gaussians.
    rbf : str
        The radial basis functions to use. Supported options are: ``'gto'`` or ``'polynomial'``.
    weighting : dict
        Contains the options which control the weighting of the atomic density.
    compression : dict
        Feature compression options.
    average : str
        The averaging mode over the centers of interest. Supported options are: ``'off'``,
        ``'inner'`` or ``'outer'``.
    elements : list
        List of atomic numbers or chemical symbols.
    periodic : bool
        Whether to consider periodic boundary conditions.
    sparse : bool
        Whether to return a sparse matrix or a dense array.
    dscribe_n_jobs : int
        Number of jobs used by dscribe to calculate the interaction matrix.
    dscribe_only_physical_cores : bool
        Whether to only use physicsl cores.
    n_procs : int (optional)
        Number of parallel processes.
    chunksize : int (optional)
        Number of structures handed to each process at once.
    verbose : bool (optional)
        Whether to print a progress bar.
    """

    _supported_params = (
        "r_cut",
        "n_max",
        "l_max",
        "sigma",
        "weighting",
        "compression",
        "average",
        "elements",
        "periodic",
        "sparse",
        "dscribe_n_jobs",
        "dscribe_only_physical_cores",
    )

    def __init__(
        self,
        r_cut=7.5,
        n_max=8,
        l_max=6,
        sigma=1.0,
        rbf="gto",
        weighting=None,
        compression={"mode": "off", "species_weighting": None},
        average="off",
        elements=None,
        periodic=False,
        sparse=False,
        dscribe_n_jobs=1,
        dscribe_only_physical_cores=False,
        n_procs=1,
        chunksize=50,
        verbose=True,
    ):
        """Initialize object."""
        _BaseStructureTransformer.__init__(self, n_procs, chunksize, verbose)
        self.r_cut = r_cut
        self.n_max = n_max
        self.l_max = l_max
        self.sigma = sigma
        self.rbf = rbf
        self.weighting = weighting
        self.compression = compression
        self.average = average
        self.elements = elements
        self.periodic = periodic
        self.sparse = sparse
        self.dscribe_n_jobs = dscribe_n_jobs
        self.dscribe_only_physical_cores = dscribe_only_physical_cores

    def _get_strct_op_properties(self, label_list, strct_op, input_p):
        return strct_op[label_list].perform_analysis(method=calc_soap_descriptor, kwargs=input_p)


class StructureMBTRTransformer(_BaseDscribeTransformer):
    """
    Extract MBTR descriptor as defined in :doi:`10.1088/2632-2153/aca005`. This transformer class
    is based on the implementations of the dscribe python package.

    Attributes
    ----------
    geometry : dict
        Setup the geometry function.
    grid : dict
        Setup the discretization grid.
    weighting : dict
        Setup the weighting function and its parameters.
    normalize_gaussians : bool
        Whether to normalize the gaussians to an area of 1.
    normalization : str
        Method for normalizing. Supported options are ``'none'``, ``'l2'``, ``'n_atoms'``,
        ``'valle_oganov'``.
    elements : list
        List of atomic numbers or chemical symbols.
    periodic : bool
        Whether to consider periodic boundary conditions.
    sparse : bool
        Whether to return a sparse matrix or a dense array.
    dscribe_n_jobs : int
        Number of jobs used by dscribe to calculate the interaction matrix.
    dscribe_only_physical_cores : bool
        Whether to only use physicsl cores.
    n_procs : int (optional)
        Number of parallel processes.
    chunksize : int (optional)
        Number of structures handed to each process at once.
    verbose : bool (optional)
        Whether to print a progress bar.
    """

    _supported_params = (
        "geometry",
        "grid",
        "weighting",
        "normalize_gaussians",
        "normalization",
        "elements",
        "periodic",
        "sparse",
        "dscribe_n_jobs",
        "dscribe_only_physical_cores",
    )

    def __init__(
        self,
        geometry={"function": "inverse_distance"},
        grid={"min": 0, "max": 1, "n": 100, "sigma": 0.1},
        weighting={"function": "exp", "scale": 1.0, "threshold": 1e-3},
        normalize_gaussians=True,
        normalization="l2",
        elements=None,
        periodic=False,
        sparse=False,
        dscribe_n_jobs=1,
        dscribe_only_physical_cores=False,
        n_procs=1,
        chunksize=50,
        verbose=True,
    ):
        """Initialize object."""
        _BaseStructureTransformer.__init__(self, n_procs, chunksize, verbose)
        self.geometry = geometry
        self.grid = grid
        self.weighting = weighting
        self.normalize_gaussians = normalize_gaussians
        self.normalization = normalization
        self.elements = elements
        self.periodic = periodic
        self.sparse = sparse
        self.dscribe_n_jobs = dscribe_n_jobs
        self.dscribe_only_physical_cores = dscribe_only_physical_cores

    def _get_strct_op_properties(self, label_list, strct_op, input_p):
        return strct_op[label_list].perform_analysis(method=calc_mbtr_descriptor, kwargs=input_p)
