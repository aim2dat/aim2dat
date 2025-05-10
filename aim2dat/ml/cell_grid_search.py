"""Methods to fit the cell parameters of crystalline materials."""

# Standard library imports
import itertools

# Third party library imports
import numpy as np
from scipy.spatial.transform import Rotation

# Internal library imports
from aim2dat.strct import StructureOperations, StructureCollection, Structure
from aim2dat.utils.space_groups import get_lattice_type
from aim2dat.utils.maths import calc_angle


class CellGridSearch:
    """
    Class to fit the cell parameters of an initial structure to a final structure using a
    brute-force grid search approach. The space group is maintained during the fitting process.

    Attributes
    ----------
    length_scaling_factors : list
        Scaling factors for the cell lengths.
    angle_scaling_factors : list
        Scaling factors for the cell angles.
    symprec : float
        Tolerance for spglib and length and angle comparison.
    angle_tolerance : float
        Tolerance parameter for spglib.
    hall_number : int (optional)
        The argument to constrain the space-group-type search only for the Hall symbol
        corresponding to it.
    ffprint_r_max : float
        Cut-off value for the maximum distance between two atoms.
    ffprint_delta_bin : float (optional)
        Bin size to descritize the function.
    ffprint_sigma : float (optional)
        Smearing parameter for the Gaussian function.
    ffprint_use_weights : bool (optional)
        Whether to use importance weights for the element pairs.
    ffprint_distinguish_kinds: bool (optional)
        Whether different kinds should be distinguished e.g. Ni0 and Ni1 would be considered as
        different elements if ``True``.
    target_value : float (optional)
        Target value used to calculate score if a model is set via the ``set_model`` function.
    """

    def __init__(
        self,
        length_scaling_factors=[0.8, 1.0, 1.2],
        angle_scaling_factors=[0.9, 1.0, 1.1],
        symprec=0.005,
        angle_tolerance=-1.0,
        hall_number=0,
        ffprint_r_max=10.0,
        ffprint_delta_bin=0.005,
        ffprint_sigma=0.05,
        ffprint_use_weights=True,
        ffprint_distinguish_kinds=False,
        target_value=0.0,
    ):
        """Construct object."""
        self._strct_ops = StructureOperations(structures=StructureCollection())
        self._transformer = None
        self._model = None
        self._fit_info = None

        self.length_scaling_factors = length_scaling_factors
        self.angle_scaling_factors = angle_scaling_factors
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        self.hall_number = hall_number
        self.ffprint_r_max = ffprint_r_max
        self.ffprint_delta_bin = ffprint_delta_bin
        self.ffprint_sigma = ffprint_sigma
        self.ffprint_use_weights = ffprint_use_weights
        self.ffprint_distinguish_kinds = ffprint_distinguish_kinds
        self.target_value = target_value

    def set_initial_structure(self, structure):
        """
        Set initial crystal structure.

        Parameters
        ----------
        structure : aim2dat.strct.Structure
            Initial structure.
        """
        structure = structure.copy()
        self._strct_ops.structures["initial"] = structure

    def set_model(self, model, function_name="predict", single=False, transformer=None):
        """
        Set scikit-learn model to predict the target value.

        Parameters
        ----------
        model :
            Object that takes structures or features as input to predicts a target value.
        function_name : str (optional)
            Function name to retrieve the property prediction.
        single : bool (optional)
            Whether a single structure/features or a list of structures/features is predicted at
            once.
        transformer : aim2dat.ml.transformers (optional)
            Structure transformer.
        """
        if transformer is not None:
            self._transformer = transformer
        self._model_fct = (function_name, single)
        self._model = model

    def set_target_structure(self, structure):
        """
        Set target crystal structure.

        Parameters
        ----------
        structure : aim2dat.strct.Structure
            Target structure.
        """
        structure = structure.copy()
        self._strct_ops.structures["target"] = structure

    def get_optimized_structure(self):
        """
        Get optimized structure with the lowest score.

        Returns
        -------
        : aim2dat.strct.Structure
            Optimized structure.
        """
        if self._fit_info is None:
            self.fit()
        return self._strct_ops.structures[self._fit_info[0]].copy()

    def return_search_space(self):
        """
        Return list of parameter sets that are varied to fit the initial to the final structure.

        Returns
        -------
        list
            List of parameter sets that are varied.
        """
        search_space = []
        space_group = self._strct_ops["initial"].calc_space_group(
            symprec=self.symprec,
            angle_tolerance=self.angle_tolerance,
            hall_number=self.hall_number,
        )
        lattice_type = get_lattice_type(space_group["space_group"]["number"])
        print(
            "Space group of initial crystal: ",
            space_group["space_group"]["number"],
            "(" + lattice_type + ")",
        )
        cell = self._strct_ops.structures["initial"]["cell"]
        if lattice_type == "triclinic":
            length_combinations = list(itertools.product(self.length_scaling_factors, repeat=3))
            angle_combinations = list(itertools.product(self.length_scaling_factors, repeat=3))
            for l_comb in length_combinations:
                for a_comb in angle_combinations:
                    search_space.append([sf for sf in l_comb + a_comb])
        elif lattice_type == "monoclinic":
            comb = self._check_length_angles(cell, self.symprec, same_length=False, angles=[90.0])
            if len(comb) != 2:
                raise ValueError("Could not detect monoclinic lattice type.")
            angle_idx = [idx0 for idx0 in range(3) if idx0 in comb[0] and idx0 in comb[1]]
            length_combinations = list(itertools.product(self.length_scaling_factors, repeat=3))
            angle_combinations = self.angle_scaling_factors
            for l_comb in length_combinations:
                for a_comb in angle_combinations:
                    param = [l_comb[0], l_comb[1], l_comb[2], 1.0, 1.0, 1.0]
                    param[3 + angle_idx[0]] = a_comb
                    search_space.append(param)
        elif lattice_type == "orthorhombic":
            comb = self._check_length_angles(cell, self.symprec, same_length=False, angles=[90.0])
            if len(comb) != 3:
                raise ValueError("Could not detect orthorhombic lattice type.")
            for scaling_factor_a in self.length_scaling_factors:
                for scaling_factor_b in self.length_scaling_factors:
                    for scaling_factor_c in self.length_scaling_factors:
                        search_space.append(
                            [scaling_factor_a, scaling_factor_b, scaling_factor_c, 1.0, 1.0, 1.0]
                        )
        elif lattice_type == "tetragonal":
            comb = self._check_length_angles(cell, self.symprec, same_length=True, angles=[90.0])
            if len(comb) != 1:
                raise ValueError("Could not detect tetragonal lattice type.")
            length_combinations = list(itertools.product(self.length_scaling_factors, repeat=2))
            for l_comb in length_combinations:
                param = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                for idx0 in range(3):
                    if idx0 in comb[0]:
                        param[idx0] = l_comb[0]
                    else:
                        param[idx0] = l_comb[1]
                search_space.append(param)
        elif lattice_type == "trigonal" or lattice_type == "hexagonal":
            comb = self._check_length_angles(
                cell, self.symprec, same_length=True, angles=[60.0, 120.0]
            )
            if len(comb) != 1:
                raise ValueError("Could not detect trigonal or hexagonal lattice type.")
            for scaling_factor_ab in self.length_scaling_factors:
                for scaling_factor_c in self.length_scaling_factors:
                    params = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
                    for idx0 in range(3):
                        if idx0 in comb[0]:
                            params[idx0] = scaling_factor_ab
                        else:
                            params[idx0] = scaling_factor_c
                    search_space.append(params)
        elif lattice_type == "cubic":
            comb = self._check_length_angles(cell, self.symprec, same_length=True, angles=[90.0])
            if len(comb) != 3:
                raise ValueError("Could not detect cubic lattice type.")
            for sf_abc in self.length_scaling_factors:
                search_space.append([sf_abc, sf_abc, sf_abc, 1.0, 1.0, 1.0])
        return search_space

    def fit(self, search_space=None):
        """
        Fit the initial to the final structure by varying the cell parameters.

        Parameters
        ----------
        search_space : list or None
            Defines the cell parameter variations. If set to ``None`` the parameters are obtained
            via the ``return_search_space``-function.

        Returns
        -------
        max_score : float
            Score of the best match.
        max_params : list
            Parameters that give the best match.
        """
        if search_space is None:
            search_space = self.return_search_space()
        initial_sg = self._strct_ops["initial"].calc_space_group(
            symprec=self.symprec,
            angle_tolerance=self.angle_tolerance,
            hall_number=self.hall_number,
        )["space_group"]["number"]
        initial_strct = self._strct_ops.structures["initial"]

        labels = []
        for idx0, params in enumerate(search_space):
            cell = np.array(initial_strct["cell"])
            for vec_idx, scaling_factor in enumerate(params[:3]):
                cell[vec_idx] *= scaling_factor
            for angle_idx, scaling_factor in enumerate(params[3:]):
                vec_indices = [idx0 for idx0 in range(3) if idx0 != angle_idx]
                rot_v = cell[angle_idx].copy()
                rot_v /= np.linalg.norm(rot_v)
                rot_angle = calc_angle(cell[vec_indices[0]], cell[vec_indices[1]])
                rot_angle *= scaling_factor - 1.0
                rot = Rotation.from_rotvec(rot_angle * rot_v)
                cell = np.dot(rot.as_matrix(), cell.T).T
            self._strct_ops.structures[str(idx0)] = Structure(
                elements=initial_strct.elements,
                positions=initial_strct.scaled_positions,
                pbc=initial_strct.pbc,
                cell=cell,
                is_cartesian=False,
            )
            trial_sg = self._strct_ops[str(idx0)].calc_space_group(
                symprec=self.symprec,
                angle_tolerance=self.angle_tolerance,
                hall_number=self.hall_number,
            )["space_group"]["number"]
            if initial_sg != trial_sg:
                raise ValueError("Space groups don't match!")
            labels.append(str(idx0))

        scores = self._calculate_scores(labels)

        min_score = scores[0]
        min_label = labels[0]
        min_params = search_space[0]
        for label, score, params in zip(labels, scores, search_space):
            if score < min_score:
                min_label = label
                min_score = score
                min_params = params
        self._fit_info = (min_label, min_score, min_params)
        return min_score, min_params

    def return_initial_score(self):
        """
        Return score of the initial structure.

        Returns
        -------
        float
            Score of the initial structure.
        """
        return self._calculate_scores(["initial"])[0]

    def _calculate_scores(self, labels):
        if "target" in self._strct_ops.structures.labels:
            return self._compare_with_target_structure_ffprint(labels)
        elif self._model is not None:
            return self._get_model_predictions(labels)

    def _compare_with_target_structure_ffprint(self, labels):
        scores = []
        comparisons = self._strct_ops.compare_structures_via_ffingerprint(
            labels,
            ["target"] * len(labels),
            r_max=self.ffprint_r_max,
            delta_bin=self.ffprint_delta_bin,
            sigma=self.ffprint_sigma,
            use_weights=self.ffprint_use_weights,
            distinguish_kinds=self.ffprint_distinguish_kinds,
        )

        for label in labels:
            scores.append(comparisons[(label, "target")])
        return scores

    def _get_model_predictions(self, labels):
        predict_fct = getattr(self._model, self._model_fct[0])
        features = [strct for strct in self._strct_ops.structures if strct.label in labels]
        if self._transformer is not None:
            features = self._transformer.transform(features)
        if self._model_fct[1]:
            predictions = [abs(predict_fct(feat) - self.target_value) for feat in features]
        else:
            predictions = np.absolute(predict_fct(features) - self.target_value).tolist()
        return predictions

    @staticmethod
    def _check_length_angles(cell, tol, same_length=False, angles=None):
        cell = np.array(cell)
        found_combintations = []
        for comb in [(0, 1), (0, 2), (1, 2)]:
            comb_found = True
            if (
                same_length
                and abs(np.linalg.norm(cell[comb[0]]) - np.linalg.norm(cell[comb[1]])) > tol
            ):
                comb_found = False
            if angles is not None:
                angle = calc_angle(cell[comb[0]], cell[comb[1]]) * 180.0 / np.pi
                if all(abs(angle - ref) > tol for ref in angles):
                    comb_found = False
            if comb_found:
                found_combintations.append(comb)
        return found_combintations
