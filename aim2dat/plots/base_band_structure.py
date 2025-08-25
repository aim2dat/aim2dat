"""Base classes for band structure plots."""

# Standard library imports
import re
import itertools

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.strct.analysis.geometry import _calc_reciprocal_cell
from aim2dat.utils.maths import calc_angle


class _BaseBandStructure:
    def __init__(self):
        # The first imported band structure will set the segments.
        # We store in _path_specifications a list of dictionaries:
        # {kpoint_st: first k-point of the segment,
        #  label_st: tuple of the label position and the label string,
        #  kpoint_e: last k-point of the segment,
        #  label_e: tuple of the label position and the label string,
        #  pos_st: position on the x-axis
        #  pos_e: position on the x-axis
        #  scale: how much the segment is scaled based on the reciprocal cell (if given)
        #  }
        # The reference cell gives the right relative length of the path-segments.
        # If the cell is set a scaling parameter will be added to each path-segment.
        #
        self._path_specifications = None
        self._reciprocal_cell = None

    def set_reference_cell(self, reference_cell):
        """
        Set reference cell.

        Just taking the relative positions of the high-symmetry k-points will distort the path if
        if the unit cell vectors and consequently also the basis vectors of the reciprocal cell
        have different lengths.
        This can be accounted for by setting the reference cell and thereby scaling the path
        segments accordingly. The reference cell needs to be set before importing band structures.

        Parameters
        ----------
        reference_cell : list or np.array
            Nested 3x3 list of the cell vectors.
        """
        if isinstance(reference_cell, (list, np.ndarray)):
            reference_cell = np.array(reference_cell).reshape((3, 3))
        else:
            raise TypeError("'cell' must be a list or numpy array.")
        self._reciprocal_cell = np.array(_calc_reciprocal_cell(reference_cell))

    def import_band_structure(
        self,
        data_label,
        kpoints,
        bands,
        path_labels=None,
        occupations=None,
        unit_y="eV",
        align_to_vbm=False,
    ):
        """
        Import a data set.

        Parameters
        ----------
        data_label : str
            Internal label used to plot and compare multiple data sets.
        kpoints : list
            List of k-points.
        bands : list
            List of eigenvalues (nkpoints x neigenvalues or nspins x nkpoints x neigenvalues).
        path_labels : list (optional)
            List of path labels.
        occupations : list (optional)
            List of occupations.
        unit_y : str (optional)
            Unit to be used in the y-label.
        align_to_vbm : bool
            Whether the bands should be aligned to the valence band maximum. Works only
            if occupation numbers are given. A band is defined as unoccupied if the number of
            electrons is below 0.5.
        """
        if data_label in self._data:
            if "band_structure" in self._data[data_label]:
                raise ValueError(f"Data label {data_label} contains already band structure data.")
        else:
            self._data[data_label] = {}

        bands_data = self._process_band_structure(
            kpoints, bands, path_labels, occupations, align_to_vbm
        )
        if unit_y:
            bands_data["unit_y"] = unit_y
        if data_label not in self._data:
            self._data[data_label] = {}
        self._data[data_label]["band_structure"] = bands_data

    def import_from_aiida_bandsdata(self, data_label, bandsdata_node, align_to_vbm=False):
        """
        Read band structure from an aiida bandsdata node.

        Parameters
        ----------
        data_label : str
            Internal label of the data set.
        bandsdata_node : aiida.bands, int or str
            Data node containing the band structure or its primary key or uuid.
        align_to_vbm : bool
            Whether the bands should be aligned to the valence band maximum. Works only
            if occupation numbers are given. A band is defined as unoccupied if the number of
            electrons is below 0.5. The default value is ``False``.
        """
        from aim2dat.ext_interfaces.aiida import _load_data_node

        bandsdata_node = _load_data_node(bandsdata_node)
        kpoints = [[coord for coord in kpoint] for kpoint in bandsdata_node.get_kpoints()]
        labels = bandsdata_node.labels
        bands_incl_occ = bandsdata_node.get_bands(also_occupations=True)
        bands = bands_incl_occ[0]
        occupations = bands_incl_occ[1]
        unit_y = bandsdata_node.units
        if hasattr(bandsdata_node, "cell") and bandsdata_node.cell is not None:
            self.set_reference_cell(bandsdata_node.cell)
        self.import_band_structure(
            data_label,
            kpoints,
            bands,
            path_labels=labels,
            occupations=occupations,
            unit_y=unit_y,
            align_to_vbm=align_to_vbm,
        )

    def calculate_band_gap(self, data_label, vbm_band_idx=None):
        """
        Calculate the direct and indirect band gap of the band structure.

        With the parameter ``vbm_band_idx`` the index of the highest valence band can be passed on.

        Otherwise, if the occupations are given for the eigenvalues, the highest valence band and
        lowest conduction band are determined based on the occupation number; an occupation below
        0.5 is considered as unoccupied.

        If the previous cases do not apply the energy level 0~eV is considered to be within the
        band gap.

        Parameters
        ----------
        data_label : str
            Internal label of the data set.
        vbm_band_idx : int or list (optional)
            Index of the highest occupied band. In case of spin non-degenerate calculations a list
            of two indices can be given (otherwise the indices for spin-up and spin-down channels
            are assumed to be the same). If an occupation list is not available this parameter is
            used to calculate the band gap. Default value is ``Ǹone``.

        Returns
        -------
        band_gap : dict or list
            A dictionary containing information on the size and nature of the band gap, e.g.
            ``{'direct_gap': 2.605352729999999, 'direct_gap_kpoint': [0.0, 0.0, 0.0],
            'indirect_gap': 0.6067583699999997, 'vbm_kpoint': [0.0, 0.0, 0.0], 'vbm_energy': 0.0,
            'vbm_band_idx': 3, 'cbm_kpoint': [0.41447368, 0.0, 0.41447368],
            'cbm_energy': 0.6067583699999997, 'cbm_band_idx': 4}``. In case of spin non-degeneracy
            the dictionary is given for both spins as a list of dictionaries.
        """
        bands_data = self._return_data_set(data_label, dict_tree=["band_structure"])
        band_gap = []
        vbm_band_indices = []
        if "bands" in bands_data:
            band_types = [""]
        else:
            band_types = ["_spinup", "_spindown"]

        if vbm_band_idx is not None:
            if not isinstance(vbm_band_idx, list):
                vbm_band_indices = [vbm_band_idx] * len(band_types)
        elif "occupations" in bands_data or "occupations_spinup" in bands_data:
            for band_type in band_types:
                vbm_band_indices.append(
                    self._check_occupations(bands_data["occupations" + band_type]) - 1
                )
        else:
            for band_type in band_types:
                for band_idx, band in enumerate(bands_data["bands" + band_type]):
                    if min(band) > 0.0:
                        vbm_band_indices.append(band_idx - 1)
                        break

        for band_type, vbm_idx in zip(band_types, vbm_band_indices):
            direct_gap = self._calculate_smallest_direct_energy_diff(
                bands_data["kpoints"],
                bands_data["bands" + band_type][vbm_idx],
                bands_data["bands" + band_type][vbm_idx + 1],
            )
            indirect_gap = self._calculate_smallest_energy_diff(
                bands_data["kpoints"],
                bands_data["bands" + band_type][vbm_idx],
                bands_data["bands" + band_type][vbm_idx + 1],
            )
            band_gap.append(
                {
                    "direct_gap": direct_gap["energy"],
                    "direct_gap_kpoint": direct_gap["kpoint"],
                    "direct_gap_label": direct_gap["label"],
                    "direct_gap_rel_distance": direct_gap["rel_distance"],
                    "indirect_gap": indirect_gap["energy"],
                    "vbm_kpoint": indirect_gap["kpoint_max"],
                    "vbm_label": indirect_gap["label_max"],
                    "vbm_rel_distance": indirect_gap["rel_distance_max"],
                    "vbm_energy": max(bands_data["bands" + band_type][vbm_idx]),
                    "vbm_band_idx": vbm_idx,
                    "cbm_kpoint": indirect_gap["kpoint_min"],
                    "cbm_energy": min(bands_data["bands" + band_type][vbm_idx + 1]),
                    "cbm_label": indirect_gap["label_min"],
                    "cbm_rel_distance": indirect_gap["rel_distance_min"],
                    "cbm_band_idx": vbm_idx + 1,
                }
            )
        if len(band_gap) == 1:
            band_gap = band_gap[0]
        return band_gap

    def calculate_smallest_energy_diff(self, data_label, band_idx1, band_idx2):
        """
        Calculate the smallest energy difference between two bands.

        Parameters
        ----------
        data_label : str
            Internal label of the data set.
        band_idx1 : int
            Index of the first band.
        band_idx2 : int
            Index of the second band.

        Returns
        -------
        energy_diff : dict or list
            Dictionary containing the energy difference and the k-points of the band maximum and
            band minimum, e.g.: ``{'energy': 3.7729889100000005, 'kpoint_min': [0.5, 0.25, 0.75],
            'kpoint_max' : [0.5, 0.25, 0.75]}``.
            In case of spin non-degeneracy the dictionary is given for both spins as a list of
            dictionaries.
        """
        bands_data = self._return_data_set(data_label, dict_tree=["band_structure"])
        if "bands" in bands_data:
            band_types = [""]
        else:
            band_types = ["_spinup", "_spindown"]

        energy_diff = []
        for band_type in band_types:
            energy_diff.append(
                self._calculate_smallest_energy_diff(
                    bands_data["kpoints"],
                    bands_data["bands" + band_type][band_idx1],
                    bands_data["bands" + band_type][band_idx2],
                )
            )
        if len(energy_diff) == 1:
            energy_diff = energy_diff[0]
        return energy_diff

    def calculate_smallest_direct_energy_diff(self, data_label, band_idx1, band_idx2):
        """
        Calculate the smallest direct energy difference between two bands.

        Parameters
        ----------
        data_label : str
            Internal label of the data set.
        band_idx1 : int
            Index of the first band.
        band_idx2 : int
            Index of the second band.

        Returns
        -------
        energy_diff : dict or list
            Dictionary containing the energy difference and the k-points of the band maximum and
            band minimum, e.g.: ``{'energy': 2.605352729999999, 'kpoint': [0.0, 0.0, 0.0]}``.
            In case of spin non-degeneracy the dictionary is given for both spins as a list of
            dictionaries.
        """
        bands_data = self._return_data_set(data_label, dict_tree=["band_structure"])
        if "bands" in bands_data:
            band_types = [""]
        else:
            band_types = ["_spinup", "_spindown"]

        energy_diff = []
        for band_type in band_types:
            energy_diff.append(
                self._calculate_smallest_direct_energy_diff(
                    bands_data["kpoints"],
                    bands_data["bands" + band_type][band_idx1],
                    bands_data["bands" + band_type][band_idx2],
                )
            )
        if len(energy_diff) == 1:
            energy_diff = energy_diff[0]
        return energy_diff

    def calculate_energy_diff_at_kpoint(self, data_labels, band_idx1, band_idx2, kpoint):
        """
        Calculate the energy difference between two bands at a certain k-point.

        Parameters
        ----------
        data_labels : str or list
            Internal labels of the data sets. In case only one label is given as a string both
            bands are assumed to be part of the same data set.
        band_idx1 : int
            Index of the first band.
        band_idx2 : int
            Index of the second band.
        kpoint : list
            K-point at which the difference is calculated.

        Returns
        -------
        energy_diff : float
            Energy difference between the two bands.
        """
        if isinstance(data_labels, str):
            data_labels = [data_labels, data_labels]
        band_indices = (band_idx1, band_idx2)

        energies = [[], []]
        is_spin_pol = [False, False]
        for data_label, band_idx, energies_list, spin_pol_bool in zip(
            data_labels, band_indices, energies, is_spin_pol
        ):
            bands_data = self._return_data_set(data_label, dict_tree=["band_structure"])
            if "bands" in bands_data:
                band_types = [""]
            else:
                band_types = ["_spinup", "_spindown"]
                spin_pol_bool = True

            for band_type in band_types:
                for kpoint0, energy in zip(
                    bands_data["kpoints"], bands_data["bands" + band_type][band_idx]
                ):
                    if all(kpt0 == kpt1 for kpt0, kpt1 in zip(kpoint0, kpoint)):
                        energies_list.append(energy)
                        if not spin_pol_bool:
                            energies_list.append(energy)

        energy_diff = []
        for spin_idx in range(2):
            energy_diff.append(energies[1][spin_idx] - energies[0][spin_idx])

        if not any(is_spin_pol):
            energy_diff = energy_diff[0]
        return energy_diff

    def analyse_band(self, data_label, band_idx):
        """
        Calculate the minimum and the maximum energy of the band as well as their k-points.

        Parameters
        ----------
        data_label : str
            Internal label of the data set.
        band_idx : int
            Index of the band that is analysed.

        Returns
        -------
        band_min : dict
            Dictionary containing the minimum energy and the corresponding k-point.
        band_max : dict
            Dictionary containing the maximum energy and the corresponding k-point.
        """
        bands_data = self._return_data_set(data_label, dict_tree=["band_structure"])
        if "bands" in bands_data:
            band_min, band_max = self._analyse_band(
                bands_data["bands"][band_idx], bands_data["kpoints"]
            )
        else:
            band_min = []
            band_max = []
            for spin in ["_spinup", "_spindown"]:
                band_min_spin, band_max_spin = self._analyse_band(
                    bands_data["bands" + spin][band_idx], bands_data["kpoints"]
                )
                band_min.append(band_min_spin)
                band_max.append(band_max_spin)
        return band_min, band_max

    def shift_bands(self, data_label, energy_shift, spin="up"):
        """
        Shift the bands of one data set.

        Parameters
        ----------
        data_label : str
            Data label of the data set.
        energy_shift : float
            Value to shift the bands.
        spin : str (optional)
            In case of the spin non-degenerate case the spin ('up' or 'down').
        """
        bands_data = self._return_data_set(
            data_label, dict_tree=["band_structure"], deepcopy=False
        )
        if "bands" in bands_data:
            self._data[data_label]["bands"] = self._shift_bands(bands_data["bands"], energy_shift)
        else:
            self._data[data_label]["bands_spin" + spin] = self._shift_bands(
                bands_data["bands_spin" + spin], energy_shift
            )

    def _process_band_structure(self, kpoints, bands, path_labels, occupations, align_to_vbm):
        bands_data = {}
        if self._path_specifications is None:
            segments, path_segments = self._initialize_segments(kpoints, path_labels)
            self._path_specifications = segments
        else:
            path_segments = self._check_segments(kpoints, path_labels)
        bands_data["path_segments"] = path_segments
        bands_data["kpoints"] = kpoints

        # Process eigenvalues and occupations:
        if isinstance(bands[0][0], (list, tuple, np.ndarray)):
            # Spin non-degenerate case:
            # Store bands and occupations:
            bands_transposed = np.asarray(bands[0]).transpose()
            bands_data["bands_spinup"] = bands_transposed.tolist()
            bands_transposed = np.asarray(bands[1]).transpose()
            bands_data["bands_spindown"] = bands_transposed.tolist()
            if occupations is not None:
                occ_transposed = np.asarray(occupations[0]).transpose()
                bands_data["occupations_spinup"] = occ_transposed.tolist()
                occ_transposed = np.asarray(occupations[1]).transpose()
                bands_data["occupations_spindown"] = occ_transposed.tolist()
            # Align bands:
            if align_to_vbm:
                vbm_energies = []
                for spin in ["_spinup", "_spindown"]:
                    lowest_unocc_idx = self._check_occupations(bands_data["occupations" + spin])
                    valence_band = bands_data["bands" + spin][lowest_unocc_idx - 1]
                    _, vbm = self._analyse_band(valence_band, bands_data["kpoints"])
                    vbm_energies.append(vbm["energy"])
                vbm_energy_max = max(vbm_energies)
                for spin in ["_spinup", "_spindown"]:
                    bands_data["bands" + spin] = self._shift_bands(
                        bands_data["bands" + spin], -1.0 * vbm_energy_max
                    )

        else:
            # Spin degenerate case:
            # Store bands and occupations:
            bands_transposed = np.asarray(bands).transpose()
            bands_data["bands"] = bands_transposed.tolist()
            if occupations is not None:
                occ_transposed = np.asarray(occupations).transpose()
                bands_data["occupations"] = occ_transposed.tolist()
            # align bands:
            if align_to_vbm:
                lowest_unocc_idx = self._check_occupations(bands_data["occupations"])
                valence_band = bands_data["bands"][lowest_unocc_idx - 1]
                _, vbm = self._analyse_band(valence_band, bands_data["kpoints"])
                bands_data["bands"] = self._shift_bands(bands_data["bands"], -1.0 * vbm["energy"])
        return bands_data

    @staticmethod
    def _analyse_band(band, kpoints):
        """Return the minimum and maximum value of a band and the corresponding kpoints."""
        band_min = min(band)
        band_max = max(band)
        kpoint_min = kpoints[band.index(min(band))]
        kpoint_max = kpoints[band.index(max(band))]
        return {"energy": band_min, "kpoint": kpoint_min}, {
            "energy": band_max,
            "kpoint": kpoint_max,
        }

    @staticmethod
    def _shift_bands(bands, energy_shift):
        """Shift bands for a certain value."""
        for band_idx in range(len(bands)):
            for kpoint_idx in range(len(bands[band_idx])):
                bands[band_idx][kpoint_idx] += energy_shift
        return bands

    def _calculate_smallest_direct_energy_diff(self, kpoints, band1, band2):
        """Calculate smallest direct energy difference between two bands."""
        energy_diff = {"energy": band2[0] - band1[0], "kpoint": kpoints[0]}
        for kpoint, band1_en, band2_en in zip(kpoints, band1, band2):
            if energy_diff["energy"] > band2_en - band1_en:
                energy_diff["energy"] = max(band2_en - band1_en, 0.0)
                energy_diff["kpoint"] = kpoint
        energy_diff["label"], energy_diff["rel_distance"] = self._find_kpoint_label_and_distance(
            energy_diff["kpoint"]
        )
        return energy_diff

    def _calculate_smallest_energy_diff(self, kpoints, band1, band2):
        """Calculate smallest energy difference between two bands."""
        band_min = min(band2)
        band_max = max(band1)
        kpoint_min = kpoints[band2.index(min(band2))]
        kpoint_max = kpoints[band1.index(max(band1))]
        energy_diff = {
            "energy": max(band_min - band_max, 0.0),
            "kpoint_min": kpoint_min,
            "kpoint_max": kpoint_max,
        }
        for bnd_type, kpoint in zip(("min", "max"), (kpoint_min, kpoint_max)):
            (
                energy_diff[f"label_{bnd_type}"],
                energy_diff[f"rel_distance_{bnd_type}"],
            ) = self._find_kpoint_label_and_distance(kpoint)
        return energy_diff

    @staticmethod
    def _check_occupations(occupations):
        """Check where occupation drops to below 0.5 electrons."""
        lowest_unocc_idx = 0
        for occ_index, occ_band in enumerate(occupations):
            if occ_band[0] < 0.5:
                lowest_unocc_idx = occ_index
                break
        return lowest_unocc_idx

    def _find_kpoint_label_and_distance(self, kpoint):
        """Find k-point label or relative distance to the previous high-symmetry point."""
        label = None
        distance = 0.0
        for path_segment in self._path_specifications:
            for label_type in ["st", "e"]:
                if all(
                    abs(kpt0 - kpt1) < 1e-5
                    for kpt0, kpt1 in zip(kpoint, path_segment[f"kpoint_{label_type}"])
                ):
                    label = path_segment[f"label_{label_type}"]
                    distance = 0.0
                    break
            if label is None:
                dir_seg = np.array(path_segment["kpoint_e"]) - np.array(path_segment["kpoint_st"])
                dir_kpt = np.array(kpoint) - np.array(path_segment["kpoint_st"])
                angle = calc_angle(dir_seg, dir_kpt)
                if abs(angle) <= 0.01:
                    distance = np.linalg.norm(dir_kpt) / np.linalg.norm(dir_seg)
                    break
        return label, distance

    def _initialize_segments(self, kpoints, path_labels):
        """Generate the segments of the k-path stored in ``_path_specifications`` and the path."""
        specifications = []
        path_segments = []

        # Initialize first segment:
        path = [0.0]
        end_idx = 1
        dir_path = np.subtract(kpoints[1], kpoints[0])

        for idx_kpoint in range(len(kpoints[1:])):
            append_segment = False
            dir0 = np.subtract(kpoints[idx_kpoint + 1], kpoints[idx_kpoint])

            # In case dir_path is [0.0, 0.0, 0.0] we set dir as dir_path
            if np.linalg.norm(dir_path) < 10.0 ** (-4.0):
                dir_path = dir0

            # Calculate distance between the two adjacent k-points to the path:
            dist = np.linalg.norm(dir0)

            # If the distance is very small or if there is a change in direction we assume a
            # segment is finished:
            if dist > 10.0 ** (-4.0):
                path.append(path[-1] + dist)
                # A change in direction is indicated by a change in the angle:
                if abs(calc_angle(dir0, dir_path)) > 0.01:
                    end_idx = idx_kpoint
                    path_st = [path[-2], path.pop(-1)]
                    append_segment = True
            elif len(path) > 1:
                end_idx = idx_kpoint
                path_st = [path[-1]]
                append_segment = True

            # Append segment:
            if append_segment:
                # scale path based on the reciprocal cell (if given):
                scaling_factor = self._calculate_scaling_factor(
                    kpoints[end_idx - len(path) + 1], kpoints[end_idx]
                )
                path[1:] = [
                    path[0] + (path_val - path[0]) * scaling_factor for path_val in path[1:]
                ]
                path_st = [path[0] + (path_val - path[0]) * scaling_factor for path_val in path_st]

                # The new direction of the path is set:
                dir_path = dir0

                # Create segment for _path_specifications
                path_label_st = self._find_path_label(path_labels, end_idx - len(path) + 1)
                path_label_e = self._find_path_label(path_labels, end_idx)
                specification = {
                    "label_st": path_label_st,
                    "label_e": path_label_e,
                    "kpoint_st": kpoints[end_idx - len(path) + 1],
                    "kpoint_e": kpoints[end_idx],
                    "pos_st": path[0],
                    "pos_e": path[-1],
                    "scaling_factor": scaling_factor,
                }

                # Create segment dict for path
                segment_dict = {
                    "path": path,
                    "index_st": end_idx - len(path) + 1,
                    "index_e": end_idx,
                }

                # A segment with less than three points is considered as discontinuity:
                if len(path) > 2:
                    specifications.append(specification)
                    path_segments.append(segment_dict)
                else:
                    # We subtract the discontinouos part from the path:
                    path_st = [path_el - path[1] + path[0] for path_el in path_st]

                # Define new path
                path = path_st

        # Append final segment:
        scaling_factor = self._calculate_scaling_factor(
            kpoints[len(kpoints) - len(path)], kpoints[-1]
        )
        path[1:] = [path[0] + (path_val - path[0]) * scaling_factor for path_val in path[1:]]
        path_label_st = self._find_path_label(path_labels, len(kpoints) - len(path))
        path_label_e = self._find_path_label(path_labels, len(kpoints) - 1)
        specification = {
            "label_st": path_label_st,
            "label_e": path_label_e,
            "kpoint_st": kpoints[len(kpoints) - len(path)],
            "kpoint_e": kpoints[-1],
            "pos_st": path[0],
            "pos_e": path[-1],
            "scaling_factor": scaling_factor,
        }
        segment_dict = {
            "path": path,
            "index_st": len(kpoints) - len(path),
            "index_e": len(kpoints) - 1,
        }
        if len(path) > 2:
            specifications.append(specification)
            path_segments.append(segment_dict)
        return specifications, path_segments

    @staticmethod
    def _find_path_label(path_labels, kpoint_idx):
        """Check for a label for a certain index that is stored in ``_path_specifications``."""
        path_label = r"\,"
        if path_labels:
            for label in path_labels:
                if label[0] == kpoint_idx:
                    if "DELTA" in label[1].upper():
                        path_label = "\\Delta_0"
                    elif label[1].upper() == "GAMMA":
                        path_label = "\\Gamma"
                    elif "SIGMA" in label[1].upper():
                        path_label = "\\Sigma_0"
                    else:
                        path_label = re.sub(r"^([A-Za-z]+)(\d+)$", r"\1_\2", label[1])
        return path_label

    def _check_segments(self, kpoints, path_labels):
        """Check the compatibility of the k-points and generates the path."""
        path = []
        used_indices = []

        for segment in self._path_specifications:
            # Search for start and end-kpoints of the segments:
            start_indices = []
            end_indices = []
            for kpoint_idx, kpoint in enumerate(kpoints):
                if kpoint_idx in used_indices:
                    continue
                vec_st = np.subtract(kpoint, segment["kpoint_st"])
                vec_e = np.subtract(kpoint, segment["kpoint_e"])
                if np.linalg.norm(vec_st) < 10.0 ** (-4.0):
                    if len(start_indices) > 0 and start_indices[-1] == kpoint_idx - 1:
                        start_indices[-1] = kpoint_idx
                    else:
                        start_indices.append(kpoint_idx)
                if np.linalg.norm(vec_e) < 10.0 ** (-4.0):
                    if len(end_indices) > 0 and end_indices[-1] == kpoint_idx - 1:
                        pass
                    else:
                        end_indices.append(kpoint_idx)

            # Check that indices are found and end_idx is larger than start index_st
            # paths from right-->left are not yet implemented (extension for later)
            indices_comb = itertools.product(start_indices, end_indices)
            for start_idx, end_idx in indices_comb:
                if end_idx > start_idx:
                    path0 = [segment["pos_st"]]
                    for kpoint_idx in range(start_idx + 1, end_idx + 1, 1):
                        dist = (
                            np.linalg.norm(
                                np.subtract(kpoints[kpoint_idx], kpoints[kpoint_idx - 1])
                            )
                            * segment["scaling_factor"]
                        )
                        path0.append(path0[-1] + dist)
                    # The segment needs to have the same length to be added:
                    if abs(path0[-1] - segment["pos_e"]) < 0.01:
                        path.append({"path": path0, "index_st": start_idx, "index_e": end_idx})
                        used_indices += list(range(start_idx, end_idx + 1))
                        break
        return path

    def _calculate_scaling_factor(self, kpoint_scaled_st, kpoint_scaled_e):
        """Calculate the scaling factor for the path segment based on the reciprocal cell."""
        scaling_factor = 1.0
        if self._reciprocal_cell is not None:
            kpoint_st = np.zeros(3)
            kpoint_e = np.zeros(3)
            for dir_idx in range(3):
                kpoint_st += kpoint_scaled_st[dir_idx] * self._reciprocal_cell[dir_idx]
                kpoint_e += kpoint_scaled_e[dir_idx] * self._reciprocal_cell[dir_idx]
            segment_length = np.linalg.norm(np.subtract(kpoint_e, kpoint_st))
            scaling_factor = segment_length / np.linalg.norm(
                np.subtract(kpoint_scaled_e, kpoint_scaled_st)
            )
        return scaling_factor

    def _process_path_labels(self):
        """Process path labels for the plot."""
        path_ticks = [self._path_specifications[0]["pos_st"]]
        path_labels = ["$" + self._path_specifications[0]["label_st"] + "$"]
        for seg_idx, segment in enumerate(self._path_specifications):
            path_ticks.append(segment["pos_e"])

            if (seg_idx < len(self._path_specifications) - 1) and (
                segment["label_e"] != self._path_specifications[seg_idx + 1]["label_st"]
            ):
                path_labels.append(
                    "$\\frac{"
                    + segment["label_e"]
                    + "}{"
                    + self._path_specifications[seg_idx + 1]["label_st"]
                    + "}$"
                )
            else:
                path_labels.append("$" + segment["label_e"] + "$")
        return path_ticks, path_labels

    def _generate_data_sets_for_plot(self, data_idx, data_label):
        bands_data = self._return_data_set(data_label, dict_tree=["band_structure"])
        data_sets = []
        for path_segment in bands_data["path_segments"]:
            if "bands" in bands_data:
                for band in bands_data["bands"]:
                    y_values = band[path_segment["index_st"] : path_segment["index_e"] + 1]
                    data_set = {
                        "x_values": path_segment["path"],
                        "y_values": y_values,
                        "color": data_idx,
                        "linestyle": data_idx,
                        "linewidth": data_idx,
                        "legendgroup": data_label,
                    }
                    data_sets.append(data_set)
                if len(data_sets) > 0:
                    data_sets[0]["label"] = data_label
            elif "bands_spinup" in bands_data:
                for spin_idx, band_type in enumerate(["bands_spinup", "bands_spindown"]):
                    for band in bands_data[band_type]:
                        y_values = band[path_segment["index_st"] : path_segment["index_e"] + 1]
                        data_set = {
                            "x_values": path_segment["path"],
                            "y_values": y_values,
                            "color": 2 * data_idx + spin_idx,
                            "linestyle": 2 * data_idx + spin_idx,
                            "linewidth": 2 * data_idx + spin_idx,
                            "legendgroup": data_label,
                        }
                        data_sets.append(data_set)
                if len(data_sets) > 0:
                    data_sets[0]["label"] = data_label
        return data_sets
