"""
Module to plot the band structure and the density of states separately or combined.

This module is still work in progress.
"""

# Internal library imports
from aim2dat.plots.base_plot import _BasePlot
from aim2dat.plots.base_mixin import _HLineMixin, _VLineMixin
from aim2dat.plots.base_band_structure import _BaseBandStructure
from aim2dat.plots.base_dos import _BaseDensityOfStates


def BandStructure(*args, **kwargs):
    """Depreciated band structure class."""
    from warnings import warn

    warn(
        "This class will be removed, please use `BandStructurePlot` instead.",
        DeprecationWarning,
        2,
    )
    return BandStructurePlot(*args, **kwargs)


def DensityOfStates(*args, **kwargs):
    """Depreciated DOS class."""
    from warnings import warn

    warn(
        "This class will be removed, please use `DOSPlot` instead.",
        DeprecationWarning,
        2,
    )
    return DOSPlot(*args, **kwargs)


def BandStructureDensityOfStates(*args, **kwargs):
    """Depreciated band structure DOS class."""
    from warnings import warn

    warn(
        "This class will be removed, please use `BandStructureDOSPlot` instead.",
        DeprecationWarning,
        2,
    )
    return BandStructureDOSPlot(*args, **kwargs)


class BandStructurePlot(_BasePlot, _HLineMixin, _VLineMixin, _BaseBandStructure):
    """
    Plot the band structure of a crystalline structure.
    """

    _object_title = "Band Structure Plot"

    def __init__(self, **kwargs):
        """Initialize class."""
        _BasePlot.__init__(self, **kwargs)
        _BaseBandStructure.__init__(self)

    def _print_extra_properties(self):
        """Print extra properties specific for this plot."""
        ref_cell_set = True
        if self._reciprocal_cell is None:
            ref_cell_set = False
        output_str = f" Reference cell set: {ref_cell_set}.\n"
        if ref_cell_set:
            output_str += "  Rec. cell vectors: "
            for vector_idx in range(3):
                for coord_idx in range(3):
                    output_str += (
                        "{:.5f}".format(self._reciprocal_cell[vector_idx][coord_idx]) + "  "
                    )
                if vector_idx < 2:
                    output_str += "\n                     "
            output_str += "\n"
        output_str += "\n"
        if self._path_specifications is not None:
            output_str += " Path segments:\n"
            for segment_idx in range(len(self._path_specifications)):
                segment = self._path_specifications[segment_idx]
                segment_str = f"  {segment_idx + 1}. Segment: "
                output_str += segment_str
                for coord_idx in range(3):
                    output_str += "{:.5f}".format(segment["kpoint_st"][coord_idx]) + "  "
                output_str += segment["label_st"] + "\n"
                output_str += " ".join(["" for idx in range(len(segment_str))]) + " "
                for coord_idx in range(3):
                    output_str += "{:.5f}".format(segment["kpoint_e"][coord_idx]) + "  "
                output_str += segment["label_e"] + "\n"
        return output_str

    def _prepare_to_plot(self, data_labels, subplot_assignment):
        if not self.x_range:
            self.x_range = [0.0, self._path_specifications[-1]["pos_e"]]

        general_data_sets = []
        path_ticks, path_labels = self._process_path_labels()
        for x_val in path_ticks:
            general_data_sets.append(
                {"x": x_val, "color": "black", "linestyle": "-", "scaled": True, "type": "vline"}
            )
        general_data_sets.append(
            {
                "y": 0.0,
                "xmin": 0.0,
                "xmax": self._path_specifications[-1]["pos_e"],
                "color": "black",
                "linestyle": "dashed",
                "scaled": False,
                "type": "hline",
            }
        )
        data_sets = [general_data_sets.copy() for idx0 in range(max(subplot_assignment) + 1)]
        for idx, (data_label, subp_a) in enumerate(zip(data_labels, subplot_assignment)):
            data_sets[subp_a] += self._generate_data_sets_for_plot(idx, data_label)
        self._auto_set_axis_properties(y_label="Energy in eV")
        return data_sets, path_ticks, None, path_labels, None, None


class DOSPlot(_BasePlot, _VLineMixin, _BaseDensityOfStates):
    """
    Class to plot the density of states.

    Attributes
    ----------
    dos_comp_threshold : float
        Threshold to compare the density of states if ``detect_equivalent_kinds`` is set to
        ``True`` when importing projected density of states data sets.
    sum_pdos : bool
        Whether to sum all pDOS data sets to obtain a tDOS.
    per_atom : bool
        Normalize all density of states data sets to the numer of atoms.
    """

    _object_title = "DOS Plot"

    def __init__(
        self,
        ratio=(10, 4),
        pdos_plot_type="line",
        tdos_plot_type="fill",
        dos_comp_threshold=0.47,
        smearing_method="gaussian",
        smearing_delta=0.005,
        smearing_sigma=5.0,
        sum_pdos=False,
        per_atom=False,
        **kwargs,
    ):
        """Initialize class."""
        _BasePlot.__init__(self, ratio=ratio, **kwargs)
        _BaseDensityOfStates.__init__(
            self,
            pdos_plot_type,
            tdos_plot_type,
            dos_comp_threshold,
            smearing_method,
            smearing_delta,
            smearing_sigma,
            sum_pdos,
            per_atom,
        )

    def _prepare_to_plot(self, data_labels, subplot_assignment):
        pdos_all = [[] for idx0 in range(max(subplot_assignment) + 1)]
        tdos_all = [[] for idx0 in range(max(subplot_assignment) + 1)]

        color_map = {}
        ls_map = {}

        for idx, (data_label, subp_a) in enumerate(zip(data_labels, subplot_assignment)):
            dos_data = self._return_data_set(data_label)
            nr_atoms = 1
            y_label = "DOS in states/eV/cell"
            if self.per_atom and "chem_formula" in dos_data:
                nr_atoms = sum([el_quant for el_quant in dos_data["chem_formula"].values()])
                y_label = "DOS in states/eV/atom"

            # Generate pDOS data sets:
            if "pdos" in dos_data and self.sum_pdos:
                dos_data["tdos"] = self._sum_pdos(dos_data["pdos"])
            if "tdos" in dos_data:
                tdos_all[subp_a] += self._dos_create_plot_data_set(
                    color_map, ls_map, dos_data["tdos"], 1.0, self.tdos_plot_type, 0
                )
            if "pdos" in dos_data:
                pdos_all[subp_a] += self._dos_create_plot_data_set(
                    color_map, ls_map, dos_data["pdos"], nr_atoms, self.pdos_plot_type, 1
                )

        data_sets = []
        for pdos_ds, tdos_ds in zip(pdos_all, tdos_all):
            data_sets.append(
                [{"x": 0.0, "color": "black", "linestyle": "--", "scaled": True, "type": "vline"}]
                + tdos_ds
                + pdos_ds
            )
        self._auto_set_axis_properties(x_label="Energy in eV", y_label=y_label)
        return data_sets, None, None, None, None, None


class BandStructureDOSPlot(
    _BasePlot, _HLineMixin, _VLineMixin, _BaseBandStructure, _BaseDensityOfStates
):
    """
    Class to plot the band structure combined with the density of states.

    Attributes
    ----------
    dos_comp_threshold : float
        Threshold to compare the density of states if ``detect_equivalent_kinds`` is set to
        ``True`` when importing projected density of states data sets.
    sum_pdos : bool
        Whether to sum all pDOS data sets to obtain a tDOS.
    per_atom : bool
        Normalize all density of states data sets to the numer of atoms.
    """

    def __init__(
        self,
        ratio=(12, 7),
        show_legend=[False, True],
        subplot_hspace=5,
        subplot_wspace=0.1,
        subplot_nrows=1,
        subplot_ncols=3,
        subplot_sharex=False,
        subplot_sharey=True,
        subplot_gridspec=[(0, 1, 0, 2), (0, 1, 2, 3)],
        pdos_plot_type="line",
        tdos_plot_type="fill",
        dos_comp_threshold=0.47,
        smearing_method="gaussian",
        smearing_delta=0.005,
        smearing_sigma=5.0,
        sum_pdos=False,
        per_atom=False,
        **kwargs,
    ):
        """Initialize class."""
        _BasePlot.__init__(
            self,
            ratio=ratio,
            show_legend=show_legend,
            subplot_hspace=subplot_hspace,
            subplot_wspace=subplot_wspace,
            subplot_nrows=subplot_nrows,
            subplot_ncols=subplot_ncols,
            subplot_sharex=subplot_sharex,
            subplot_sharey=subplot_sharey,
            subplot_gridspec=subplot_gridspec,
            **kwargs,
        )
        _BaseBandStructure.__init__(self)
        _BaseDensityOfStates.__init__(
            self,
            pdos_plot_type,
            tdos_plot_type,
            dos_comp_threshold,
            smearing_method,
            smearing_delta,
            smearing_sigma,
            sum_pdos,
            per_atom,
        )

    def shift_bands_and_dos_to_vbm(self, data_label):
        """
        Shift the bands and the density of states such that the VBM is zero.

        Parameters
        ----------
        data_label : str
            Data label of the data set.
        """
        data_set = self._return_data_set(data_label, deepcopy=False)
        if "occupations" in data_set["band_structure"]:
            lowest_unocc_idx = self._check_occupations(data_set["band_structure"]["occupations"])
            valence_band = data_set["band_structure"]["bands"][lowest_unocc_idx - 1]
            _, vbm = self._analyse_band(valence_band, data_set["band_structure"]["kpoints"])
            data_set["band_structure"]["bands"] = self._shift_bands(
                data_set["band_structure"]["bands"], -1.0 * vbm["energy"]
            )
            vbm_energy_max = vbm["energy"]
            # if "pdos" in data_set:
            # self.__shift_projected_dos__(data_set["pdos"], -1.0 * vbm["energy"])
            # if "tdos" in data_set:
            # self.__shift_total_dos__(data_set["tdos"], -1.0 * vbm["energy"])
        elif "occupations_spinup" in data_set["band_structure"]:
            vbm_energies = []
            for spin in ["_spinup", "_spindown"]:
                lowest_unocc_idx = self._check_occupations(
                    data_set["band_structure"]["occupations" + spin]
                )
                valence_band = data_set["band_structure"]["bands" + spin][lowest_unocc_idx - 1]
                _, vbm = self._analyse_band(valence_band, data_set["band_structure"]["kpoints"])
                vbm_energies.append(vbm["energy"])
            vbm_energy_max = max(vbm_energies)
            for spin in ["_spinup", "_spindown"]:
                data_set["band_structure"]["bands" + spin] = self._shift_bands(
                    data_set["band_structure"]["bands" + spin], -1.0 * vbm_energy_max
                )
        self.shift_dos(data_label, -1.0 * vbm_energy_max)

    def shift_bands_and_dos(self, data_label, shift):
        """
        Shift band structure and density of states.

        TODO: include spin-polarized case.

        Parameters
        ----------
        data_label : str
            Data label of the data set.
        shift : float
            Value to shift band structure and density of states.
        """
        data_set = self._return_data_set(data_label, deepcopy=False)
        if "band_structure" in data_set:
            data_set["band_structure"]["bands"] = self._shift_bands(
                data_set["band_structure"]["bands"], shift
            )
        self.shift_dos(data_label, shift)

    def _prepare_to_plot(self, data_labels, subplot_assignment):
        # data_label = data_labels[0]
        # data_set = self._return_data_set(data_label)

        path_ticks, path_labels = self._process_path_labels()
        data_set_bands = [
            {
                "y": 0.0,
                "color": "black",
                "linestyle": "dashed",
                "scaled": True,
                "type": "hline",
            }
        ]
        for x_val in path_ticks:
            data_set_bands.append(
                {"x": x_val, "color": "black", "linestyle": "-", "scaled": True, "type": "vline"}
            )
        n_tdos = 0
        n_pdos = 0
        for idx, data_label in enumerate(data_labels):
            data_set_bands += self._generate_data_sets_for_plot(idx, data_label)
            dos_data = self._return_data_set(data_label)
            if "pdos" in dos_data:
                n_pdos += 1
            if "tdos" in dos_data or ("pdos" in dos_data and self.sum_pdos):
                n_tdos += 1

        data_set_dos = [
            {"y": 0.0, "color": "black", "linestyle": "--", "scaled": True, "type": "hline"}
        ]
        color_map_pdos = {"_": i0 for i0 in range(len(data_labels))}
        color_map_tdos = {}
        ls_map = {}
        for idx, data_label in enumerate(data_labels):
            dos_label_suffix = ["", ""]
            if n_tdos > 1:
                dos_label_suffix[0] = " " + data_label
            if n_pdos > 1:
                dos_label_suffix[1] = " " + data_label
            dos_data = self._return_data_set(data_label)
            nr_atoms = 1
            x_label = [None, "DOS in states/eV/cell"]
            if self.per_atom and "chem_formula" in dos_data:
                nr_atoms = sum([el_quant for el_quant in dos_data["chem_formula"].values()])
                x_label[1] = "DOS in states/eV/atom"

            # Generate pDOS data sets:
            if "pdos" in dos_data and self.sum_pdos:
                dos_data["tdos"] = self._sum_pdos(dos_data["pdos"])
            proc_dos_data = []
            if "tdos" in dos_data:
                dos_labels = list(dos_data["tdos"]["labels"].keys())
                for label in dos_labels:
                    orb_labels = dos_data["tdos"]["labels"].pop(label)
                    orb_keywords = list(orb_labels.keys())
                    for keyw in orb_keywords:
                        orb_labels[(keyw[0] + dos_label_suffix[0], keyw[1])] = orb_labels.pop(keyw)
                    dos_data["tdos"]["labels"][label + dos_label_suffix[0]] = orb_labels
                proc_dos_data += self._dos_create_plot_data_set(
                    color_map_tdos, ls_map, dos_data["tdos"], 1.0, self.tdos_plot_type, 0
                )
            if "pdos" in dos_data:
                dos_labels = list(dos_data["pdos"]["labels"].keys())
                for label in dos_labels:
                    dos_data["pdos"]["labels"][label + dos_label_suffix[1]] = dos_data["pdos"][
                        "labels"
                    ].pop(label)
                proc_dos_data += self._dos_create_plot_data_set(
                    color_map_pdos, ls_map, dos_data["pdos"], nr_atoms, self.pdos_plot_type, 1
                )
            for data_set0 in proc_dos_data:
                new_ds = dict(data_set0)
                new_ds["x_values"] = data_set0["y_values"]
                new_ds["y_values"] = data_set0["x_values"]
                data_set_dos.append(new_ds)

        data_set_plot = [data_set_bands, data_set_dos]

        if self.x_range is None:
            self.x_range = [[0.0, self._path_specifications[-1]["pos_e"]], None]
        elif isinstance(self.x_range[0], (list, tuple)) or self.x_range[0] is None:
            self.x_range = [[0.0, self._path_specifications[-1]["pos_e"]]] + list(self.x_range[1:])
        else:
            self.x_range = [[0.0, self._path_specifications[-1]["pos_e"]], self.x_range]
        self._auto_set_axis_properties(x_label=x_label, y_label=["Energy in eV", None])
        return data_set_plot, [path_ticks, None], None, [path_labels, None], None, None
