"""Module containing base classes for density of states processing and plotting."""

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.plots.base_mixin import _SmearingMixin
from aim2dat.ext_interfaces import _return_ext_interface_modules


def _validate_plot_type(value, dos_type):
    if not isinstance(value, str):
        raise TypeError(f"`{dos_type}_plot_type` needs to be of type str.")
    if "line" not in value and "fill" not in value:
        raise ValueError(f"`{dos_type}_plot_type` must contain 'line' and/or 'fill'.")
    return value


class _BaseDensityOfStates(_SmearingMixin):
    """
    Base class for density of states plots.
    """

    _quantum_numbers = {
        "s": ("s"),
        "p": ("px", "py", "pz"),
        "d": ("d-2", "d-1", "d0", "d+1", "d+2", "dx2", "dz2", "dxy", "dyz", "dxz"),
        "f": ("f-3", "f-2", "f-1", "f0", "f+1", "f+2", "f+3"),
        "g": ("g-4", "g-3", "g-2", "g-1", "g0", "g+1", "g+2", "g+3", "g+4"),
        "i": ("i-5", "i-4", "i-3", "i-2", "i-1", "i0", "i+1", "i+2", "i+3", "i+4", "i+5"),
    }

    def __init__(
        self,
        pdos_plot_type,
        tdos_plot_type,
        dos_comp_threshold,
        smearing_method,
        smearing_delta,
        smearing_sigma,
        sum_pdos,
        per_atom,
    ):
        """Initialize class."""
        self.pdos_plot_type = pdos_plot_type
        self.tdos_plot_type = tdos_plot_type
        self.dos_comp_threshold = dos_comp_threshold
        self.smearing_method = smearing_method
        self.smearing_delta = smearing_delta
        self.smearing_sigma = smearing_sigma
        self.sum_pdos = sum_pdos
        self.per_atom = False

    @property
    def pdos_plot_type(self):
        """
        str: plot type of the pDOS data sets, supported options are ``'line'``, ``'fill'``
        and ``'linefill'``.
        """
        return self._pdos_plot_type

    @pdos_plot_type.setter
    def pdos_plot_type(self, value):
        self._pdos_plot_type = _validate_plot_type(value, "pdos")

    @property
    def tdos_plot_type(self):
        """
        str: plot type of the tDOS data sets, supported options are ``'line'``, ``'fill'``
        and ``'linefill'``.
        """
        return self._tdos_plot_type

    @tdos_plot_type.setter
    def tdos_plot_type(self, value):
        self._tdos_plot_type = _validate_plot_type(value, "tdos")

    def import_total_dos(
        self, data_label, energy, tdos, use_smearing=False, unit_x="eV", shift_dos=0.0
    ):
        """
        Import total density of states.

        Parameters
        ----------
        data_label : str
            Internal label used to plot and compare multiple data sets.
        energy : list
            List of energy values.
        tdos : list
            List of tDOS values.
        use_smearing : bool (optional)
            Whether to smear out the density of states. The default value is ``False``.
        unit_x : str (optional)
            Unit of the energy. The default value is ``'eV'``.
        shift_dos : float (optional)
            Shift the density of states by constant value. The default value is ``0.0``.
        """
        if data_label not in self._data:
            self._data[data_label] = {}
        elif "tdos" in self._data[data_label]:
            raise ValueError(f"Data label '{data_label}' already contains tDOS data.")

        # Fix for fill in case the first or last value is not zero:
        energy = np.concatenate(([min(energy) - 1.0e-5], energy, [max(energy) + 1.0e-5]))
        energy += shift_dos
        tdos = np.array(tdos)
        if len(tdos.shape) == 1:
            tdos = [tdos]
        else:
            tdos = tdos.tolist()
        for idx in range(len(tdos)):
            tdos[idx] = np.concatenate(([0.0], tdos[idx], [0.0]))
            if energy.shape != tdos[idx].shape:
                raise ValueError(
                    f"Energy and DOS have different shapes: {energy.shape[0] - 2} != "
                    + f"{tdos[idx].shape[0] - 2}.",
                )
            if idx == 1:
                tdos[idx] *= -1.0
        tdos = np.array(tdos)
        if use_smearing:
            energy, tdos = self._apply_smearing(energy, tdos)
        self._data[data_label]["tdos"] = {
            "energy": energy,
            "dos": tdos,
            "labels": {"tDOS": {("", idx): idx for idx in range(len(tdos))}},
            "unit_x": unit_x,
        }

    def import_projected_dos(
        self,
        data_label,
        energy,
        pdos,
        unit_x="eV",
        shift_dos=0.0,
        use_smearing=False,
        sum_kinds=False,
        sum_principal_qn=True,
        sum_azimuth_qn=False,
        sum_magnetic_qn=True,
        detect_equivalent_kinds=False,
        custom_kind_dict=None,
    ):
        """
        Import projected density of states.

        Parameters
        ----------
        data_label : str
            Internal label used to plot and compare multiple data sets.
        energy : list
            List of energy values.
        pdos : list
            List of projected density of states data. The list should consist of dictionaries with
            the orbital labels as keys and the pdos as values. Additionallly, the element and/or
            kind of the atom should be included.
        unit_x : str (optional)
            Unit of the energy. The default value is ``'eV'``.
        shift_dos : float (optional)
            Shift the density of states by constant value. The default value is ``0.0``.
        use_smearing : bool (optional)
            Whether to smear out the density of states. The default value is ``False``.
        sum_kinds : bool (optional)
            Whether to sum different kinds of the same element. The default value is ``False``.
        sum_principal_qn : bool (optional)
            Whether to sum up the principal quantum numbers. The default value is ``True``.
        sum_azimuth_qn : bool (optional)
            Whether to sum up the azimuth quantum numbers.
        sum_magnetic_qn : bool (optional)
            Whether to sum up the magnetic quantum numbers. The default value is ``True``.
        detect_equivalent_kinds : bool (optional)
            Tries to identify equivalent sites by calculating the difference of the projected
            densities at each energy value. The default value is ``False``.
        custom_kind_dict : dict or None (optional)
            Group the projected densities and put custom labels, e.g.
            ``{"label_1": (0, 1, 2), "label_2": (3, 4, 5)}``.
        """
        if data_label not in self._data:
            self._data[data_label] = {}
        elif "pdos" in self._data[data_label]:
            raise ValueError(f"Data label '{data_label}' already contains pDOS data.")

        energy = np.array(energy) + shift_dos

        proc_pdos = []
        elements = {}
        plot_labels = {}
        chem_formula = {}
        for pdos_idx, single_pdos in enumerate(pdos):
            plot_label, add_pdos = self._pdos_get_custom_plot_label(custom_kind_dict, pdos_idx)
            if not add_pdos:
                continue
            kind, element = self._get_kind_element(single_pdos, chem_formula)
            if plot_label is None:
                if sum_kinds:
                    plot_label = element
                else:
                    plot_label = kind
            for orb_label, density in single_pdos.items():
                if orb_label in ["kind", "element"]:
                    continue

                # Determine plot labels:
                principal_qn, azimuth_qn, magnetic_qn, spin = self._pdos_get_quantum_nrs(orb_label)
                plot_orb_label = ""
                if not sum_principal_qn and principal_qn is not None:
                    plot_orb_label += str(principal_qn)
                if not sum_azimuth_qn and azimuth_qn is not None:
                    plot_orb_label += azimuth_qn
                if not sum_magnetic_qn and magnetic_qn is not None:
                    plot_orb_label += magnetic_qn
                # labels = (plot_label, plot_orb_label, spin)

                # Add density
                density = np.array(density)
                if density.shape != energy.shape:
                    raise ValueError(
                        f"Energy and DOS have different shapes: {energy.shape[0]} != "
                        + f"{density.shape[0]}."
                    )
                if spin == 1:
                    density *= -1.0

                if plot_label not in plot_labels:
                    plot_labels[plot_label] = {}
                if (plot_orb_label, spin) in plot_labels[plot_label]:
                    proc_pdos[plot_labels[plot_label][(plot_orb_label, spin)]] += density
                else:
                    plot_labels[plot_label][(plot_orb_label, spin)] = len(proc_pdos)
                    elements[plot_label] = element
                    proc_pdos.append(density)
        proc_pdos = np.array(proc_pdos)

        # Apply smearing:
        if use_smearing:
            energy, proc_pdos = self._apply_smearing(energy, proc_pdos)

        # Detect equivalent kinds:
        if detect_equivalent_kinds and custom_kind_dict is None:
            self._pdos_detect_equivalent_sites(plot_labels, elements, proc_pdos)

        # Store pDOS
        self._data[data_label]["pdos"] = {
            "labels": plot_labels,
            "energy": energy,
            "dos": proc_pdos,
            "unit_x": unit_x,
        }

    def import_from_aiida_xydata(
        self,
        data_label,
        pdosdata,
        shift_dos=0.0,
        use_smearing=False,
        sum_kinds=False,
        sum_principal_qn=True,
        sum_azimuth_qn=False,
        sum_magnetic_qn=True,
        detect_equivalent_kinds=False,
        custom_kind_dict=None,
    ):
        """
        Read projected density of states from an AiiDA xy-data node.

        Parameters
        ----------
        data_label : str
            Internal label used to plot and compare multiple data sets.
        pdosdata : aiida.orm.array.xy or dict
            AiiDA data node or dictionary of AiiDA data nodes for each kind.
        shift_dos : float
            Shift the density of states by constant value. The default value is ``0.0``.
        use_smearing : bool (optional)
            Whether to smear out the density of states. The default value is ``False``.
        sum_kinds : bool (optional)
            Whether to sum different kinds of the same element. The default value is ``False``.
        sum_principal_qn : bool (optional)
            Whether to sum up the principal quantum numbers. The default value is ``True``.
        sum_azimuth_qn : bool (optional)
            Whether to sum up the azimuth quantum numbers.
        sum_magnetic_qn : bool (optional)
            Whether to sum up the magnetic quantum numbers. The default value is ``True``.
        detect_equivalent_kinds : bool (optional)
            Tries to identify equivalent sites by calculating the difference of the projected
            densities at each energy value. The default value is ``False``.
        custom_kind_dict : dict or None (optional)
            Group the projected densities and put custom labels, e.g.
            ``{"label_1": (0, 1, 2), "label_2": (3, 4, 5)}``.
        """
        backend_module = _return_ext_interface_modules("aiida")

        # Check if input is a single xydata node:
        # try:
        pdosdata = backend_module._load_data_node(pdosdata)

        energy = pdosdata.get_x()[1]
        unit_x = pdosdata.get_x()[2]
        pdos_aiida = pdosdata.get_y()
        pdos = []
        kinds_dict = {}
        kind_index = -1

        for pdos0_aiida in pdos_aiida:
            if pdos0_aiida[0] == "occupation":
                continue
            pdos_label = pdos0_aiida[0].split("_")
            if pdos_label[-1].lower() in ["alpha", "beta"]:
                kind = "_".join(pdos0_aiida[0].split("_")[:-2])
                orbital = "_".join(pdos0_aiida[0].split("_")[-2:])
            else:
                kind = "_".join(pdos0_aiida[0].split("_")[:-1])
                orbital = "_".join(pdos0_aiida[0].split("_")[-1:])
            if kind not in kinds_dict:
                pdos.append({"kind": kind})
                kind_index = len(pdos) - 1
                kinds_dict[kind] = kind_index
            else:
                kind_index = kinds_dict[kind]

            # Add to pdos:
            if orbital in pdos[kind_index]:
                pdos[kind_index][orbital] += pdos0_aiida[1]
            else:
                pdos[kind_index][orbital] = pdos0_aiida[1]

        self.import_projected_dos(
            data_label,
            energy,
            pdos,
            unit_x,
            shift_dos,
            use_smearing,
            sum_kinds,
            sum_principal_qn,
            sum_azimuth_qn,
            sum_magnetic_qn,
            detect_equivalent_kinds,
            custom_kind_dict,
        )

    def shift_dos(self, data_label, shift):
        """
        Shift density of states.

        Parameters
        ----------
        data_label : str
            Data label of the data set.
        shift : float
            Value to shift band structure and density of states.
        """
        data_set = self._return_data_set(data_label, deepcopy=False)
        if "pdos" in data_set:
            data_set["pdos"]["energy"] += shift
        if "tdos" in data_set:
            data_set["tdos"]["energy"] += shift

    @staticmethod
    def _get_kind_element(single_pdos, chem_formula):
        """Get kind name and element symbol."""
        element = None
        kind = ""
        if "element" in single_pdos:
            element = single_pdos["element"]
            kind = element
            if element in chem_formula:
                kind += str(chem_formula[element] + 1)
            else:
                kind += str(1)
        if "kind" in single_pdos:
            kind = single_pdos["kind"]
            if element is None:
                element = kind.split("_")[0]
            kind = "".join(kind.split("_"))
        # Add element to chem_formula:
        if element not in chem_formula:
            chem_formula[element] = 1
        else:
            chem_formula[element] += 1
        return kind, element

    def _pdos_detect_equivalent_sites(self, plot_labels, elements, pdos):
        """Detect equivalent sites and sum pDOS."""
        # TODO check kind labels?
        plot_label_list = [label for label in plot_labels.keys()]
        pdos2del = []
        for idx1, label1 in enumerate(plot_label_list):
            if label1 not in plot_labels:
                continue
            element1 = elements[label1]
            for idx2 in range(idx1 + 1, len(plot_label_list)):
                label2 = plot_label_list[idx2]
                if label2 not in plot_labels:
                    continue
                element2 = elements[label2]
                if element1 != element2:
                    continue
                if len(plot_labels[label1]) != len(plot_labels[label2]):
                    continue
                eq_density = True
                for orb_label, d_idx1 in plot_labels[label1].items():
                    if orb_label not in plot_labels[label2]:
                        eq_density = False
                        break
                    d_idx2 = plot_labels[label2][orb_label]
                    if any(
                        np.absolute(np.subtract(pdos[d_idx1], pdos[d_idx2]))
                        > self.dos_comp_threshold
                    ):
                        eq_density = False
                        break
                if eq_density:
                    for orb_label, d_idx1 in plot_labels[label1].items():
                        d_idx2 = plot_labels[label2][orb_label]
                        pdos[d_idx1] += pdos[d_idx2]
                        pdos2del.append(d_idx2)
                    del plot_labels[label2]
                    del elements[label2]
        return np.delete(pdos, pdos2del, axis=0)

    @staticmethod
    def _pdos_get_custom_plot_label(custom_kind_dict, pdos_idx):
        """Get custom plot label and check whether pDOS should be imported."""
        plot_label = None
        add_pdos = True
        if custom_kind_dict is not None:
            add_pdos = False
            for label, indices in custom_kind_dict.items():
                if pdos_idx in indices:
                    add_pdos = True
                    plot_label = label
                    break
        return plot_label, add_pdos

    def _pdos_get_quantum_nrs(self, orb_label):
        """Retrieve quantum numbers for the pDOS."""

        def check_az_mag_qn(orbital_label, quantum_numbers):
            if orbital_label == "total":
                azimuth_qn = None
                magnetic_qn = None
            elif orbital_label in quantum_numbers:
                azimuth_qn = orbital_label
                magnetic_qn = None
            elif orbital_label[0] in quantum_numbers:
                azimuth_qn = orbital_label[0]
                magnetic_qn = orbital_label[1:]
                if orbital_label not in quantum_numbers[azimuth_qn]:
                    raise ValueError(
                        f"Orbital projection on '{orb_label}' could not be processed."
                    )
            else:
                raise ValueError(f"Orbital projection on '{orb_label}' could not be processed.")
            return azimuth_qn, magnetic_qn

        principal_qn = None
        azimuth_qn = None
        magnetic_qn = None
        spin_idx = 0
        orb_splitted = orb_label.split("_")
        if orb_splitted[-1].lower() in ["alpha", "beta"]:
            spin = orb_splitted.pop(-1).lower()
            if spin == "beta":
                spin_idx = 1

        if len(orb_splitted) > 1:
            principal_qn = int(orb_splitted[0])
            azimuth_qn, magnetic_qn = check_az_mag_qn(orb_splitted[1], self._quantum_numbers)
        else:
            azimuth_qn, magnetic_qn = check_az_mag_qn(orb_splitted[0], self._quantum_numbers)
        return principal_qn, azimuth_qn, magnetic_qn, spin_idx

    def _dos_create_plot_data_set(
        self, label_mapping, orbital_mapping, dos_data, divisor, plot_type, alpha_idx
    ):
        plot_data = []
        for label, orbitals in dos_data["labels"].items():
            if label in label_mapping:
                color_idx = label_mapping[label]
            else:
                color_idx = max(list(label_mapping.values()) + [-1]) + 1
                label_mapping[label] = color_idx
            for (orb_label, _), dos_idx in orbitals.items():
                data_set = {
                    "x_values": dos_data["energy"],
                    "y_values": dos_data["dos"][dos_idx] / divisor,
                }
                if "line" in plot_type:
                    if orb_label in orbital_mapping:
                        orb_idx = orbital_mapping[orb_label]
                    else:
                        orb_idx = max([val for val in orbital_mapping.values()] + [-1]) + 1
                        orbital_mapping[orb_label] = orb_idx
                    data_set["color"] = color_idx
                    data_set["linestyle"] = orb_idx
                    data_set["linewidth"] = orb_idx
                if "fill" in plot_type:
                    data_set["facecolor"] = color_idx
                    data_set["alpha"] = alpha_idx
                    data_set["use_fill"] = True
                if "line" in plot_type and "fill" in plot_type:
                    aux_ds = {
                        "x_values": dos_data["energy"],
                        "y_values": dos_data["dos"][dos_idx] / divisor,
                    }
                    for attr in ["facecolor", "alpha", "use_fill"]:
                        aux_ds[attr] = data_set.pop(attr)
                    plot_data.append(aux_ds)
                if orb_label == "" or orb_label.startswith(" "):
                    data_set["label"] = label
                else:
                    data_set["legendgrouptitle_text"] = label
                    data_set["legendgroup"] = label
                    data_set["label"] = orb_label + "-orbitals"
                    data_set["group_by"] = "color"
                plot_data.append(data_set)
        return plot_data

    def _sum_pdos(self, pdos_data):
        tdos = np.zeros((2, pdos_data["dos"].shape[1]))
        max_spin = 1
        for orb_labels in pdos_data["labels"].values():
            for orb_l, idx in orb_labels.items():
                if orb_l[1] == 1:
                    max_spin = 2
                tdos[orb_l[1]] += pdos_data["dos"][idx]
        return {
            "energy": pdos_data["energy"],
            "dos": tdos[:max_spin],
            "labels": {"tDOS": {("", idx): idx for idx in range(max_spin)}},
        }
