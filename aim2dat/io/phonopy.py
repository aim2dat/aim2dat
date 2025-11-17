"""
Module of functions to read output-files from phonopy.
"""

# Standard library imports
from typing import Tuple, Union

# Internal library imports
import aim2dat.io.utils as io_utils
from aim2dat.ext_interfaces import _return_ext_interface_modules


def _create_load_parameters(
    parameter_file_name, force_sets_file_name, force_constants_file_name, phonopy_kwargs
):
    load_parameters = {
        "phonopy_yaml": parameter_file_name,
    }
    if force_sets_file_name is not None:
        load_parameters["force_sets_filename"] = force_sets_file_name
    elif force_constants_file_name is not None:
        load_parameters["force_constants_filename"] = force_constants_file_name
    else:
        raise ValueError("`force_sets_file_path` or `force_constants_file_path` needs to be set.")
    if phonopy_kwargs is not None:
        load_parameters.update(phonopy_kwargs)
    return load_parameters


@io_utils.read_band_structure(r".*\.ya?ml$")
def read_phonopy_band_structure(
    file_path: str,
    kpath: list,
    npoints: int,
    force_sets_file_path: str = None,
    force_constants_file_path: str = None,
    path_labels: list = None,
    phonopy_kwargs: dict = None,
) -> Tuple[dict, list]:
    """
    Read band structure using phonopy.

    Parameters
    ------------
    file_path : str
        Phonopy yaml-file containing all settings and parameters.
    kpath : list
        Nested list of high-symmetry points.
    npoints : int
        Nr of points on the k-path.
    force_sets_file_path : str or None
        Force sets file. If set to ``None``, ``force_constants_file_path`` needs to be set.
    force_constants_file_path : str or None
        Force constants file. If set to ``None``, ``force_sets_file_path`` needs to be set.
    path_labels : list or None
        List of labels for the high-symmetry points specified in ``kpath``.
    phonopy_kwargs : dict
        Additional keywords for ``phonopy.load``.

    Returns
    -------
    output_bands : dict
        Dictionary containing the k-path and the eigenvalues.
    reference_cell : list
        Reference cell.
    """
    load_parameters = _create_load_parameters(
        file_path, force_sets_file_path, force_constants_file_path, phonopy_kwargs
    )
    ph_bands, reference_cell = _return_ext_interface_modules("phonopy")._extract_band_structure(
        load_parameters, kpath, path_labels, npoints, False
    )
    output_bands = {
        "kpoints": [],
        "unit_y": "THz",
        "bands": [],
        "path_labels": [],
    }

    counter_points = 0
    counter_seg = 0
    path_start = 0
    for cont_idx, cont_segment in enumerate(kpath):
        path_end = path_start + len(cont_segment) - 1
        for qpoints, band_seg in zip(
            ph_bands["qpoints"][path_start:path_end], ph_bands["frequencies"][path_start:path_end]
        ):
            output_bands["path_labels"].append((counter_points, path_labels[counter_seg]))
            counter_seg += 1
            counter_points += npoints - 1
            output_bands["kpoints"] += [
                [float(pt[0]), float(pt[1]), float(pt[2])] for pt in qpoints
            ]
            output_bands["bands"] += [[float(ev) for ev in pt] for pt in band_seg]
            output_bands["path_labels"].append((counter_points, path_labels[counter_seg]))
            counter_points += 1
        path_start = path_end
        counter_seg += 1
    return output_bands, reference_cell.tolist()


@io_utils.read_total_dos(r".*\.ya?ml$")
def read_phonopy_total_dos(
    file_path: str,
    mesh: Union[int, list] = 100,
    force_sets_file_path: str = None,
    force_constants_file_path: str = None,
    phonopy_kwargs: dict = None,
) -> dict:
    """
    Read the total density of phonon states from phonopy.

    Parameters
    ----------
    file_path : str
        Phonopy yaml-file containing all settings and parameters.
    mesh : list
        Number of points in each dimension.
    force_sets_file_path : str or None
        Force sets file. If set to ``None``, ``force_constants_file_path`` needs to be set.
    force_constants_file_path : str or None
        Force constants file. If set to ``None``, ``force_sets_file_path`` needs to be set.
    phonopy_kwargs : dict
        Additional keywords for ``phonopy.load``.

    Returns
    -------
    dict
        Dictionary containing the total density of states.
    """
    load_parameters = _create_load_parameters(
        file_path, force_sets_file_path, force_constants_file_path, phonopy_kwargs
    )
    phonopy_tdos = _return_ext_interface_modules("phonopy")._extract_total_dos(
        load_parameters, mesh
    )
    return {
        "energy": phonopy_tdos["frequency_points"].tolist(),
        "tdos": phonopy_tdos["total_dos"].tolist(),
        "unit_x": "states/THz/cell",
    }


@io_utils.read_proj_dos(r".*\.ya?ml$")
def read_phonopy_proj_dos(
    file_path: str,
    mesh: Union[int, list] = 100,
    force_sets_file_path: str = None,
    force_constants_file_path: str = None,
    phonopy_kwargs: dict = None,
) -> dict:
    """
    Read the atom projected density of phonon states from phonopy.

    Parameters
    ----------
    file_path : str
        Phonopy yaml-file containing all settings and parameters.
    mesh : list
        Number of points in each dimension.
    force_sets_file_path : str or None
        Force sets file. If set to ``None``, ``force_constants_file_path`` needs to be set.
    force_constants_file_path : str or None
        Force constants file. If set to ``None``, ``force_sets_file_path`` needs to be set.
    phonopy_kwargs : dict
        Additional keywords for ``phonopy.load``.

    Returns
    -------
    dict
        Dictionary containing the projected density of states for each kind.
    """
    load_parameters = _create_load_parameters(
        file_path, force_sets_file_path, force_constants_file_path, phonopy_kwargs
    )
    phonopy_pdos, symbols = _return_ext_interface_modules("phonopy")._extract_projected_dos(
        load_parameters, mesh
    )
    # TODO: check equivalent sites?
    pdos = {
        "energy": phonopy_pdos["frequency_points"].tolist(),
        "unit_x": "states/THz/cell",
        "pdos": [
            {"kind": symbol, "total": pdos.tolist()}
            for symbol, pdos in zip(symbols, phonopy_pdos["projected_dos"])
        ],
    }
    return pdos


def read_phonopy_thermal_properties(
    file_path: str,
    mesh: Union[int, list] = 100,
    t_min: int = 0,
    t_max: int = 1000,
    t_step: int = 10,
    force_sets_file_path: str = None,
    force_constants_file_path: str = None,
    phonopy_kwargs: dict = None,
):
    """
    Read the thermal properties from phonopy.

    Parameters
    ------------
    file_path : str
        Phonopy yaml-file containing all settings and parameters.
    mesh : list
        Number of points in each dimension.
    t_min : int
        Minimum temperature
    t_max : int
        Maximum temperature
    force_sets_file_path : str or None
        Force sets file. If set to ``None``, ``force_constants_file_path`` needs to be set.
    force_constants_file_path : str or None
        Force constants file. If set to ``None``, ``force_sets_file_path`` needs to be set.
    phonopy_kwargs : dict
        Additional keywords for ``phonopy.load``.

    Returns
    -------
    dict
         Dictionary containing the thermal properties.
    """
    load_parameters = _create_load_parameters(
        file_path, force_sets_file_path, force_constants_file_path, phonopy_kwargs
    )
    thermal_properties = _return_ext_interface_modules("phonopy")._extract_thermal_properties(
        load_parameters, mesh, t_min, t_max, t_step
    )
    return {
        "temperatures": thermal_properties["temperatures"].tolist(),
        "free_energy": thermal_properties["free_energy"].tolist(),
        "entropy": thermal_properties["entropy"].tolist(),
        "heat_capacity": thermal_properties["heat_capacity"].tolist(),
    }


def read_phonopy_qha_properties(
    calculation_folder_paths: list = None,
    thermal_properties_file_paths: list = None,
    ev_file_path: str = None,
    stdout_file_path: str = None,
    mesh: Union[int, list] = 100,
    t_min: int = 0,
    t_max: int = 1000,
    t_step: int = 10,
    phonopy_kwargs: dict = None,
):
    """
    Read the outputs from a quasi-harmonic approximation calculation.

    Parameters
    ----------
    calculation_folder_paths : list or None
        Folders containing the calculations of different volumes. If ``None``
        ``thermal_properties_file_paths`` needs to be set instead.
    thermal_properties_file_paths : list or None
        List of thermal properties yaml files generated by phonopy.
    thermal_properties_file_paths : str or None
        File containing volumes and energies.
    ev_file_path : str or None
        Path to the e-V file.
    stdout_file_path : str or None
        File name of the standard output file.
    mesh : list or int
        Number of points in each dimension.
    t_min : int
        Minimum temperature
    t_max : int
        Maximum temperature
    phonopy_kwargs : dict
        Additional keywords for ``phonopy.load``.

    Returns
    -------
    dict
        Dictionary containing the qha properties.
    """
    qha_kwargs = {}
    if ev_file_path is None and (calculation_folder_paths is None and stdout_file_path is None):
        raise ValueError(
            "Either `ev_file_path` or `calculation_folder_paths` and `stdout_file_path` need to "
            "be set."
        )
    if thermal_properties_file_paths is None and calculation_folder_paths is None:
        raise ValueError(
            "Either `thermal_properties_file_paths` or `calculation_folder_paths` needs to be set."
        )
    if thermal_properties_file_paths is not None:
        if not isinstance(thermal_properties_file_paths, (list, tuple)):
            raise TypeError("`thermal_properties_file_paths` needs to be of type list or tuple.")
        (
            qha_kwargs["temperatures"],
            qha_kwargs["cv"],
            qha_kwargs["entropy"],
            qha_kwargs["free_energy"],
            num_modes,
            num_integrated_modes,
        ) = _return_ext_interface_modules("phonopy")._read_thermal_properties_yaml_files(
            thermal_properties_file_paths
        )
    if ev_file_path is not None:
        qha_kwargs["volumes"], qha_kwargs["electronic_energies"] = _return_ext_interface_modules(
            "phonopy"
        )._read_v_e_file(ev_file_path)
    if calculation_folder_paths is not None:
        if thermal_properties_file_paths is None:
            qha_kwargs["cv"] = [[] for idx0 in range(int(t_min), int(t_max + t_step), int(t_step))]
            qha_kwargs["entropy"] = [
                [] for idx0 in range(int(t_min), int(t_max + t_step), int(t_step))
            ]
            qha_kwargs["free_energy"] = [
                [] for idx0 in range(int(t_min), int(t_max + t_step), int(t_step))
            ]
            for calc_folder in calculation_folder_paths:
                t_p = read_phonopy_thermal_properties(
                    calc_folder + "/phonopy_disp.yaml",
                    force_sets_file_path=calc_folder + "/FORCE_SETS",
                    mesh=mesh,
                    t_min=t_min,
                    t_max=t_max,
                    t_step=t_step,
                    phonopy_kwargs=phonopy_kwargs,
                )
                qha_kwargs["temperatures"] = t_p["temperatures"]
                for idx0 in range(len(t_p["temperatures"])):
                    qha_kwargs["cv"][idx0].append(t_p["heat_capacity"][idx0])
                    qha_kwargs["entropy"][idx0].append(t_p["entropy"][idx0])
                    qha_kwargs["free_energy"][idx0].append(t_p["free_energy"][idx0])
        if ev_file_path is None:
            raise ValueError(
                "Parsing volumes and energies from output files is not yet supported."
            )
    qha_properties = _return_ext_interface_modules("phonopy")._extract_qha_properties(**qha_kwargs)
    qha_properties["temperatures"] = [
        float(val)
        for val in qha_kwargs["temperatures"][: len(qha_properties["thermal_expansion"])]
    ]
    qha_properties["bulk_modulus"] = float(qha_properties["bulk_modulus"])
    qha_properties["thermal_expansion"] = [
        float(val) for val in qha_properties["thermal_expansion"]
    ]
    qha_properties["helmholtz_volume"] = qha_properties["helmholtz_volume"].tolist()
    qha_properties["gibbs_temperature"] = qha_properties["gibbs_temperature"].tolist()
    qha_properties["bulk_modulus_temperature"] = qha_properties[
        "bulk_modulus_temperature"
    ].tolist()
    qha_properties["volume_temperature"] = qha_properties["volume_temperature"].tolist()
    qha_properties["heat_capacity_P_numerical"] = [
        float(val) for val in qha_properties["heat_capacity_P_numerical"]
    ]
    qha_properties["heat_capacity_P_polyfit"] = [
        float(val) for val in qha_properties["heat_capacity_P_polyfit"]
    ]
    qha_properties["gruneisen_temperature"] = [
        float(val) for val in qha_properties["gruneisen_temperature"]
    ]
    qha_properties["bulk_modulus_parameters"] = [
        float(val) for val in qha_properties["bulk_modulus_parameters"]
    ]
    qha_properties["volumes"] = qha_kwargs["volumes"].tolist()
    return qha_properties


def read_band_structure(
    parameter_file_name,
    path,
    npoints,
    force_sets_file_name=None,
    force_constants_file_name=None,
    path_labels=None,
    phonopy_kwargs=None,
):
    """
    Read band structure using phonopy.

    Notes
    -----
        This function is deprecated and will be removed, please use
        `aim2dat.io.read_phonopy_band_structure` instead.

    Parameters
    ------------
    parameter_file_name : str
        Phonopy yaml-file containing all settings and parameters.
    path : list
        Nested list of high-symmetry points.
    force_sets_file_name : str or None
        Force sets file. If set to ``None`` force_constants_filename needs to be set.
    force_constants_file_name : str or None
        Force constants file. If set to ``None`` force_sets_file_name needs to be set.
    path_labels : list or None
        List of labels for the high-symmetry points specified in ``path``.
    phonopy_kwargs : dict
        Additional keywords for ``phonopy.load``.

    Returns
    -------
    output_bands : dict
        Dictionary containing the k-path and the eigenvalues.
    reference_cell : list
        Reference cell.
    """
    from warnings import warn

    warn(
        "This function will be removed, please use `aim2dat.io.read_phonopy_band_structure` "
        "instead.",
        DeprecationWarning,
        2,
    )
    return read_phonopy_band_structure(
        file_path=parameter_file_name,
        kpath=path,
        npoints=npoints,
        force_sets_file_path=force_sets_file_name,
        force_constants_file_path=force_constants_file_name,
        path_labels=path_labels,
        phonopy_kwargs=phonopy_kwargs,
    )


def read_total_density_of_states(
    parameter_file_name,
    mesh=100,
    force_sets_file_name=None,
    force_constants_file_name=None,
    phonopy_kwargs=None,
):
    """
    Read the total density of phonon states from phonopy.

    Notes
    -----
        This function is deprecated and will be removed, please use
        `aim2dat.io.read_phonopy_total_dos` instead.

    Parameters
    ----------
    parameter_file_name : str
        Phonopy yaml-file containing all settings and parameters.
    mesh : list
        Number of points in each dimension.
    force_sets_file_name : str or None
        Force sets file. If set to ``None`` force_constants_filename needs to be set.
    force_constants_file_name : str or None
        For constants file. If set to ``None`` force_sets_file_name needs to be set.
    phonopy_kwargs : dict
        Additional keywords for ``phonopy.load``.

    Returns
    -------
    dict
        Dictionary containing the total density of states.
    """
    from warnings import warn

    warn(
        "This function will be removed, please use `aim2dat.io.read_phonopy_total_dos` "
        "instead.",
        DeprecationWarning,
        2,
    )
    return read_phonopy_total_dos(
        file_path=parameter_file_name,
        mesh=mesh,
        force_sets_file_path=force_sets_file_name,
        force_constants_file_path=force_constants_file_name,
        phonopy_kwargs=phonopy_kwargs,
    )


def read_atom_proj_density_of_states(
    parameter_file_name,
    mesh=100,
    force_sets_file_name=None,
    force_constants_file_name=None,
    phonopy_kwargs=None,
):
    """
    Read the atom projected density of phonon states from phonopy.

    Notes
    -----
        This function is deprecated and will be removed, please use
        `aim2dat.io.read_phonopy_proj_dos` instead.

    Parameters
    ----------
    parameter_file_name : str
        Phonopy yaml-file containing all settings and parameters.
    mesh : list
        Number of points in each dimension.
    force_sets_file_name : str or None
        Force sets file. If set to ``None`` force_constants_filename needs to be set.
    force_constants_file_name : str or None
        For ceonstants file. If set to ``None`` force_sets_file_name needs to be set.
    phonopy_kwargs : dict
        Additional keywords for ``phonopy.load``.

    Returns
    -------
    dict
        Dictionary containing the projected density of states for each kind.
    """
    from warnings import warn

    warn(
        "This function will be removed, please use `aim2dat.io.read_phonopy_proj_dos` " "instead.",
        DeprecationWarning,
        2,
    )
    return read_phonopy_proj_dos(
        file_path=parameter_file_name,
        mesh=mesh,
        force_sets_file_path=force_sets_file_name,
        force_constants_file_path=force_constants_file_name,
    )


def read_thermal_properties(
    parameter_file_name,
    mesh=100,
    t_min=0,
    t_max=1000,
    t_step=10,
    force_sets_file_name=None,
    force_constants_file_name=None,
    phonopy_kwargs={},
):
    """
    Read the thermal properties from phonopy.

    Notes
    -----
        This function is deprecated and will be removed, please use
        `aim2dat.io.read_phonopy_thermal_properties` instead.

    Parameters
    ------------
    parameter_file_name : str
        Phonopy yaml-file containing all settings and parameters.
    mesh : list
        Number of points in each dimension.
    t_min : int
        Minimum temperature
    t_max : int
        Maximum temperature
    force_sets_file_name : str or None
        Force sets file. If set to ``None`` force_constants_filename needs to be set.
    force_constants_file_name : str or None
        For ceonstants file. If set to ``None`` force_sets_file_name needs to be set.
    phonopy_kwargs : dict
        Additional keywords for ``phonopy.load``.

    Returns
    -------
    dict
         Dictionary containing the thermal properties.
    """
    from warnings import warn

    warn(
        "This function will be removed, please use `aim2dat.io.read_phonopy_thermal_properties` "
        "instead.",
        DeprecationWarning,
        2,
    )
    return read_phonopy_thermal_properties(
        file_path=parameter_file_name,
        mesh=mesh,
        t_min=t_min,
        t_max=t_max,
        t_step=t_step,
        force_sets_file_path=force_sets_file_name,
        force_constants_file_path=force_constants_file_name,
        phonopy_kwargs=phonopy_kwargs,
    )


def read_qha_properties(
    calculation_folders=None,
    thermal_properties_file_names=None,
    ev_file_name=None,
    std_output_file_name=None,
    mesh=100,
    t_min=0,
    t_max=1000,
    t_step=10,
    phonopy_kwargs=None,
):
    """
    Read the outputs from a quasi-harmonic approximation calculation.

    Notes
    -----
        This function is deprecated and will be removed, please use
        `aim2dat.io.read_phonopy_qha_properties` instead.

    Parameters
    ----------
    calculation_folders : list or None
        Folders containing the calculations of different volumes. If ``None``
        ``thermal_properties_file_names`` needs to be set instead.
    thermal_properties_file_names : list or None
        List of thermal properties yaml files generated by phonopy.
    ev_file_name : str or None
        File containing volumes and energies.
    std_output_file_name : str or None
        File name of the standard output files.
    mesh : list or int
        Number of points in each dimension.
    t_min : int
        Minimum temperature
    t_max : int
        Maximum temperature
    phonopy_kwargs : dict
        Additional keywords for ``phonopy.load``.

    Returns
    -------
    dict
        Dictionary containing the qha properties.
    """
    from warnings import warn

    warn(
        "This function will be removed, please use `aim2dat.io.read_phonopy_qha_properties` "
        "instead.",
        DeprecationWarning,
        2,
    )
    return read_phonopy_qha_properties(
        calculation_folder_paths=calculation_folders,
        thermal_properties_file_paths=thermal_properties_file_names,
        ev_file_path=ev_file_name,
        stdout_file_path=std_output_file_name,
        mesh=mesh,
        t_min=t_min,
        t_max=t_max,
        t_step=t_step,
        phonopy_kwargs=phonopy_kwargs,
    )
