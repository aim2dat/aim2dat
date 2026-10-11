"""Interface to the phonopy library."""

# Standard library imports
import re

# Third party library imports
import phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.file_IO import read_v_e, read_thermal_properties_yaml


def _read_v_e_file(file_path):
    return read_v_e(file_path)


def _read_thermal_properties_yaml_files(file_paths):
    return read_thermal_properties_yaml(file_paths)


def _extract_structure_from_atoms(atoms):
    """Extract a dictionary with structural parameters from the phonopy atoms object."""
    elements = atoms.symbols
    cell = atoms.cell.tolist()
    positions = atoms.positions.tolist()
    # `permutation_types` doesn't exist on PhonopyAtoms in phonopy 4.2.1 (the
    # version installed here) -- getattr guards against the AttributeError
    # that otherwise crashes every displacement generation. Falls through to
    # the digit-suffix branch below, which is what actually handles kinds for
    # elements with more than one AiiDA kind (see commit 1edd820).
    permutation_types = getattr(atoms, "permutation_types", None)
    if permutation_types is not None:
        kinds = []
        for element, permutation in zip(elements, permutation_types.tolist()):
            kinds.append(f"{element}{permutation}")
    elif any(re.findall(r"\d+", el) for el in elements):
        kinds = []
        for el in elements:
            if re.findall(r"\d+", el):
                kinds.append(el)
            else:
                kinds.append(f"{el}{0}")
    else:
        kinds = None
    elements = [re.findall(r"\D+", el)[0] for el in elements]
    structure_dict = {
        "elements": elements,
        "kinds": kinds,
        "positions": positions,
        "cell": cell,
        "pbc": True,
        "is_cartesian": True,
        "attributes": {},
        "site_attributes": {},
    }
    return structure_dict


def _extract_band_structure(
    load_parameters,
    path,
    path_labels,
    npoints,
    with_eigenvectors,
    pre_load=None,
):
    """Get phonon band structure."""
    qpoints, connections = get_band_qpoints_and_path_connections(path, npoints)
    phonon = __load_phonopy(load_parameters, pre_load)
    phonon.run_band_structure(
        qpoints,
        path_connections=connections,
        labels=path_labels,
        with_eigenvectors=with_eigenvectors,
    )
    return {
        "qpoints": phonon.band_structure.qpoints,
        "frequencies": phonon.band_structure.frequencies,
    }, phonon.primitive.cell


def _extract_projected_dos(load_parameters, mesh, pre_load=None):
    """Get phonon projected DOS."""
    phonon = __load_phonopy(load_parameters, pre_load)
    phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
    phonon.run_projected_dos()
    return {
        "pdos": phonon.projected_dos.projected_dos,
        "frequency_points": phonon.projected_dos.frequency_points,
    }, phonon.primitive.symbols


def _extract_total_dos(load_parameters, mesh, pre_load=None):
    """Get phonon total DOS."""
    phonon = __load_phonopy(load_parameters, pre_load)
    phonon.run_mesh(mesh)
    phonon.run_total_dos()
    return {
        "total_dos": phonon.total_dos.dos,
        "frequency_points": phonon.total_dos.frequency_points,
    }


def _extract_thermal_properties(load_parameters, mesh, t_min, t_max, t_step, pre_load=None):
    """Get thermal properties."""
    phonon = __load_phonopy(load_parameters, pre_load)
    phonon.run_mesh(mesh)
    phonon.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step)
    return {
        "temperatures": phonon.thermal_properties.temperatures,
        "free_energy": phonon.thermal_properties.thermal_properties[1],
        "entropy": phonon.thermal_properties.thermal_properties[2],
        "heat_capacity": phonon.thermal_properties.thermal_properties[3],
    }


def _extract_qha_properties(
    # thermal_properties_file_names=None,
    # ev_file_name=None,
    volumes=None,
    electronic_energies=None,
    temperatures=None,
    free_energy=None,
    cv=None,
    entropy=None,
    pressure=None,
    eos="vinet",
    t_max=None,
):
    """Get quasi-harmonic properties."""
    phonopy_qha = phonopy.PhonopyQHA(
        volumes=volumes,
        electronic_energies=electronic_energies,
        temperatures=temperatures,
        free_energy=free_energy,
        cv=cv,
        entropy=entropy,
        pressure=pressure,
        eos=eos,
        t_max=t_max,
    )
    return {
        "bulk_modulus": phonopy_qha.bulk_modulus,
        "thermal_expansion": phonopy_qha.thermal_expansion,
        "helmholtz_volume": phonopy_qha.helmholtz_volume,
        "volume_temperature": phonopy_qha.volume_temperature,
        "gibbs_temperature": phonopy_qha.gibbs_temperature,
        "bulk_modulus_temperature": phonopy_qha.bulk_modulus_temperature,
        "heat_capacity_P_numerical": phonopy_qha.heat_capacity_P_numerical,
        "heat_capacity_P_polyfit": phonopy_qha.heat_capacity_P_polyfit,
        "gruneisen_temperature": phonopy_qha.gruneisen_temperature,
        "bulk_modulus_parameters": phonopy_qha.get_bulk_modulus_parameters(),
    }


def _create_phonopy_atoms(structure):
    """Create phonopy atoms object from structure dictionary."""
    if not all(structure.pbc):
        raise ValueError("`cell` must be set if `pbc` is set to true for one or more direction.")
    # Kind names are passed to phonopy only when they encode a real
    # distinction (an element with more than one kind). PhonopyAtoms species
    # indices must start at 1, while kind names in the wild are often
    # zero-indexed (e.g. AiiDA kinds "Na0"/"Cl1"), so kinds are re-indexed
    # per element in order of first appearance instead of passed verbatim.
    symbols = structure.elements
    kinds = structure.kinds
    if kinds is not None and all(kinds) and len(set(kinds)) > len(set(structure.elements)):
        kinds_per_element = {}
        for element, kind in zip(structure.elements, kinds):
            element_kinds = kinds_per_element.setdefault(element, [])
            if kind not in element_kinds:
                element_kinds.append(kind)
        symbols = [
            (
                element
                if len(kinds_per_element[element]) == 1
                else f"{element}{kinds_per_element[element].index(kind) + 1}"
            )
            for element, kind in zip(structure.elements, kinds)
        ]
    return PhonopyAtoms(
        symbols=symbols,
        cell=structure.cell,
        positions=structure.positions,
    )


def _build_phonopy(
    unitcell=None, supercell_matrix=None, primitive_matrix=None, symprec=None, calculator=None
):
    """Instantiate a ``Phonopy`` object."""
    return phonopy.Phonopy(
        unitcell=unitcell,
        supercell_matrix=supercell_matrix,
        primitive_matrix=primitive_matrix,
        symprec=symprec,
        calculator=calculator,
    )


def __load_phonopy(load_parameters, pre_load):
    return pre_load if type(pre_load) is phonopy.Phonopy else phonopy.load(**load_parameters)
