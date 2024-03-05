"""Interface to the phonopy library."""

# Third party library imports
import phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.file_IO import read_v_e, read_thermal_properties_yaml


def _read_v_e_file(file_path):
    return read_v_e(file_path)


def _read_thermal_properties_yaml_files(file_paths):
    return read_thermal_properties_yaml(file_paths)


def _extract_band_structure(load_parameters, path, path_labels, npoints, with_eigenvectors):
    """Get phonon band structure."""
    qpoints, connections = get_band_qpoints_and_path_connections(path, npoints)
    phonon = phonopy.load(**load_parameters)
    phonon.run_band_structure(
        qpoints,
        path_connections=connections,
        labels=path_labels,
        with_eigenvectors=with_eigenvectors,
    )
    return phonon.get_band_structure_dict(), phonon.primitive.cell


def _extract_projected_dos(load_parameters, mesh):
    """Get phonon projected DOS."""
    phonon = phonopy.load(**load_parameters)
    phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
    phonon.run_projected_dos()
    return phonon.get_projected_dos_dict(), phonon.primitive.symbols


def _extract_total_dos(load_parameters, mesh):
    """Get phonon total DOS."""
    phonon = phonopy.load(**load_parameters)
    phonon.run_mesh(mesh)
    phonon.run_total_dos()
    return phonon.get_total_dos_dict()


def _extract_thermal_properties(load_parameters, mesh, t_min, t_max, t_step):
    """Get thermal properties."""
    phonon = phonopy.load(**load_parameters)
    phonon.run_mesh(mesh)
    phonon.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step)
    return phonon.get_thermal_properties_dict()


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
