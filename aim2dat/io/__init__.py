"""Subpackage containing functions to read or write files."""

from aim2dat.io.cp2k.bands_dos import read_cp2k_band_structure, read_cp2k_proj_dos
from aim2dat.io.cp2k.restart import read_cp2k_restart_structure
from aim2dat.io.cp2k.stdout import read_cp2k_stdout
from aim2dat.io.cif import read_cif_file
from aim2dat.io.critic2 import read_critic2_stdout, read_critic2_plane
from aim2dat.io.fhi_aims import (
    read_fhiaims_band_structure,
    read_fhiaims_total_dos,
    read_fhiaims_proj_dos,
)
from aim2dat.io.hdf5 import read_hdf5_structure, write_hdf5_structure
from aim2dat.io.phonopy import (
    read_phonopy_band_structure,
    read_phonopy_total_dos,
    read_phonopy_proj_dos,
    read_phonopy_thermal_properties,
    read_phonopy_qha_properties,
)
from aim2dat.io.qe import (
    read_qe_xml,
    read_qe_input_structure,
    read_qe_band_structure,
    read_qe_total_dos,
    read_qe_proj_dos,
)
from aim2dat.io.xmgrace import read_xmgrace_file, read_xmgrace_band_structure
from aim2dat.io.yaml import read_yaml_file, write_yaml_file
from aim2dat.io.zeo import write_zeo_file


__all__ = [
    "read_cp2k_band_structure",
    "read_cp2k_proj_dos",
    "read_cp2k_restart_structure",
    "read_cp2k_optimized_structure",
    "read_cp2k_stdout",
    "read_cif_file",
    "read_critic2_stdout",
    "read_critic2_plane",
    "read_fhiaims_band_structure",
    "read_fhiaims_total_dos",
    "read_fhiaims_proj_dos",
    "read_hdf5_structure",
    "write_hdf5_structure",
    "read_phonopy_band_structure",
    "read_phonopy_total_dos",
    "read_phonopy_proj_dos",
    "read_phonopy_thermal_properties",
    "read_phonopy_qha_properties",
    "read_qe_xml",
    "read_qe_input_structure",
    "read_qe_band_structure",
    "read_qe_total_dos",
    "read_qe_proj_dos",
    "read_xmgrace_file",
    "read_xmgrace_band_structure",
    "read_yaml_file",
    "write_yaml_file",
    "write_zeo_file",
]
