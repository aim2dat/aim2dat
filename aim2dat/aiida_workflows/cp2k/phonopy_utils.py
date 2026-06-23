"""
Calcfunctions wrapping phonopy for finite-displacement phonon workflows.

This module provides the *forward* half of a finite-displacement phonon
calculation as AiiDA calcfunctions, mirroring the idiom used by
``seekpath_structure_analysis`` in ``aim2dat.aiida_workflows.utils``:

* :func:`phonopy_generate_displacements` builds the supercell and the
  symmetry-reduced set of displaced supercells from an optimized unit cell.
* :func:`parse_cp2k_forces` reads the atomic forces from the retrieved CP2K
  outputs (via phonopy's CP2K interface) and stacks them in displacement order.
* :func:`phonopy_collect_phonons` assembles the force constants from those
  forces and extracts the phonon band structure and DOS.

The *backward* half (force constants -> band structure / DOS / thermal
properties) is delegated to the existing extraction helpers in
``aim2dat.ext_interfaces.phonopy`` so that a single phonopy code path is kept
across the package.

Note on force ingestion: the ``aim2dat.cp2k`` parser does not expose an atomic
forces output port, and CP2K writes the atomic forces into its main output file
(retrieved by default). :func:`parse_cp2k_forces` therefore reads them from the
retrieved output via ``phonopy.interface.cp2k.parse_set_of_forces``, which
handles the CP2K version detection, unit conversion, and drift correction.

Proposed AiiDA workflow entry points (pyproject.toml)::

    [project.entry-points."aiida.workflows"]
    "aim2dat.phonopy.displacements" =
        "aim2dat.aiida_workflows.cp2k.phonopy_utils:phonopy_generate_displacements"
    "aim2dat.phonopy.parse_forces" =
        "aim2dat.aiida_workflows.cp2k.phonopy_utils:parse_cp2k_forces"
    "aim2dat.phonopy.collect" =
        "aim2dat.aiida_workflows.cp2k.phonopy_utils:phonopy_collect_phonons"
"""

# Standard library imports
import os
import tempfile

# Third party library imports
import numpy as np
import aiida.orm as aiida_orm
from aiida.engine import calcfunction
from aiida.plugins import DataFactory
import phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.interface.cp2k import parse_set_of_forces

# Internal library imports -- reuse the existing extraction interface.
from aim2dat.ext_interfaces.phonopy import (
    _extract_band_structure,
    _extract_total_dos,
    _extract_projected_dos,
    _extract_thermal_properties,
)

StructureData = DataFactory("core.structure")

# Default name of the CP2K output file produced by the aim2dat.cp2k calcjob.
# VERIFY against ``ForceWorkChain``'s ``metadata.options.output_filename``.
_CP2K_OUTPUT_FILENAME = "aiida.out"


# --------------------------------------------------------------------------- #
# Converters
# --------------------------------------------------------------------------- #
def _structuredata_to_phonopy_atoms(structure):
    """Convert an AiiDA ``StructureData`` node into a ``PhonopyAtoms`` object."""
    ase_atoms = structure.get_ase()
    return PhonopyAtoms(
        symbols=ase_atoms.get_chemical_symbols(),
        cell=ase_atoms.get_cell().array,
        scaled_positions=ase_atoms.get_scaled_positions(),
    )


def _phonopy_atoms_to_structuredata(ph_atoms):
    """Convert a ``PhonopyAtoms`` object into an AiiDA ``StructureData`` node."""
    from ase import Atoms

    ase_atoms = Atoms(
        symbols=ph_atoms.symbols,
        cell=ph_atoms.cell,
        scaled_positions=ph_atoms.scaled_positions,
        pbc=True,
    )
    return StructureData(ase=ase_atoms)


def _build_phonopy(unitcell, settings):
    """Instantiate a ``Phonopy`` object from a unit cell and settings dict."""
    return phonopy.Phonopy(
        unitcell,
        supercell_matrix=settings["supercell_matrix"],
        primitive_matrix=settings.get("primitive_matrix", "auto"),
        symprec=settings.get("symprec", 1e-5),
        calculator="cp2k",
    )


# --------------------------------------------------------------------------- #
# Calcfunctions
# --------------------------------------------------------------------------- #
@calcfunction
def phonopy_generate_displacements(structure, parameters):
    """
    Generate the symmetry-reduced set of displaced supercells.

    Parameters
    ----------
    structure : aiida.orm.StructureData
        The optimized unit cell. NOTE: to keep the displaced cell consistent
        with the optimization, ``symprec`` should match the ``eps_symmetry``
        used by the upstream ``cell_opt`` task (default 0.005 in the standard
        protocols) rather than phonopy's looser default.
    parameters : aiida.orm.Dict
        Settings dictionary with the keys:
            ``supercell_matrix`` (list[list[int]] | list[int]) - required,
            ``primitive_matrix`` (str | list) - optional, default ``"auto"``,
            ``symprec`` (float) - optional, default ``1e-5``,
            ``displacement`` (float) - optional displacement amplitude in
            Angstrom, default ``0.01``.

    Returns
    -------
    dict
        ``supercell_0000`` ... ``supercell_NNNN`` : aiida.orm.StructureData
            One displaced supercell per displacement, fed to the per-displacement
            force calculations (:class:`ForceWorkChain`, RUN_TYPE ENERGY_FORCE).
        ``phonon_setting_info`` : aiida.orm.Dict
            Displacement dataset + supercell settings, carried forward to
            :func:`parse_cp2k_forces` and :func:`phonopy_collect_phonons`.
    """
    settings = parameters.get_dict()
    unitcell = _structuredata_to_phonopy_atoms(structure)
    phonon = _build_phonopy(unitcell, settings)
    phonon.generate_displacements(distance=settings.get("displacement", 0.01))

    supercells = phonon.supercells_with_displacements

    outputs = {}
    for i, supercell in enumerate(supercells):
        outputs[f"supercell_{i:04d}"] = _phonopy_atoms_to_structuredata(supercell)

    outputs["phonon_setting_info"] = aiida_orm.Dict(
        dict={
            "supercell_matrix": settings["supercell_matrix"],
            "primitive_matrix": settings.get("primitive_matrix", "auto"),
            "symprec": settings.get("symprec", 1e-5),
            "displacement": settings.get("displacement", 0.01),
            "displacement_dataset": phonon.dataset,
            "number_of_displacements": len(supercells),
            "supercell_n_atoms": len(supercells[0]),
        }
    )
    return outputs


@calcfunction
def parse_cp2k_forces(phonon_setting_info, **retrieved):
    """
    Read CP2K atomic forces from the retrieved outputs, in displacement order.

    Reuses ``phonopy.interface.cp2k.parse_set_of_forces`` so CP2K version
    detection, unit conversion, and drift correction match the rest of phonopy.

    Parameters
    ----------
    phonon_setting_info : aiida.orm.Dict
        Output of :func:`phonopy_generate_displacements`.
    **retrieved : aiida.orm.FolderData
        One retrieved folder per displacement, each containing the CP2K output
        file. Keys are sorted to recover the displacement order, so pass them as
        ``supercell_0000=<retrieved>, supercell_0001=<retrieved>, ...``.

    Returns
    -------
    dict
        ``force_sets`` : aiida.orm.ArrayData
            Array ``force_sets`` of shape ``(n_displacements, n_atoms, 3)``.
    """
    n_atoms = phonon_setting_info.get_dict()["supercell_n_atoms"]

    with tempfile.TemporaryDirectory() as tmp:
        filenames = []
        for i, key in enumerate(sorted(retrieved)):
            folder = retrieved[key]
            content = folder.get_object_content(_CP2K_OUTPUT_FILENAME)
            path = os.path.join(tmp, f"force_{i:04d}.out")
            with open(path, "w") as handle:
                handle.write(content)
            filenames.append(path)
        force_sets = parse_set_of_forces(n_atoms, filenames, verbose=False)

    array = aiida_orm.ArrayData()
    array.set_array("force_sets", np.array(force_sets))
    return {"force_sets": array}


@calcfunction
def phonopy_collect_phonons(structure, phonon_setting_info, force_sets, parameters):
    """
    Assemble force constants and extract the phonon band structure and DOS.

    Parameters
    ----------
    structure : aiida.orm.StructureData
        The same optimized unit cell passed to
        :func:`phonopy_generate_displacements`.
    phonon_setting_info : aiida.orm.Dict
        The ``phonon_setting_info`` output of
        :func:`phonopy_generate_displacements`.
    force_sets : aiida.orm.ArrayData
        Array ``force_sets`` of shape ``(n_displacements, n_atoms, 3)``, as
        produced by :func:`parse_cp2k_forces`.
    parameters : aiida.orm.Dict
        Post-processing settings:
            ``band_path`` (list) - q-point path segments (from SeekPath),
            ``band_labels`` (list) - high-symmetry point labels,
            ``band_npoints`` (int) - points per segment, default ``101``,
            ``with_eigenvectors`` (bool) - default ``False``,
            ``dos_mesh`` (list[int]) - q-mesh for the DOS, default ``[20,20,20]``,
            ``thermal_properties`` (bool) - default ``False``,
            ``t_min`` / ``t_max`` / ``t_step`` (float) - thermal range.

    Returns
    -------
    dict
        ``band_structure`` : aiida.orm.Dict - phonopy band-structure dict.
        ``total_dos`` : aiida.orm.XyData - frequency vs. total phonon DOS.
        ``thermal_properties`` : aiida.orm.Dict - present only if requested.
    """
    settings = phonon_setting_info.get_dict()
    p_dict = parameters.get_dict()

    unitcell = _structuredata_to_phonopy_atoms(structure)
    phonon = _build_phonopy(unitcell, settings)
    phonon.dataset = settings["displacement_dataset"]
    phonon.forces = force_sets.get_array("force_sets")
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()

    # --- v2 TODO: non-analytical correction (NAC) -------------------------- #
    # For polar frameworks, set the Born effective charges + dielectric tensor
    # here (computed by CP2K) to capture LO-TO splitting near Gamma:
    #     phonon.nac_params = {"born": ..., "dielectric": ..., "factor": ...}
    # The input spec deliberately reserves a `nac_parameters` slot so adding
    # this in v2 does not break the API.
    # ----------------------------------------------------------------------- #

    outputs = {}
    # Reuse aim2dat's existing extraction interface, which loads from a
    # phonopy_params.yaml. Save the force constants to a temporary file and
    # point `load_parameters` at it.
    #
    # NOTE for review: a small refactor of `ext_interfaces.phonopy._extract_*`
    # to optionally accept a pre-built `Phonopy` object would avoid this file
    # round-trip. Kept file-based here to leave the existing interface
    # untouched for the first PR.
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = os.path.join(tmp, "phonopy_params.yaml")
        phonon.save(filename=yaml_path, settings={"force_constants": True})
        load_parameters = {"phonopy_yaml": yaml_path, "calculator": "cp2k"}

        # Band structure
        band_dict, _primitive_cell = _extract_band_structure(
            load_parameters,
            p_dict["band_path"],
            p_dict["band_labels"],
            p_dict.get("band_npoints", 101),
            p_dict.get("with_eigenvectors", False),
        )
        outputs["band_structure"] = aiida_orm.Dict(dict=_jsonify(band_dict))

        # Total DOS -> XyData
        dos_dict = _extract_total_dos(load_parameters, p_dict.get("dos_mesh", [20, 20, 20]))
        dos = aiida_orm.XyData()
        dos.set_x(np.array(dos_dict["frequency_points"]), "frequency", "THz")
        dos.set_y(np.array(dos_dict["total_dos"]), "total_dos", "states/THz")
        outputs["total_dos"] = dos

        # Optional thermal properties
        if p_dict.get("thermal_properties", False):
            thermal = _extract_thermal_properties(
                load_parameters,
                p_dict.get("dos_mesh", [20, 20, 20]),
                p_dict.get("t_min", 0.0),
                p_dict.get("t_max", 1000.0),
                p_dict.get("t_step", 10.0),
            )
            outputs["thermal_properties"] = aiida_orm.Dict(dict=_jsonify(thermal))

    return outputs


def _jsonify(obj):
    """Recursively convert numpy arrays/scalars to JSON-serializable types.

    phonopy's result dicts contain numpy arrays which AiiDA's ``Dict`` cannot
    store directly.
    """
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj