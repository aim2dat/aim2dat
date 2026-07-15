"""
AiiDA work chain for finite-displacement phonon calculations with CP2K.

Unlike the single-calculation property work chains (band_structure, pdos, ...),
a phonon calculation runs *N* force calculations -- one per symmetry-reduced
displacement -- so this is a fan-out orchestrator (a plain ``WorkChain``, like
``combined_work_chains``) rather than a ``_BaseCoreWorkChain`` subclass.

Pipeline
--------
1. ``phonopy_generate_displacements`` -> N displaced supercells + the
   undisplaced reference supercell + displacement dataset
2. ``ForceWorkChain`` on the undisplaced reference supercell -> abort early
   (exit code 402) if any force exceeds ``phonopy_p.ref_forces_threshold``:
   the structure is not at a force-free minimum or the numerical setup
   produces spurious forces (basis/grid). Runs *before* the fan-out so no
   compute is wasted on a bad structure/setup. NOTE: on high-symmetry
   crystals where atoms sit on special positions, site symmetry can cancel
   systematic force errors (e.g. egg-box) in the reference cell, so a
   passing check there is necessary but not sufficient.
3. ``ForceWorkChain`` x N            -> CP2K ENERGY_FORCE per displacement
4. ``phonopy_calculate_phonons``     -> phonon band structure + DOS
"""

# Third party library imports
import numpy as np
from aiida.orm import Bool, Str, Int, Float, List, Dict
from aiida.plugins import DataFactory, CalculationFactory, WorkflowFactory
from aiida.engine import WorkChain

phonopy_generate_displacements = CalculationFactory("aim2dat.phonopy.displacements")
phonopy_calculate_phonons = CalculationFactory("aim2dat.phonopy.phonons")
ForceWorkChain = WorkflowFactory("aim2dat.cp2k.forces")

StructureData = DataFactory("core.structure")
XyData = DataFactory("core.array.xy")
BandsData = DataFactory("core.array.bands")
ArrayData = DataFactory("core.array")


class PhononWorkChain(WorkChain):
    """
    AiiDA work chain to compute the phonon band structure and DOS via the
    finite-displacement method with CP2K and phonopy.
    """

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        # --- CP2K settings ------------------------------------------------ #
        spec.input(
            "force.adjust_scf_parameters",
            valid_type=Bool,
            default=lambda: Bool(False),
            help="Restart calculation with adjusted parameters if SCF-cycles are not converged.",
        )
        # --- phonopy settings I ----------------------------------------------- #
        spec.input(
            "phonopy_p.structure", valid_type=StructureData, help="The main input structure."
        )
        spec.input(
            "phonopy_p.supercell_matrix",
            valid_type=List,
            help="Supercell matrix, e.g. [[2,0,0],[0,2,0],[0,0,2]].",
        )
        spec.input(
            "phonopy_p.primitive_matrix",
            valid_type=(List, Str),
            default=lambda: Str("auto"),
            help="Primitive matrix, e.g. ``auto`` or [[2,0,0],[0,2,0],[0,0,2]].",
        )
        spec.input(
            "phonopy_p.symprec",
            valid_type=Float,
            default=lambda: Float(0.005),
            help="Symmetry tolerance; should match the `eps_symmetry` used in the "
            "upstream cell optimization.",
        )
        spec.input(
            "phonopy_p.displacement",
            valid_type=Float,
            default=lambda: Float(0.01),
            help="Displacement amplitude in Angstrom.",
        )
        # --- phonopy settings II ---------------------------------------------- #
        spec.input(
            "phonopy_p.path_parameters",
            valid_type=Dict,
            help="Band path in reciprocal coordinates retrieved from SeekPath.",
        )
        spec.input(
            "phonopy_p.band_npoints",
            valid_type=Int,
            default=lambda: Int(101),
            help="Number of q-points in each path including end points.",
        )
        spec.input(
            "phonopy_p.dos_mesh",
            valid_type=List,
            default=lambda: List(list=[20, 20, 20]),
            help="q-point mesh for the phonon DOS.",
        )
        spec.input(
            "phonopy_p.with_eigenvectors",
            valid_type=Bool,
            required=False,
            help="Whether eigenvectors are calculated.",
        )
        spec.input(
            "phonopy_p.thermal_properties",
            valid_type=Bool,
            required=False,
            help="Also compute thermal properties (free energy, Cv, entropy).",
        )
        spec.input(
            "phonopy_p.temp_range",
            valid_type=List,
            required=False,
            help="Temperature range (start, end, step) for thermal properties.",
        )
        # v2 TODO: non-analytical correction. Reserved now so adding it later
        # does not change the input spec.
        spec.input(
            "phonopy_p.nac_parameters",
            valid_type=Dict,
            required=False,
            help="Reserved (v2): Born effective charges + dielectric tensor for the "
            "non-analytical correction. Not used in v1.",
        )
        # --- reference-forces sanity gate ------------------------------------ #
        spec.input(
            "phonopy_p.ref_forces_check",
            valid_type=Bool,
            default=lambda: Bool(True),
            help="First run a force calculation on the undisplaced reference "
            "supercell and abort (before the displacement fan-out) if any force "
            "exceeds `ref_forces_threshold`. Non-zero reference forces mean an "
            "under-relaxed input structure or numerical force noise (grid/basis) "
            "-- either corrupts the force constants far more subtly downstream "
            "than failing here.",
        )
        spec.input(
            "phonopy_p.ref_forces_threshold",
            valid_type=Float,
            default=lambda: Float(5.0e-5),
            help="Maximum tolerated force norm (Ha/Bohr) on the reference "
            "supercell. Default sits between a converged cell-opt residual "
            "(~2e-5) and the smallest displacement force signal (~3e-4 for a "
            "0.01 A displacement).",
        )
        spec.input(
            "phonopy_p.subtract_ref_forces",
            valid_type=Bool,
            default=lambda: Bool(False),
            help="Subtract the reference-supercell forces from every displaced "
            "force set before building force constants (mitigates systematic "
            "residual forces; no substitute for a sound numerical setup).",
        )

        spec.output("phonon_bands", valid_type=BandsData)
        spec.output("phonon_dos", valid_type=XyData)
        spec.output("thermal_properties", valid_type=XyData, required=False)
        spec.output(
            "ref_forces",
            valid_type=ArrayData,
            required=False,
            help="Forces on the undisplaced reference supercell (Ha/Bohr); "
            "~zero for a relaxed structure and a numerically sound setup.",
        )

        # --- per-displacement force calculation ----------------------------- #
        # Exposes the CP2K code, numerical_p and SCF settings of ForceWorkChain;
        # the structure is supplied per displacement.
        spec.expose_inputs(ForceWorkChain, namespace="force", exclude=("structural_p.structure",))
        # namespace_options required=False: ForceWorkChain runs as a fan-out (N
        # per-displacement instances, run_forces/inspect_forces), so there is
        # no single coherent "force" output to collapse them into -- nothing
        # in the outline ever populates this namespace (per-displacement
        # outputs are consumed internally to build force_sets). Without this,
        # the workchain always exits 11 ("did not register a required
        # output") despite phonon_bands/phonon_dos/thermal_properties/
        # ref_forces all being set correctly (confirmed 2026-07-07).
        spec.expose_outputs(
            ForceWorkChain,
            namespace="force",
            namespace_options={"required": False},
        )

        spec.outline(
            cls.run_displacements,
            cls.run_ref_force,
            cls.inspect_ref_force,
            cls.run_forces,
            cls.inspect_forces,
            cls.run_phonopy,
            cls.post_processing,
        )

        spec.exit_code(
            401,
            "ERROR_FORCE_CALCULATION",
            message="One or more displacement force calculations did not finish OK.",
        )
        spec.exit_code(
            402,
            "ERROR_REFERENCE_FORCES",
            message="Forces on the undisplaced reference supercell exceed "
            "`phonopy_p.ref_forces_threshold`: the input structure is not at a "
            "force-free minimum or the numerical setup produces spurious forces "
            "(check basis set / grid cutoff). Force constants built on top of "
            "this would be unreliable.",
        )

    def run_displacements(self):
        """Generate displaced supercells."""
        structure = self.inputs.phonopy_p.structure
        supercell_matrix = self.inputs.phonopy_p.supercell_matrix.get_list()
        if isinstance(self.inputs.phonopy_p.primitive_matrix, List):
            primitive_matrix = self.inputs.phonopy_p.primitive_matrix.get_list()
        elif isinstance(self.inputs.phonopy_p.primitive_matrix, Str):
            primitive_matrix = self.inputs.phonopy_p.primitive_matrix.value
        symprec = self.inputs.phonopy_p.symprec.value
        displacement = self.inputs.phonopy_p.displacement.value

        phonopy_parameters = Dict(
            dict={
                "supercell_matrix": supercell_matrix,
                "primitive_matrix": primitive_matrix,
                "symprec": symprec,
                "calculator": "cp2k",
            }
        )
        parameters = Dict(
            dict={
                "displacement": displacement,
            }
        )
        generated = phonopy_generate_displacements(structure, phonopy_parameters, parameters)
        self.ctx.displacement_dataset = generated.pop("displacement_dataset")
        self.ctx.supercell_ref = generated.pop("supercell_ref")
        self.ctx.supercells = dict(generated)  # supercell_XXXX -> StructureData
        self.ctx.run_ref = (
            self.inputs.phonopy_p.ref_forces_check.value
            or self.inputs.phonopy_p.subtract_ref_forces.value
        )
        self.report(f"Generated {len(self.ctx.supercells)} displaced supercells.")

    def run_ref_force(self):
        """Run the reference (undisplaced) supercell force calculation first.

        Gating on the reference before fanning out the displacements avoids
        burning N force calculations on a structure that is not force-free or
        a numerical setup that produces spurious forces.
        """
        if not self.ctx.run_ref:
            return
        inputs = self.exposed_inputs(ForceWorkChain, namespace="force")
        inputs.structural_p.structure = self.ctx.supercell_ref
        running = self.submit(ForceWorkChain, **inputs)
        self.report(f"Launching ForceWorkChain <{running.pk}> for the reference supercell.")
        self.to_context(force_supercell_ref=running)

    def inspect_ref_force(self):
        """Gate on the reference forces before running the displacements."""
        if not self.ctx.run_ref:
            return
        wc = self.ctx.force_supercell_ref
        if not wc.is_finished_ok:
            self.report(f"ForceWorkChain for the reference supercell failed ({wc.exit_status}).")
            return self.exit_codes.ERROR_FORCE_CALCULATION
        self.ctx.ref_forces = wc.outputs.cp2k.output_forces
        self.out("ref_forces", self.ctx.ref_forces)
        max_force = float(
            np.max(np.linalg.norm(self.ctx.ref_forces.get_array("forces"), axis=-1))
        )
        threshold = self.inputs.phonopy_p.ref_forces_threshold.value
        self.report(
            f"Reference supercell max |F| = {max_force:.3e} Ha/Bohr "
            f"(threshold {threshold:.3e})."
        )
        if self.inputs.phonopy_p.ref_forces_check.value and max_force > threshold:
            return self.exit_codes.ERROR_REFERENCE_FORCES

    def run_forces(self):
        """Fan out one ForceWorkChain per displaced supercell."""
        for key in sorted(self.ctx.supercells):
            inputs = self.exposed_inputs(ForceWorkChain, namespace="force")
            inputs.structural_p.structure = self.ctx.supercells[key]
            running = self.submit(ForceWorkChain, **inputs)
            self.report(f"Launching ForceWorkChain <{running.pk}> for {key}.")
            self.to_context(**{f"force_{key}": running})

    def inspect_forces(self):
        """Verify every force calculation finished OK."""
        for key in sorted(self.ctx.supercells):
            wc = self.ctx[f"force_{key}"]
            if not wc.is_finished_ok:
                self.report(f"ForceWorkChain for {key} failed ({wc.exit_status}).")
                return self.exit_codes.ERROR_FORCE_CALCULATION

    def run_phonopy(self):
        """Collect forces and assemble the band structure + DOS."""
        structure = self.inputs.phonopy_p.structure
        supercell_matrix = self.inputs.phonopy_p.supercell_matrix.get_list()
        if isinstance(self.inputs.phonopy_p.primitive_matrix, List):
            primitive_matrix = self.inputs.phonopy_p.primitive_matrix.get_list()
        elif isinstance(self.inputs.phonopy_p.primitive_matrix, Str):
            primitive_matrix = self.inputs.phonopy_p.primitive_matrix.value
        symprec = self.inputs.phonopy_p.symprec.value

        displacement_dataset = self.ctx.displacement_dataset.get_dict()

        force_list = []
        for key in sorted(self.ctx.supercells):
            force_list.append(
                self.ctx[f"force_{key}"].outputs.cp2k.output_forces.get_array("forces")
            )
        force_sets = np.array(force_list)
        if force_sets.ndim == 4:  # parser may stack as (1, n_atoms, 3) per calc
            force_sets = force_sets.reshape(force_sets.shape[0], *force_sets.shape[-2:])
        if self.inputs.phonopy_p.subtract_ref_forces.value:
            ref = self.ctx.ref_forces.get_array("forces")
            force_sets = force_sets - np.asarray(ref).reshape(1, *force_sets.shape[-2:])

        path_parameters = self.inputs.phonopy_p.path_parameters.get_dict()
        band_path, band_labels = _seekpath_to_phonopy_path(path_parameters)

        band_npoints = self.inputs.phonopy_p.band_npoints.value
        dos_mesh = self.inputs.phonopy_p.dos_mesh.get_list()

        phonopy_parameters = Dict(
            dict={
                "supercell_matrix": supercell_matrix,
                "primitive_matrix": primitive_matrix,
                "symprec": symprec,
                "calculator": "cp2k",
            }
        )
        parameters = {
            "displacement_dataset": displacement_dataset,
            # tolist(): keep the Dict JSON-serializable for provenance storage.
            "force_sets": force_sets.tolist(),
            "band_path": band_path,
            "band_labels": band_labels,
            "band_npoints": band_npoints,
            "dos_mesh": dos_mesh,
        }

        if "with_eigenvectors" in self.inputs.phonopy_p:
            with_eigenvectors = self.inputs.phonopy_p.with_eigenvectors.value
            parameters.update({"with_eigenvectors": with_eigenvectors})
        if "thermal_properties" in self.inputs.phonopy_p:
            thermal_properties = self.inputs.phonopy_p.thermal_properties.value
            temp_range = self.inputs.phonopy_p.temp_range.get_list()
            parameters.update(
                {
                    "thermal_properties": thermal_properties,
                    "temp_range": temp_range,
                }
            )

        parameters = Dict(dict=parameters)

        self.ctx.results = phonopy_calculate_phonons(
            structure,
            phonopy_parameters,
            parameters,
        )

    def post_processing(self):
        """Expose the outputs."""
        self.out("phonon_bands", self.ctx.results["band_structure"])
        self.out("phonon_dos", self.ctx.results["total_dos"])
        if "thermal_properties" in self.ctx.results:
            self.out("thermal_properties", self.ctx.results["thermal_properties"])


def _seekpath_to_phonopy_path(path_parameters):
    """Convert SeekPath ``path_parameters`` into a phonopy band path + labels.

    NOTE for review/validation: the band path is defined in the reciprocal
    coordinates of SeekPath's primitive cell, and phonopy is run on that same
    primitive cell (``primitive_matrix='auto'``). For low-symmetry frameworks
    the primitive-cell conventions should be cross-checked against the
    script-pipeline ground truth (e.g. ZIF-8) -- this is the main correctness
    point to validate.
    """
    point_coords = path_parameters["point_coords"]
    band_path, band_labels = [], []
    for segment in path_parameters["path"]:
        band_path.append([point_coords[segment[0]], point_coords[segment[1]]])
        band_labels.extend([segment[0], segment[1]])
    return band_path, band_labels
