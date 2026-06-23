"""
AiiDA work chain for finite-displacement phonon calculations with CP2K.

Unlike the single-calculation property work chains (band_structure, pdos, ...),
a phonon calculation runs *N* force calculations -- one per symmetry-reduced
displacement -- so this is a fan-out orchestrator (a plain ``WorkChain``, like
``combined_work_chains``) rather than a ``_BaseCoreWorkChain`` subclass.

Pipeline
--------
1. ``seekpath_structure_analysis``  -> primitive cell + q-point band path
2. ``phonopy_generate_displacements`` -> N displaced supercells + setting info
3. ``ForceWorkChain`` x N            -> CP2K ENERGY_FORCE per displacement
4. ``parse_cp2k_forces``             -> stacked force sets
5. ``phonopy_collect_phonons``       -> phonon band structure + DOS

Proposed entry point (pyproject.toml)::

    [project.entry-points."aiida.workflows"]
    "aim2dat.cp2k.phonons" =
        "aim2dat.aiida_workflows.cp2k.phonon_work_chain:PhononWorkChain"
"""

# Third party library imports
import aiida.orm as aiida_orm
from aiida.engine import WorkChain, ToContext
from aiida.plugins import WorkflowFactory

# Internal library imports
from aim2dat.aiida_workflows.cp2k.work_chain_specs import structural_p_specs
from aim2dat.aiida_workflows.utils import seekpath_structure_analysis
from aim2dat.aiida_workflows.cp2k.phonopy_utils import (
    phonopy_generate_displacements,
    parse_cp2k_forces,
    phonopy_collect_phonons,
)

ForceWorkChain = WorkflowFactory("aim2dat.cp2k.force_eval")


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


class PhononWorkChain(WorkChain):
    """
    AiiDA work chain to compute the phonon band structure and DOS via the
    finite-displacement method with CP2K and phonopy.
    """

    @classmethod
    def define(cls, spec):
        """Specify inputs, outputs and the outline."""
        super().define(spec)
        spec = structural_p_specs(spec)

        # --- phonopy settings ------------------------------------------------ #
        spec.input(
            "phonopy_p.supercell_matrix",
            valid_type=aiida_orm.List,
            help="Supercell matrix, e.g. [[2,0,0],[0,2,0],[0,0,2]].",
        )
        spec.input(
            "phonopy_p.displacement",
            valid_type=aiida_orm.Float,
            default=lambda: aiida_orm.Float(0.01),
            help="Displacement amplitude in Angstrom.",
        )
        spec.input(
            "phonopy_p.symprec",
            valid_type=aiida_orm.Float,
            default=lambda: aiida_orm.Float(0.005),
            help="Symmetry tolerance; should match the `eps_symmetry` used in the "
            "upstream cell optimization.",
        )
        spec.input(
            "phonopy_p.dos_mesh",
            valid_type=aiida_orm.List,
            default=lambda: aiida_orm.List(list=[20, 20, 20]),
            help="q-point mesh for the phonon DOS.",
        )
        spec.input(
            "phonopy_p.thermal_properties",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(False),
            help="Also compute thermal properties (free energy, Cv, entropy).",
        )
        # v2 TODO: non-analytical correction. Reserved now so adding it later
        # does not change the input spec.
        spec.input(
            "phonopy_p.nac_parameters",
            valid_type=aiida_orm.Dict,
            required=False,
            help="Reserved (v2): Born effective charges + dielectric tensor for the "
            "non-analytical correction. Not used in v1.",
        )

        # --- band path (SeekPath) ------------------------------------------- #
        spec.input(
            "seekpath_parameters",
            valid_type=aiida_orm.Dict,
            default=lambda: aiida_orm.Dict(dict={"reference_distance": 0.015, "symprec": 0.005}),
            help="Arguments passed to the SeekPath analysis for the band path.",
        )

        # --- per-displacement force calculation ----------------------------- #
        # Exposes the CP2K code, numerical_p and SCF settings of ForceWorkChain;
        # the structure is supplied per displacement.
        spec.expose_inputs(
            ForceWorkChain, namespace="force", exclude=("structural_p.structure",)
        )

        spec.outline(
            cls.setup,
            cls.run_forces,
            cls.inspect_forces,
            cls.collect_phonons,
            cls.result,
        )

        spec.output("phonon_bands", valid_type=aiida_orm.Dict)
        spec.output("phonon_dos", valid_type=aiida_orm.XyData)
        spec.output("thermal_properties", valid_type=aiida_orm.Dict, required=False)
        spec.output("phonon_setting_info", valid_type=aiida_orm.Dict)
        spec.output_namespace(
            "seekpath", required=False, dynamic=True,
            help="Primitive structure / path from SeekPath."
        )

        spec.exit_code(
            401,
            "ERROR_FORCE_CALCULATION",
            message="One or more displacement force calculations did not finish OK.",
        )

    def setup(self):
        """Run SeekPath for the band path, then generate displaced supercells."""
        seekpath = seekpath_structure_analysis(
            self.inputs.structural_p.structure, self.inputs.seekpath_parameters
        )
        self.ctx.primitive_structure = seekpath["primitive_structure"]
        self.ctx.path_parameters = seekpath["parameters"].get_dict()
        self.out("seekpath.primitive_structure", seekpath["primitive_structure"])
        self.out("seekpath.path_parameters", seekpath["parameters"])

        gen_parameters = aiida_orm.Dict(
            dict={
                "supercell_matrix": self.inputs.phonopy_p.supercell_matrix.get_list(),
                "displacement": self.inputs.phonopy_p.displacement.value,
                "symprec": self.inputs.phonopy_p.symprec.value,
            }
        )
        generated = phonopy_generate_displacements(self.ctx.primitive_structure, gen_parameters)
        self.ctx.phonon_setting_info = generated.pop("phonon_setting_info")
        self.ctx.supercells = dict(generated)  # supercell_XXXX -> StructureData
        self.report(f"Generated {len(self.ctx.supercells)} displaced supercells.")

    def run_forces(self):
        """Fan out one ForceWorkChain per displaced supercell."""
        for key in sorted(self.ctx.supercells):
            inputs = self.exposed_inputs(ForceWorkChain, namespace="force")
            inputs.structural_p.structure = self.ctx.supercells[key]
            running = self.submit(ForceWorkChain, **inputs)
            self.report(f"Launching ForceWorkChain <{running.pk}> for {key}.")
            self.to_context(**{f"force_{key}": running})

    def inspect_forces(self):
        """Verify every force calculation finished and collect its output folder."""
        self.ctx.retrieved = {}
        for key in sorted(self.ctx.supercells):
            wc = self.ctx[f"force_{key}"]
            if not wc.is_finished_ok:
                self.report(f"ForceWorkChain for {key} failed ({wc.exit_status}).")
                return self.exit_codes.ERROR_FORCE_CALCULATION
            # NOTE: verify the retrieved-folder output port name against the
            # aim2dat.cp2k calcjob exposed outputs (assumed `cp2k.retrieved`).
            self.ctx.retrieved[key] = wc.outputs.cp2k.retrieved

    def collect_phonons(self):
        """Parse forces and assemble the band structure + DOS."""
        force_sets = parse_cp2k_forces(self.ctx.phonon_setting_info, **self.ctx.retrieved)[
            "force_sets"
        ]
        band_path, band_labels = _seekpath_to_phonopy_path(self.ctx.path_parameters)
        collect_parameters = aiida_orm.Dict(
            dict={
                "band_path": band_path,
                "band_labels": band_labels,
                "dos_mesh": self.inputs.phonopy_p.dos_mesh.get_list(),
                "thermal_properties": self.inputs.phonopy_p.thermal_properties.value,
            }
        )
        self.ctx.results = phonopy_collect_phonons(
            self.ctx.primitive_structure,
            self.ctx.phonon_setting_info,
            force_sets,
            collect_parameters,
        )

    def result(self):
        """Expose the outputs."""
        self.out("phonon_bands", self.ctx.results["band_structure"])
        self.out("phonon_dos", self.ctx.results["total_dos"])
        self.out("phonon_setting_info", self.ctx.phonon_setting_info)
        if "thermal_properties" in self.ctx.results:
            self.out("thermal_properties", self.ctx.results["thermal_properties"])
