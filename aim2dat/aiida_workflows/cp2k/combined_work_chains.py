"""
Aiida workchains for cp2k to run advanced tasks combining different basic workchains.
"""

# Third party library imports
import aiida.orm as aiida_orm
from aiida.engine import WorkChain, ToContext, if_, while_
from aiida.plugins import CalculationFactory, WorkflowFactory, DataFactory

# Internal library imports
from aim2dat.aiida_workflows.cp2k.work_chain_specs import (
    structural_p_specs,
    numerical_p_specs,
)
from aim2dat.aiida_workflows.cp2k.auxiliary_functions import (
    return_work_chain_info,
)
from aim2dat.aiida_workflows.cp2k.surface_opt_utils import (
    surfopt_setup,
    surfopt_should_run_add_calc,
    surfopt_should_run_slab_conv,
)
from aim2dat.aiida_workflows.cp2k.el_properties_utils import elprop_setup

SurfaceData = DataFactory("aim2dat.surface")
StructureData = DataFactory("core.structure")
Cp2kCalculation = CalculationFactory("aim2dat.cp2k")
FindSCFParametersWC = WorkflowFactory("aim2dat.cp2k.find_scf_p")
CellOptWC = WorkflowFactory("aim2dat.cp2k.cell_opt")
GeoOptWC = WorkflowFactory("aim2dat.cp2k.geo_opt")
EigenvaluesWC = WorkflowFactory("aim2dat.cp2k.eigenvalues")
BandStructureWC = WorkflowFactory("aim2dat.cp2k.band_structure")
PDOSWC = WorkflowFactory("aim2dat.cp2k.pdos")
PartialChargesWC = WorkflowFactory("aim2dat.cp2k.partial_charges")


def _validate_bulk_reference(ref, _):
    """Validate bulk reference values."""
    return None


class SurfaceOptWorkChain(WorkChain):
    """
    Work chain to converge the slab size and optimize the atomic positions of the converged
    slab.
    """

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec = numerical_p_specs(spec, required=True)
        spec.input(
            "structural_p.surface",
            valid_type=SurfaceData,
            required=True,
            help="Vacuum at the bottom and top of the slab.",
        )
        spec.input(
            "structural_p.periodic",
            valid_type=aiida_orm.Bool,
            required=True,
            help="Whether the cell direction normal to the surface plan is periodic or not.",
        )
        spec.input(
            "structural_p.vacuum",
            valid_type=aiida_orm.Float,
            required=True,
            help="Vacuum at the bottom and top of the slab.",
        )
        spec.input(
            "structural_p.vacuum_factor",
            valid_type=aiida_orm.Float,
            default=lambda: aiida_orm.Float(0.0),
            required=True,
            help="Vacuum at the bottom and top of the slab.",
        )
        spec.input(
            "structural_p.minimum_slab_size",
            valid_type=aiida_orm.Float,
            default=lambda: aiida_orm.Float(15.0),
            required=True,
            help="Vacuum at the bottom and top of the slab.",
        )
        spec.input(
            "structural_p.maximum_slab_size",
            valid_type=aiida_orm.Float,
            default=lambda: aiida_orm.Float(50.0),
            required=True,
            help="Vacuum at the bottom and top of the slab.",
        )
        spec.expose_inputs(GeoOptWC, exclude="structural_p")
        spec.input_namespace("preopt_numerical_p", required=True, dynamic=False, help=".")
        spec.input(
            "preopt_numerical_p.basis_sets",
            valid_type=(aiida_orm.Str, aiida_orm.Dict),
            required=False,
            help="Basis sets used for the different species for the pre-optimization steps.",
        )
        spec.input(
            "preopt_numerical_p.cutoff_values",
            valid_type=aiida_orm.Dict,
            required=False,
            help="Cut-off values for the grids for the pre-optimization steps.",
        )
        spec.input(
            "preopt_numerical_p.basis_file",
            valid_type=DataFactory("core.singlefile"),
            help="File containing the used basis sets for the pre-optimization steps.",
            required=False,
        )
        spec.input_namespace("preopt_optimization_p", required=True, dynamic=False, help=".")
        spec.input(
            "preopt_optimization_p.max_force",
            valid_type=aiida_orm.Float,
            required=False,
            help="Convergence criterion for the maximum force component.",
        )
        spec.input(
            "preopt_optimization_p.rms_force",
            valid_type=aiida_orm.Float,
            required=False,
            help="Convergence criterion for the root mean square (RMS) force.",
        )
        spec.input(
            "preopt_optimization_p.max_dr",
            valid_type=aiida_orm.Float,
            required=False,
            help="Convergence criterion for the maximum geometry change.",
        )
        spec.input(
            "preopt_optimization_p.rms_dr",
            valid_type=aiida_orm.Float,
            required=False,
            help="Convergence criterion for the root mean square (RMS) geometry change.",
        )
        spec.input_namespace("slab_conv", required=True, dynamic=False, help=".")
        spec.input(
            "slab_conv.criteria",
            valid_type=aiida_orm.Str,
            default=lambda: aiida_orm.Str("surface_energy"),
            required=True,
            help="Convergence criteria to choose the slab size.",
        )
        spec.input(
            "slab_conv.threshold",
            valid_type=aiida_orm.Float,
            default=lambda: aiida_orm.Float(0.1),
            required=True,
            help="Convergence threshold for the surface or formation energy.",
        )
        spec.input_namespace(
            "bulk_reference",
            validator=_validate_bulk_reference,
            required=True,
            dynamic=True,
            help="Reference total energies of bulk systems.",
        )
        spec.output(
            "scf_parameters",
            valid_type=aiida_orm.Dict,
            required=True,
            help="Information on the SCF-Parameters that converge the Kohn-Sham equations.",
        )
        spec.output(
            "primitive_slab",
            valid_type=StructureData,
            required=False,
            help="Primitive unrelaxed surface slab.",
        )
        spec.output(
            "path_parameters",
            valid_type=aiida_orm.Dict,
            required=False,
            help="K-path parameters for the slab.",
        )
        spec.output_namespace(
            "cp2k",
            required=True,
            dynamic=True,
            help="Output data of the last CP2K calculation.",
        )
        spec.exit_code(
            600,
            "ERROR_INPUT_WRONG_VALUE",
            message='Input parameter "{parameter}" contains an unsupported value.',
        )
        spec.exit_code(
            700,
            "ERROR_SCF_PARAMETERS",
            message="SCF-parameters could not be retrieved.",
        )
        spec.exit_code(
            701,
            "ERROR_GEO_OPT",
            message="Surface slab could not be converged.",
        )
        spec.outline(
            cls.setup,
            while_(cls.should_run_slab_conv)(
                cls.find_scf_p,
                cls.inspect_find_scf_p_results,
                cls.geo_preopt,
                cls.inspect_geo_opt_results,
                cls.geo_opt,
                cls.inspect_geo_opt_results,
            ),
            if_(cls.should_run_add_calc)(
                cls.find_scf_p,
                cls.inspect_find_scf_p_results,
                cls.geo_preopt,
                cls.inspect_geo_opt_results,
                cls.geo_opt,
                cls.inspect_geo_opt_results,
            ),
            cls.post_processing,
        )

    def setup(self):
        """Define initial parameters."""
        exit_code = surfopt_setup(self.ctx, self.inputs)
        if exit_code is not None:
            return self.exit_codes.ERROR_INPUT_WRONG_VALUE.format(parameter="bulk_reference")

    def should_run_slab_conv(self):
        """
        Check whether the convergence criteria is fulfilled and the slab size is not
        exceeding the maximum slab size.
        """
        cond, reports = surfopt_should_run_slab_conv(self.ctx, self.inputs)
        for report in reports:
            self.report(report)
        return cond

    def should_run_add_calc(self):
        """
        Check whether additional calculations are run after the slab size is converged.
        """
        return surfopt_should_run_add_calc(self.ctx, self.inputs)

    def find_scf_p(self):
        """
        Run the FindSCFParameters work chain.
        """
        builder = FindSCFParametersWC.get_builder()
        for p_label in self.ctx.base_input_p:
            if p_label in self.inputs:
                builder[p_label] = self.inputs[p_label]
        builder.numerical_p = self.inputs.numerical_p
        builder.structural_p.structure = self.ctx.surface_slab
        if self.ctx.scf_p is not None:
            builder.structural_p.scf_parameters = self.ctx.scf_p
        running = self.submit(builder)
        self.report(f"Launching FindSCFParametersWorkChain <{running.pk}>.")
        return ToContext(find_scf_p=running)

    def geo_preopt(self):
        """
        Run the GeoOpt work chain.
        """
        builder = GeoOptWC.get_builder()
        for p_label in self.ctx.base_input_p:
            if p_label in self.inputs:
                builder[p_label] = self.inputs[p_label]
        builder.adjust_scf_parameters = self.inputs.adjust_scf_parameters
        builder.numerical_p = self.inputs.preopt_numerical_p
        builder.numerical_p.xc_functional = self.inputs.numerical_p.xc_functional
        builder.numerical_p.kpoints_ref_dist = self.inputs.numerical_p.kpoints_ref_dist
        if "pseudo_file" in self.inputs.numerical_p:
            builder.numerical_p.pseudo_file = self.inputs.numerical_p.pseudo_file
        builder.optimization_p = self.inputs.preopt_optimization_p
        builder.structural_p.structure = self.ctx.surface_slab
        if self.ctx.scf_p is not None:
            builder.structural_p.scf_parameters = self.ctx.scf_p
        if self.ctx.fix_atoms:
            p_dict = builder["cp2k"]["parameters"].get_dict()
            p_dict.update(
                {
                    "MOTION": {
                        "CONSTRAINT": {"FIXED_ATOMS": {"LIST": " ".join(self.ctx.fixed_atoms)}}
                    }
                }
            )
            builder["cp2k"]["parameters"] = aiida_orm.Dict(dict=p_dict)
        running = self.submit(builder)
        self.report(f"Launching GeoOptWorkChain <{running.pk}>.")
        return ToContext(geo_opt=running)

    def geo_opt(self):
        """
        Run the GeoOpt work chain.
        """
        builder = GeoOptWC.get_builder()
        for p_label in self.ctx.base_input_p:
            if p_label in self.inputs:
                builder[p_label] = self.inputs[p_label]
        builder.initial_opt_parameters = self.ctx.initial_opt_parameters
        builder.adjust_scf_parameters = self.inputs.adjust_scf_parameters
        builder.numerical_p = self.inputs.numerical_p
        builder.optimization_p = self.inputs.optimization_p
        builder.structural_p.structure = self.ctx.surface_slab
        if self.ctx.scf_p is not None:
            builder.structural_p.scf_parameters = self.ctx.scf_p
        if self.ctx.fix_atoms:
            p_dict = builder["cp2k"]["parameters"].get_dict()
            p_dict.update(
                {
                    "MOTION": {
                        "CONSTRAINT": {"FIXED_ATOMS": {"LIST": " ".join(self.ctx.fixed_atoms)}}
                    }
                }
            )
            builder["cp2k"]["parameters"] = aiida_orm.Dict(dict=p_dict)
        running = self.submit(builder)
        self.report(f"Launching GeoOptWorkChain <{running.pk}>.")
        return ToContext(geo_opt=running)

    def inspect_find_scf_p_results(self):
        """
        Check if the previous work chain finished successful.
        """
        if not self.ctx.find_scf_p.is_finished_ok:
            return self.exit_codes.ERROR_SCF_PARAMETERS
        self.ctx.scf_p = self.ctx.find_scf_p.outputs.scf_parameters
        self.ctx.parent_calc_folder = self.ctx.find_scf_p.outputs["cp2k"]["remote_folder"]

    def inspect_geo_opt_results(self):
        """
        Check if the previous work chain finished successful.
        """
        if "geo_opt" in self.ctx and not self.ctx.geo_opt.is_finished_ok:
            return self.exit_codes.ERROR_GEO_OPT
        self.ctx.parent_calc_folder = None
        self.ctx.surface_slab = self.ctx.geo_opt.outputs["cp2k"]["output_structure"]

    def post_processing(self):
        """
        Define outputs.
        """
        self.out("scf_parameters", self.ctx.geo_opt.outputs["scf_parameters"])
        if "k_path_parameters" in self.ctx:
            self.out("path_parameters", self.ctx.k_path_parameters)
        if "prim_slab" in self.ctx:
            self.out("primitive_slab", self.ctx.prim_slab)
        for label, val in self.ctx.geo_opt.outputs["cp2k"].items():
            self.out("cp2k." + label, val)


class ElectronicPropertiesWorkChain(WorkChain):
    """
    Work chain to optimize the unit cell and calculate different electronic properties.
    """

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec = structural_p_specs(spec)
        spec = numerical_p_specs(spec, required=True)
        spec.input_namespace("cp2k", required=True, help="Details about the used CP2K code.")
        spec.input(
            "cp2k.code",
            valid_type=DataFactory("core.code"),
            help="CP2K code to be used for the calculations.",
        )
        spec.input(
            "cp2k.metadata",
            valid_type=aiida_orm.Dict,
            help="Forwards extra information to the calculation job (e.g. for the scheduler).",
            required=False,
        )
        spec.input_namespace(
            "critic2",
            required=False,
            help="Details about the used critic2 code.",
            populate_defaults=False,
        )
        spec.input(
            "critic2.code",
            valid_type=DataFactory("core.code"),
            help="Critic2-code to be used for the calculations.",
            required=False,
        )
        spec.input(
            "critic2.metadata",
            valid_type=aiida_orm.Dict,
            help="Forwards extra information to the calculation job (e.g. for the scheduler).",
            required=False,
        )
        spec.input_namespace(
            "chargemol",
            required=False,
            help="Details about the used chargemol code.",
            populate_defaults=False,
        )
        spec.input(
            "chargemol.code",
            valid_type=DataFactory("core.code"),
            help="Chargemol-code to be used for the calculations.",
            required=False,
        )
        spec.input(
            "chargemol.metadata",
            valid_type=aiida_orm.Dict,
            help="Forwards extra information to the calculation job (e.g. for the scheduler).",
            required=False,
        )
        spec.input(
            "chargemol.path_atomic_densities",
            valid_type=aiida_orm.Str,
            required=False,
            help="Absolte path to the atomic densities needed for the chargemol calculation.",
        )
        spec.input(
            "structural_p.system_character",
            valid_type=aiida_orm.Str,
            required=False,
            help="Electronic character of the system, possible options are 'metallic' or "
            "'insulator'. In case this parameter is set to 'metallic' ('insulator') electronic "
            "smearing is always (never) applied.",
        )
        spec.input_namespace(
            "workflow", required=True, help="Parameters that define the workflow."
        )
        spec.input(
            "workflow.run_cell_optimization",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(True),
            help="Whether a unit cell optimization should be performed.",
        )
        spec.input(
            "workflow.calc_band_structure",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(False),
            help="Whether the band structure should be calculated.",
        )
        spec.input(
            "workflow.calc_eigenvalues",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(False),
            help="Whether the eigenvalues should be calculated.",
        )
        spec.input(
            "workflow.calc_pdos",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(False),
            help="Whether the projected density of states (pDOS) should be calculated.",
        )
        spec.input(
            "workflow.calc_partial_charges",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(False),
            help="Whether the partial charges should be calculated.",
        )
        spec.input(
            "workflow.protocol",
            valid_type=aiida_orm.Str,
            default=lambda: aiida_orm.Str("cp2k-crystal-standard"),
            help="Protocol type to be used for the calculations. The default value is 'standard'.",
        )
        spec.input(
            "workflow.custom_protocol",
            valid_type=aiida_orm.Dict,
            help="Use a custom protocol to set the parameters of the work chain.",
            required=False,
        )
        spec.input(
            "clean_workdir",
            valid_type=aiida_orm.Bool,
            help="Clean woring directory after calculation.",
            required=False,
        )
        spec.outline(
            cls.setup,
            cls.find_scf_parameters,
            if_(cls.should_run_cell_opt)(cls.dft_cell_opt),
            cls.electronic_structure,
            cls.post_processing,
        )
        spec.output(
            "general_info",
            valid_type=aiida_orm.Dict,
            help="General properties.",
        )
        spec.output(
            "optimized_structure",
            valid_type=DataFactory("core.structure"),
            help="Optimized unit cell of the crystal.",
            required=False,
        )
        spec.output(
            "general_info",
            valid_type=aiida_orm.Dict,
            help="Information on the constrained space group.",
            required=False,
        )
        spec.output(
            "eigenvalue_info",
            valid_type=aiida_orm.Dict,
            help="Contains all calculated eigenvalues for all k-points.",
            required=False,
        )
        spec.output(
            "band_structure",
            valid_type=DataFactory("core.array.bands"),
            help="The electronic band structure of the crysal.",
            required=False,
        )
        spec.output(
            "pdos",
            valid_type=DataFactory("core.array.xy"),
            help="The projected density of states of the crystal.",
            required=False,
        )
        spec.output(
            "mulliken_populations",
            valid_type=aiida_orm.List,
            help="The Mulliken populations of the system.",
            required=False,
        )
        spec.output(
            "hirshfeld_populations",
            valid_type=aiida_orm.List,
            help="The Hirshfeld populations of the system.",
            required=False,
        )
        spec.output(
            "bader_populations",
            valid_type=aiida_orm.List,
            help="The Bader populations of the system.",
            required=False,
        )
        spec.output(
            "ddec6_populations",
            valid_type=aiida_orm.List,
            help="The DDEC6 populations of the system.",
            required=False,
        )
        spec.exit_code(
            700,
            "ERROR_SCF_PARAMETERS",
            message="SCF-parameters could not be retrieved.",
        )
        spec.exit_code(
            701,
            "ERROR_CELL_OPT",
            message="Crystal structure could not be converged.",
        )
        spec.exit_code(
            702,
            "ERROR_BAND_STRUCTURE",
            message="Band structure could not be calculated.",
        )
        spec.exit_code(
            703,
            "ERROR_EIGENVALUES",
            message="Eigenvalues could not be calculated.",
        )
        spec.exit_code(
            704,
            "ERROR_PDOS",
            message="PDOS could not be calculated.",
        )
        spec.exit_code(
            705,
            "ERROR_PARTIAL_CHARGES",
            message="Partial charges could not be calculated.",
        )

    def setup(self):
        """Set up calculation parameters."""
        elprop_setup(self.ctx, self.inputs)

    def find_scf_parameters(self):
        """
        Find mixing parameters that converge the Kohn-Sham equations.
        """
        builder = FindSCFParametersWC.get_builder()
        for p_label, p_value in self.ctx.find_scf_parameters.items():
            self.set_input_parameter(builder, p_label, p_value)
        builder.structural_p.structure = self.inputs.structural_p.structure
        if "system_character" in self.inputs.structural_p:
            builder.structural_p.system_character = self.inputs.structural_p.system_character
        running = self.submit(builder)
        self.report(f"Launching FindSCFMixingParametersWorkChain <{running.pk}>.")
        return ToContext(find_scf_p=running)

    def should_run_cell_opt(self):
        """Whether to run a cell optimization."""
        return self.inputs.workflow.run_cell_optimization.value

    def dft_cell_opt(self):
        """
        Perform the cell relaxation.
        """
        if not self.ctx.find_scf_p.is_finished_ok:
            return self.exit_codes.ERROR_SCF_PARAMETERS

        builder = CellOptWC.get_builder()
        for p_label, p_value in self.ctx.unit_cell_opt.items():
            self.set_input_parameter(builder, p_label, p_value)
        builder.structural_p.structure = self.ctx.find_scf_p.outputs.seekpath.primitive_structure
        builder.structural_p.scf_parameters = self.ctx.find_scf_p.outputs.scf_parameters
        running = self.submit(builder)
        self.report(f"Launching CellOptWorkChain <{running.pk}>.")
        return ToContext(optimized=running)

    def electronic_structure(self):
        """
        Calculate the electronic properties of the crystal.
        """
        if self.inputs.workflow.run_cell_optimization.value:
            if not self.ctx.optimized.is_finished_ok:
                return self.exit_codes.ERROR_CELL_OPT
            structure = self.ctx.optimized.outputs.cp2k["output_structure"]
            remote_folder = self.ctx.optimized.outputs.cp2k["remote_folder"]
            scf_parameters = self.ctx.optimized.outputs.scf_parameters
        else:
            if not self.ctx.find_scf_p.is_finished_ok:
                return self.exit_codes.ERROR_SCF_PARAMETERS
            structure = self.ctx.find_scf_p.outputs.seekpath.primitive_structure
            remote_folder = self.ctx.find_scf_p.outputs.cp2k["remote_folder"]
            scf_parameters = self.ctx.find_scf_p.outputs.scf_parameters

        if self.inputs.workflow.calc_band_structure.value:
            running_bs = self.run_el_prop_wc(
                "band_structure",
                BandStructureWC,
                structure,
                remote_folder,
                scf_parameters,
                [("path_parameters", self.ctx.find_scf_p.outputs.seekpath.path_parameters)],
            )
            self.report(f"Launching BandStructureWorkChain <{running_bs.pk}>.")
            self.to_context(band_structure=running_bs)

        if self.inputs.workflow.calc_eigenvalues.value:
            running_ev = self.run_el_prop_wc(
                "eigenvalues", EigenvaluesWC, structure, remote_folder, scf_parameters, []
            )
            self.report(f"Launching EigenValuesWorkChain <{running_ev.pk}>.")
            self.to_context(eigenvalues=running_ev)

        if self.inputs.workflow.calc_pdos.value:
            running_pdos = self.run_el_prop_wc("pdos", PDOSWC, structure, None, scf_parameters, [])
            self.report(f"Launching PDOSWorkChain <{running_pdos.pk}>.")
            self.to_context(pdos=running_pdos)
        if self.inputs.workflow.calc_partial_charges.value:
            running_pc = self.run_el_prop_wc(
                "partial_charges", PartialChargesWC, structure, remote_folder, scf_parameters, []
            )
            self.report(f"Launching PartialChargesWorkChain <{running_pc.pk}>.")
            self.to_context(partial_charges=running_pc)

    def run_el_prop_wc(
        self, task_label, work_chain, structure, remote_folder, scf_parameters, extra_input
    ):
        """Run electronic properties calculation."""
        builder = work_chain.get_builder()
        input_parameters = getattr(self.ctx, task_label)
        for p_label, p_value in input_parameters.items():
            self.set_input_parameter(builder, p_label, p_value)
        builder.structural_p.structure = structure
        builder.structural_p.scf_parameters = scf_parameters
        if remote_folder is not None:
            builder.cp2k.parent_calc_folder = remote_folder
        for inp in extra_input:
            setattr(builder, inp[0], inp[1])
        return self.submit(builder)

    def post_processing(self):
        """
        Post-processing routine.
        """
        if self.inputs.workflow.calc_band_structure and not self.ctx.band_structure.is_finished_ok:
            return self.exit_codes.ERROR_BAND_STRUCTURE
        if self.inputs.workflow.calc_eigenvalues and not self.ctx.eigenvalues.is_finished_ok:
            return self.exit_codes.ERROR_EIGENVALUES
        if self.inputs.workflow.calc_pdos and not self.ctx.pdos.is_finished_ok:
            return self.exit_codes.ERROR_PDOS
        if (
            self.inputs.workflow.calc_partial_charges
            and not self.ctx.partial_charges.is_finished_ok
        ):
            return self.exit_codes.ERROR_PARTIAL_CHARGES

        # Create output
        # TO-DO include space group
        self.out(
            "general_info",
            return_work_chain_info(
                self.ctx.find_scf_p.outputs.cp2k["output_parameters"],
                self.ctx.find_scf_p.outputs.seekpath.primitive_structure,
            ),
        )
        if self.inputs.workflow.run_cell_optimization.value:
            self.out("optimized_structure", self.ctx.optimized.outputs.cp2k["output_structure"])
        if self.inputs.workflow.calc_band_structure.value:
            self.out("band_structure", self.ctx.band_structure.outputs.cp2k["output_bands"])
        if self.inputs.workflow.calc_pdos.value:
            self.out("pdos", self.ctx.pdos.outputs.cp2k["output_pdos"])
        if self.inputs.workflow.calc_eigenvalues.value:
            self.out("eigenvalue_info", self.ctx.eigenvalues.outputs.cp2k["output_eigenvalues"])
        if self.inputs.workflow.calc_partial_charges.value:
            self.out(
                "mulliken_populations",
                self.ctx.partial_charges.outputs.cp2k["output_mulliken_populations"],
            )
            self.out(
                "hirshfeld_populations",
                self.ctx.partial_charges.outputs.cp2k["output_hirshfeld_populations"],
            )
            if "critic2" in self.inputs:
                self.out(
                    "bader_populations",
                    self.ctx.partial_charges.outputs.critic2["output_bader_populations"],
                )
            if "chargemol" in self.inputs:
                self.out(
                    "ddec6_populations",
                    self.ctx.partial_charges.outputs.chargemol["output_ddec6_populations"],
                )

    @staticmethod
    def set_input_parameter(work_chain_builder, input_key, value):
        """Set input parameter for a child work chain."""
        input_setter = work_chain_builder
        input_path = input_key.split(".")
        for keyword in input_path[:-1]:
            input_setter = input_setter[keyword]
        input_setter[input_path[-1]] = value
