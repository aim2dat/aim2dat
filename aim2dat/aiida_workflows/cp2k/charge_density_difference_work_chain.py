"""
Aiida work chains for cp2k with cubecruncher to calculate charge density difference cubes.
"""

# Standard library imports
from copy import deepcopy

# Third party library imports
import aiida.orm as aiida_orm
from aiida.plugins import CalculationFactory
from aiida.engine import (
    while_,
    process_handler,
    ExitCode,
)
from aiida.common import AttributeDict

# Internal library imports
from aim2dat.aiida_workflows.cp2k.base_core_work_chain import _BaseCoreWorkChain
from aim2dat.aiida_workflows.cp2k.core_work_chain_handlers import _switch_to_atomic_scf_guess
from aim2dat.utils.dict_tools import dict_retrieve_parameter, dict_set_parameter

CubecruncherCalculation = CalculationFactory("aim2dat.cubecruncher")


class ChargeDensityDifferenceWorkChain(_BaseCoreWorkChain):
    """AiiDA work chain to calculate the charge density difference of systems."""

    _keep_scf_method_fixed = True
    _keep_smearing_fixed = True
    _initial_scf_guess = "ATOMIC"

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.input(
            "adjust_scf_parameters",
            valid_type=aiida_orm.Bool,
            default=lambda: aiida_orm.Bool(False),
            help="Restart calculation with adjusted parameters if SCF-cycles are not converged.",
        )
        spec.input(
            "fragments",
            valid_type=aiida_orm.List,
            required=False,
            help="Specify the atom number belonging to the fragments."
            "If only a few atoms are given, the rest will be considered as one fragment."
            "Make sure to count atomic index from 0.",
        )
        spec.expose_inputs(
            CubecruncherCalculation,
            namespace="cubecruncher",
            exclude=("charge_density_folder"),
        )
        spec.expose_outputs(
            CubecruncherCalculation,
            namespace="cubecruncher",
        )
        spec.outline(
            cls.setup_inputs,
            cls.setup_wc_specific_inputs,
            cls.initialize_scf_parameters,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.post_processing,
            cls.setup_external_cubecruncher,
            cls.wc_specific_post_processing,
        )

    def setup_wc_specific_inputs(self):
        """Set input parameters to calculate charge density differences."""
        self.ctx.inputs.metadata.options.parser_name = "aim2dat.cp2k.standard"
        # Check fragment input
        fragments_list = self.inputs.fragments.get_list()
        if all(isinstance(i, int) for i in fragments_list):
            fragments_list = [fragments_list]
        fragments_list = [
            [fragments] if isinstance(fragments, int) else fragments
            for fragments in fragments_list
        ]
        # Check input type
        if not all(isinstance(fragments, list) for fragments in fragments_list):
            return self.exit_codes.ERROR_INPUT_WRONG_VALUE

        fragment_numbers = [i for fragments in fragments_list for i in fragments]
        # Raise error if an atom is in two fragments
        if len(fragment_numbers) != len(set(fragment_numbers)):
            return self.exit_codes.ERROR_INPUT_WRONG_VALUE
        _setup_wc_specific_inputs(self.ctx, self.inputs)

    def setup_external_cubecruncher(self):
        """Set input parameters for external post-processing codes."""
        inputs = AttributeDict(self.exposed_inputs(CubecruncherCalculation, "cubecruncher"))
        inputs.charge_density_folder = self.ctx.children[-1].outputs.remote_folder
        running = self.submit(CubecruncherCalculation, **inputs)
        self.report(f"Launching  <{running.pk}>.")
        self.to_context(cubecruncher=running)

    def wc_specific_post_processing(self):
        """Expose outputs of the external codes."""
        self.out_many(
            self.exposed_outputs(
                self.ctx.cubecruncher, CubecruncherCalculation, namespace="cubecruncher"
            )
        )

    @process_handler(
        priority=402,
        exit_codes=ExitCode(0),
    )
    def switch_to_atomic_scf_guess(self, calc):
        """
        Switch to atomic guess for the case that the scf-cycles do not converge.
        """
        return self._execute_error_handler(calc, _switch_to_atomic_scf_guess)


def _setup_wc_specific_inputs(ctx, inputs):
    """Set input for the interaction energy calculation."""
    parameters = ctx.inputs.parameters.get_dict()
    dict_set_parameter(parameters, ["GLOBAL", "RUN_TYPE"], "BSSE")
    dict_set_parameter(parameters, ["GLOBAL", "PRINT_LEVEL"], "MEDIUM")

    # Set cube inputs
    dict_set_parameter(parameters, ["FORCE_EVAL", "DFT", "PRINT", "E_DENSITY_CUBE", "STRIDE"], 1)

    # Set ghost atoms
    kind_parameters = dict_retrieve_parameter(parameters, ["FORCE_EVAL", "SUBSYS", "KIND"])
    kind_parameters_new = deepcopy(kind_parameters)
    for kind in kind_parameters:
        kind_ghost = deepcopy(kind)
        kind_ghost["_"] += "_ghost"
        kind_ghost.pop("POTENTIAL")
        kind_ghost["GHOST"] = True
        kind_parameters_new.append(kind_ghost)
    dict_set_parameter(parameters, ["FORCE_EVAL", "SUBSYS", "KIND"], kind_parameters_new)

    # Set the fragments
    # Generate lists of fragments in a list
    sites = ctx.inputs.structure.sites
    fragments_list = inputs.fragments.get_list()

    # Checks and make list of fragment lists
    if all(isinstance(i, int) for i in fragments_list):
        fragments_list = [fragments_list]
    fragments_list = [
        [fragments] if isinstance(fragments, int) else fragments for fragments in fragments_list
    ]
    fragment_numbers = [i for fragments in fragments_list for i in fragments]

    # Find fragment which was not set
    # (e.g. if only molecule was set and framework was forgotten)
    all_numbers = set(range(len(sites)))
    missing_fragment = list(all_numbers - set(fragment_numbers))
    if missing_fragment:
        fragments_list.append(missing_fragment)

    # CP2K starts counting at 1 not at 0. Thus add 1 to each index if necessary
    if min(i for fragments in fragments_list for i in fragments) == 0:
        fragments_list = [[i + 1 for i in fragments] for fragments in fragments_list]

    # Pass the list into the input dictionary
    fragments_str_list = []
    for fragments in fragments_list:
        fragments_sorted = sorted(fragments)
        if len(fragments) > 4 and fragments_sorted[-1] - fragments_sorted[0] + 1 == len(
            fragments_sorted
        ):
            fragments_str = f"{fragments_sorted[0]}..{fragments_sorted[-1]}"
        else:
            atoms_string = [str(s) for s in fragments]
            fragments_str = " ".join(atoms_string)
        fragments_str_list.append({"LIST": fragments_str})
    dict_set_parameter(
        parameters,
        ["FORCE_EVAL", "BSSE", "FRAGMENT"],
        fragments_str_list,
    )
    ctx.inputs.settings = aiida_orm.Dict(dict={"output_check_scf_conv": True})
    ctx.inputs.parameters = aiida_orm.Dict(dict=parameters)
