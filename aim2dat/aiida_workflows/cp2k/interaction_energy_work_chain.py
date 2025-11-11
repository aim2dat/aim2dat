"""
Aiida work chains for cp2k to calculate the interaction energy.
"""

# Standard library imports
from copy import deepcopy

# Third party library imports
import aiida.orm as aiida_orm

# Internal library imports
from aim2dat.aiida_workflows.cp2k.base_core_work_chain import _BaseCoreWorkChain
from aim2dat.utils.dict_tools import dict_retrieve_parameter, dict_set_parameter


def _setup_wc_specific_inputs(ctx, inputs):
    """Set input for the interaction energy calculation."""
    ctx.inputs.metadata["options"]["parser_name"] = "aim2dat.cp2k.standard"
    parameters = ctx.inputs.parameters.get_dict()
    dict_set_parameter(parameters, ["GLOBAL", "RUN_TYPE"], "BSSE")

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
    ctx.inputs.parameters = aiida_orm.Dict(dict=parameters)


class InteractionEnergyWorkChain(_BaseCoreWorkChain):
    """
    AiiDA work chain to calculate the interaction energy using CP2K.
    """

    _keep_scf_method_fixed = True
    _keep_smearing_fixed = True
    _initial_scf_guess = "RESTART"

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

    def setup_wc_specific_inputs(self):
        """Set input for the interaction energy calculation."""
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
