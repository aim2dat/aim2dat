"""
Calcjob for the enumlib library.
"""

# Standard library imports
import os

# Third party library imports
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.engine import CalcJob
import aiida.orm as aiida_orm
from aiida.orm.nodes.data.base import to_aiida_type

# Internal library imports
import aim2dat.aiida_workflows.enumlib.utils as enum_utils


def validate_structure(value, _):
    """
    Validate the input structure. Checks that the structure has no fractional occupations and each
    site is only occupied once.
    """
    if not value.get_pymatgen().is_ordered:
        return (
            "Structure needs to be ordered. Each site can only be occupied by one atomic species."
        )

    checked_kinds = enum_utils.check_kinds(value.get_site_kindnames())
    if checked_kinds:
        return (
            f"Structure contains kinds that can not be processed: {checked_kinds}. "
            "Only chemical elements with additional numbers are allowed."
        )


def validate_makeStr_path(value, _):
    """
    Validate the absolute path of the `makeStr.py` executable.
    """
    if not os.path.isfile(value.value):
        return (
            "`makeStr.py` executable does not exist in the specified absolute path. "
            f"Given path: {value.value}"
        )


def validate_sites_to_enumerate(value, _):
    """
    Validate `sites_to_enumerate`.
    """
    sites_to_enumerate = value.get("sites_to_enumerate").get_list()

    num_sites = len(value.get("structure").sites)
    num_sites_to_enumerate = len(sites_to_enumerate)
    if num_sites_to_enumerate != num_sites:
        return (
            "Specify elements for each site in `sites_to_enumerate`. "
            f"Structure has {num_sites} but only {num_sites_to_enumerate} values were given."
        )
    enum_sites = [len(site) > 1 for site in sites_to_enumerate]
    if not any(enum_sites):
        return "To enumerate a site at least 2 different species need to be passed in a sublist."

    enum_kinds = set(sum(sites_to_enumerate), [])
    checked_kinds = enum_utils.check_kinds(enum_kinds)
    if checked_kinds:
        return (
            f"Enumerated kinds contain kinds that can not be processed: {checked_kinds}. "
            "Only chemical elements with additional numbers are allowed."
        )


def validate_elements_to_enumerate(value, _):
    """
    Validate `elements_to_enumerate`.
    """
    elements_to_enumerate = value.get("elements_to_enumerate").get_dict()
    if not len(elements_to_enumerate):
        return (
            "At least one element of the corresponding sites that should be enumerated needs "
            "to be specified."
        )

    input_structure = value.get("structure")
    initial_kind_names = list(set(input_structure.get_site_kindnames()))
    check_elements = [
        element for element in elements_to_enumerate if element not in initial_kind_names
    ]

    if check_elements:
        return (
            "Specified elements needs to be present in the input structure. The following "
            f"elements were given but could not be found in the input structure. {check_elements}"
        )

    enum_sites = [len(elements) > 1 for elements in elements_to_enumerate.values()]
    if not all(enum_sites):
        return (
            "At least 2 elements to enumerate need to be specified for an element of the "
            "parent structure, if specified."
        )

    enum_kinds = set(sum(list(elements_to_enumerate.values()), []))
    checked_kinds = enum_utils.check_kinds(enum_kinds)
    if checked_kinds:
        return (
            f"Enumerated kinds contain kinds that can not be processed: {checked_kinds}. "
            "Only chemical elements with additional numbers are allowed."
        )


def validate_inputs(value, _):
    """
    Validate the input parameters `sites_to_enumerate`, `elements_to_emnumerate` and
    `concentration_restrictions`.
    """
    if value.get("sites_to_enumerate") and value.get("elements_to_enumerate"):
        return (
            "`sites_to_enumerate` and `elements_to_enumerate` were specified both. "
            "Only specify once at a time."
        )

    if value.get("sites_to_enumerate"):
        validated = validate_sites_to_enumerate(value, _)
        if validated:
            return "`sites_to_enumerate`: " + validated
        to_enumerate = value.get("sites_to_enumerate").get_list()
    elif value.get("elements_to_enumerate"):
        validated = validate_elements_to_enumerate(value, _)
        if validated:
            return "`elements_to_enumerate`: " + validated
        to_enumerate = value.get("elements_to_enumerate").get_dict()

    # Validate concentration restrictions
    if value.get("concentration_restrictions"):
        structure = value.get("structure")
        kind_names = enum_utils.get_kindnames(structure, to_enumerate)
        concentration_restrictions = value.get("concentration_restrictions").get_dict()
        check_concentrations = [el for el in kind_names if el not in concentration_restrictions]
        if check_concentrations:
            return (
                "Concentration restrictions need to be specified for all elements. "
                "Otherwise specify nothing."
            )


class EnumlibCalculation(CalcJob):
    """`CalcJob` implementation for the enum.x code of the enumlib library."""

    @classmethod
    def define(cls, spec):
        """Define input/output and outline."""
        super().define(spec)

        spec.input(
            "structure",
            valid_type=aiida_orm.StructureData,
            help="The parent structure that will be enumerated.",
            # validator=validate_structure,
        )
        spec.input(
            "path_to_makeStr",
            valid_type=aiida_orm.Str,
            help="Absolute path to the `makeStr.py` executable",
            serializer=to_aiida_type,
            validator=validate_makeStr_path,
        )
        spec.input(
            "sites_to_enumerate",
            help=(
                "List that contains a list for each site consisting of the elements that should "
                "be enumerated at that site. "
                "Custom kind names like `Ni1` or `Ca0` are also possible."
            ),
            required=False,
            valid_type=aiida_orm.List,
            serializer=to_aiida_type,
        )
        spec.input(
            "elements_to_enumerate",
            help=(
                "Dictionary that consists of elements (custom kind names are possible as well) of "
                "sites that should be enumerated as keys. The values are lists that specify the "
                "elements that will be placed on the site positions during the enumeration."
            ),
            required=False,
            valid_type=aiida_orm.Dict,
            serializer=to_aiida_type,
        )
        spec.input(
            "concentration_restrictions",
            help=(
                "Dictionary to restrict the concentrations during the enumeration. The element is "
                "used as the key and the values are lists that have to be of the following shape: "
                "[numerator1, numerator2, denominator]. The concentration will then be limited to "
                "the range numerator1/denominator - numerator2/denominator."
            ),
            required=False,
            valid_type=aiida_orm.Dict,
            serializer=to_aiida_type,
        )
        spec.input(
            "min_cell_size",
            help="Minimum size of the enumerated super cells compared to the parent cell.",
            required=False,
            valid_type=aiida_orm.Int,
            default=lambda: aiida_orm.Int(1),
            serializer=to_aiida_type,
        )
        spec.input(
            "max_cell_size",
            help="Maximum size of the enumerated super cells compared to the parent cell.",
            valid_type=aiida_orm.Int,
            serializer=to_aiida_type,
        )
        spec.input(
            "eps",
            help=(
                "Small real number that is used as an epsilon to compare two numbers "
                "(to avoid finite precision errors)."
            ),
            required=False,
            valid_type=aiida_orm.Float,
            default=lambda: aiida_orm.Float(0.001),
            serializer=to_aiida_type,
        )
        spec.input(
            "structures_to_return",
            help="List that specifies the range/indices of structures that should be returned.",
            required=False,
            valid_type=(aiida_orm.List, aiida_orm.Int),
            serializer=to_aiida_type,
        )
        spec.input(
            "structures_hard_cutoff",
            help=(
                "Maximum number of structures. If more structures are created, an error will "
                "be raised (default: No cutoff)"
            ),
            required=False,
            valid_type=aiida_orm.Int,
        )
        spec.inputs["metadata"]["options"]["input_filename"].default = "aiida.in"
        spec.inputs["metadata"]["options"]["output_filename"].default = "struct_enum.out"
        spec.inputs["metadata"]["options"]["parser_name"].default = "aim2dat.enumlib"
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
        spec.inputs.validator = validate_inputs

        spec.output_namespace(
            "output_structures", valid_type=aiida_orm.StructureData, dynamic=True
        )

        spec.exit_code(
            310,
            "ERROR_READING_STRUCTURE_OUTPUT_FILE",
            message="The structure output file could not be read.",
        )
        spec.exit_code(320, "ERROR_NO_POSCAR_FILES", message="No POSCAR files were found.")
        spec.exit_code(
            330,
            "ERROR_TOO_MANY_STRUCTURES",
            message=(
                "The number of created structures exceeds the maximum allowed number specified "
                "in `structures_hard_cutoff`."
            ),
        )

    def _prepare_enumeration(self):
        """
        Prepare enumeration process. Creates a dictionary that contains the site positions as keys
        and a list of the according elements that should be enumerated as values.
        """
        structure = self.inputs.structure
        if self.inputs.get("sites_to_enumerate"):
            to_enumerate = self.inputs.get("sites_to_enumerate").get_list()
            sites_to_enumerate = to_enumerate
        elif self.inputs.get("elements_to_enumerate"):
            to_enumerate = self.inputs.get("elements_to_enumerate").get_dict()
            sites_to_enumerate = []
            for kind in structure.get_site_kindnames():
                kinds = to_enumerate.get(kind, [kind])
                sites_to_enumerate.append(kinds)

        kind_names = enum_utils.get_kindnames(structure, to_enumerate)
        site_group = {}

        for site, site_species in zip(structure.sites, sites_to_enumerate):
            site_group[site.position] = [kind_names.index(specie) for specie in site_species]

        return site_group, kind_names

    def prepare_for_submission(self, folder):
        """
        Create input file.
        """
        coord_format = "{:.6f} {:.6f} {:.6f}"

        structure = self.inputs.structure
        input_content = [structure.get_formula(), "bulk"]

        for lat_vec in structure.cell:
            input_content.append("{:.6f} {:.6f} {:.6f}".format(*lat_vec))

        grouped_sites, kind_names = self._prepare_enumeration()
        input_content.append(f"{len(kind_names)}")
        input_content.append(f"{len(structure.sites)}")

        coords = []
        for pos, sp in grouped_sites.items():
            species_labels = "/".join([str(l0) for l0 in sorted(sp)])
            coords.append(f"{coord_format.format(*pos)} {species_labels}")
        input_content.extend(coords)

        input_content.append(
            f"{self.inputs.min_cell_size.value} {self.inputs.max_cell_size.value}"
        )
        input_content.append(f"{self.inputs.eps.value}")
        input_content.append("full")

        concentrations = self.inputs.get("concentration_restrictions")
        if concentrations:
            for kind in kind_names:
                input_content.append("{} {} {}".format(*concentrations[kind]))

        with folder.open(self.options.input_filename, "w", encoding="utf8") as handle:
            handle.write("\n".join(input_content))

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = [self.options.input_filename]

        calcinfo = CalcInfo()
        if isinstance(self.inputs.structures_to_return, aiida_orm.Int):
            calcinfo.append_text = "python3 {} {}".format(
                self.inputs.path_to_makeStr.value,
                self.inputs.structures_to_return.value,
            )
        else:
            structures_to_return = self.inputs.structures_to_return.get_list()
            if len(structures_to_return) == 2:
                calcinfo.append_text = "python3 {} {} {}".format(
                    self.inputs.path_to_makeStr.value,
                    self.inputs.structures_to_return.get_list()[0],
                    self.inputs.structures_to_return.get_list()[1],
                )
            elif len(structures_to_return) > 2:
                with folder.open("index_file.txt", "w", encoding="utf8") as handle:
                    handle.write("\n".join([str(idx) for idx in structures_to_return]))
                calcinfo.append_text = "python3 {} list -index_file {}".format(
                    self.inputs.path_to_makeStr.value, "index_file.txt"
                )
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = [self.options.output_filename]
        calcinfo.retrieve_temporary_list = [("vasp.*", ".", 1)]

        return calcinfo
