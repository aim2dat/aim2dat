"""Module that implements an interface to online databases and random crystal generation."""

# Standard library imports
import time
import math
import itertools
from typing import List, Union
import uuid


# Internal library imports
from aim2dat.strct.structure import Structure
from aim2dat.strct.structure_collection import StructureCollection
from aim2dat.ext_interfaces import _return_ext_interface_modules
from aim2dat.strct.constraints_mixin import ConstraintsMixin
import aim2dat.utils.print as utils_pr
from aim2dat.chem_f import transform_dict_to_str, transform_str_to_dict


def _update_import_details(import_details, provider, structures):
    if isinstance(structures, Structure):
        n_structures = 1
        elements = sorted(set(structures.elements))
    else:
        n_structures = len(structures)
        elements = structures.get_all_elements()
    if provider not in import_details:
        import_details[provider] = [n_structures, elements]
    else:
        import_details[provider][0] += n_structures
        import_details[provider][1] = sorted(set(import_details[provider][1] + elements))


class StructureImporter(ConstraintsMixin):
    """Imports structures from online databases."""

    def __init__(
        self, structures: StructureCollection = None, neglect_elemental_structures: bool = False
    ):
        """Initialize object."""
        if structures is None:
            structures = StructureCollection()
        self.structures = structures
        self.neglect_elemental_structures = neglect_elemental_structures

        self._import_details = {}

    def __str__(self):
        """
        Represent object as string.
        """
        output_str = utils_pr._print_title("Structure Collection") + "\n\n"
        for provider, details in self._import_details.items():
            output_str += utils_pr._print_subtitle("Imported from: " + provider) + "\n"
            output_str += "   - Number of structures: " + str(details[0]) + "\n"
            output_str += "   - Elements: " + "-".join(details[1]) + "\n\n"

        output_str += utils_pr._print_hline() + "\n\n"
        output_str += utils_pr._print_subtitle("Chemical element constraints")
        output_str += "\n"
        output_str += f"   Neglecting elemental structures: {self.neglect_elemental_structures}\n"
        if hasattr(self, "_conc_constraints") and len(self._conc_constraints) > 0:
            for element, constraint in self._conc_constraints.items():
                output_str += utils_pr._print_list(
                    "  " + element + ":",
                    ["min: " + str(constraint[0]), "max: " + str(constraint[1])],
                )
        # else:
        #     output_str += "   not set.\n"
        output_str += "\n"
        output_str += utils_pr._print_subtitle("Chemical formula constraints")
        output_str += "\n"
        if hasattr(self, "_formula_constraints") and len(self._formula_constraints) > 0:
            chemical_formulas = []
            for formula in self._formula_constraints:
                if "element_set" in formula:
                    chemical_formulas.append("-".join(formula["element_set"]))
                else:
                    formula_str = transform_dict_to_str(formula["formula"])
                    if formula["is_reduced"]:
                        formula_str += " (reduced)"
                    chemical_formulas.append(formula_str)
            output_str += utils_pr._print_list(" ", chemical_formulas)
        else:
            output_str += "   Not set.\n"
        output_str += "\n"
        output_str += utils_pr._print_subtitle("Attribute constraints")
        output_str += "\n"
        if hasattr(self, "_attr_constraints") and len(self._attr_constraints) > 0:
            for element, constraint in self._attr_constraints.items():
                output_str += utils_pr._print_list(
                    "  " + element + ":",
                    ["min: " + str(constraint[0]), "max: " + str(constraint[1])],
                )
        else:
            output_str += "   Not set.\n"
        output_str += "\n" + utils_pr._print_hline()
        return output_str

    @property
    def structures(self) -> StructureCollection:
        """Return the internal ``StructureCollection`` object."""
        return self._structures

    @structures.setter
    def structures(self, value: StructureCollection):
        if isinstance(value, StructureCollection):
            self._structures = value
        else:
            raise TypeError("`structures` needs to be of type `StructureCollection`.")

    def append_from_mp_by_id(
        self,
        entry_id: str,
        api_key: str,
        property_data: list = None,
        structure_type: str = "initial",
    ) -> Structure:
        """
        Append structure via the database-id.

        Parameters
        ----------
        entry_id : str
            Database id of the entry.
        api_key : str
            API key for the database, can be obtained here:
            https://www.materialsproject.org/dashboard
        property_data : list (optional)
            Extra data that is queried for each entry. The properties need to be passed as a list
            of strings (e.g. ``['el_band_structure', 'el_dos']`` to obtain the electronic band
            structure and the electronic density of states).
        structure_type : str (optional)
            Materials project includes the initial and final (relaxed) stucture in the database.
            The intial or final structure can be queried by setting this attribute
            to ``initial`` or ``final``, respectively.
        """
        if not isinstance(api_key, str):
            raise TypeError(
                "API key needs to be set. "
                "It can be obtained at https://www.materialsproject.org/dashboard"
            )
        if property_data is None:
            property_data = []
        if structure_type == "initial":
            property_data.append("initial_structure")
        backend_module = _return_ext_interface_modules("mp_openapi")
        entry = backend_module._download_structure_by_id(
            entry_id, api_key, structure_type, property_data
        )
        entry = Structure(**entry)
        self.structures.append_structure(entry)
        _update_import_details(self._import_details, "mp_openapi", entry)
        return entry

    def import_from_mp(
        self,
        formulas: Union[str, List[str]],
        api_key: str,
        compatible_only: bool = True,
        conv_unit_cell: bool = False,
        property_data: list = [],
        structure_type: str = "initial",
    ) -> StructureCollection:
        """
        Import structures from the crystal database Materials Project using the pymatgen interface.

        Parameters
        ----------
        formulas : str or list of str
            List of chemical formulas or systems that are queried
            from the database. E.g. ``'Fe2O3'`` - defined chemical composition,
            ``'Cs'`` - all entries of elemental phases Cs, ``'Cs-Te'`` - all entries that
            exclusively contain the elements Cs and/or Te.
        api_key : str
            API key for the database, can be obtained here:
            https://www.materialsproject.org/dashboard
        compatible_only : bool (optional)
            Whether to only query compatible data. The default value is ``True``.
        conv_unit_cell : bool (optional)
            Query the conventional unit cell instead of the primitive unit cell. The default value
            is ``False``.
        property_data : list (optional)
            Extra data that is queried for each entry. The properties need to be passed as a list
            of strings (e.g. ``['el_bandstructure', 'el_dos']`` to obtain the electronic band
            structure and the electronic density of states). The default value is ``[]``.
        structure_type : str (optional)
            Materials project includes the initial and final (relaxed) structure in the database.
            The initial or final structure can be queried by setting this attribute
            to ``initial`` or ``final``, respectively. The default setting is ``initial``.
        """
        if not isinstance(api_key, str):
            raise TypeError(
                "API key needs to be set. "
                "It can be obtained at https://www.materialsproject.org/dashboard"
            )
        if structure_type not in ["initial", "final"]:
            raise ValueError("`structure_type` must be 'initial' or 'final'.")

        download_kwargs = {
            "mp_api_key": api_key,
            "inc_structure": structure_type,
            "property_data": list(
                set(
                    [
                        "band_gap",
                        "spacegroup",
                        "total_magnetization",
                        "formation_energy_per_atom",
                        "e_above_hull",
                        "icsd_ids",
                    ]
                    + property_data
                )
            ),
            "conventional_unit_cell": conv_unit_cell,
            "compatible_only": compatible_only,
        }
        if structure_type == "initial":
            download_kwargs["property_data"].append("initial_structure")
        return self._import_from_odb("mp_openapi", formulas, {}, download_kwargs)

    def import_from_mofxdb(
        self,
        name: str = None,
        mofid: str = None,
        mofkey: str = None,
        vf_range: tuple = (None, None),
        lcd_range: tuple = (None, None),
        pld_range: tuple = (None, None),
        sa_m2g_range: tuple = (None, None),
        sa_m2cm3_range: tuple = (None, None),
        adsorbates: Union[str, List[str]] = None,
        database: str = None,
        store_uptake: bool = False,
        query_limit: int = 1000,
    ) -> StructureCollection:
        """
        Import structures from the MOFX database using the fetch function.
        If no parameters are set, the whole dabatabse will be imported.

        Parameters
        ----------
        name : str (optional)
            Name of the MOF in the corresponding DB.
        mofid : str (optional)
            The unique ID for the MOF.
        mofkey : str (optional)
            A specific key, often used for subcategorization or indexing.
        vf_range : tuple (optional)
            Minimum and maximum values for the void fraction (VF).
        lcd_range : tuple (optional)
            Minimum and maximum values for the largest cavity diameter (LCD).
        pld_range : tuple (optional)
            Minimum and maximum values for the pore limiting diameter (PLD).
        sa_m2g_range : tuple (optional)
            Minimum and maximum values for the surface area (SA) per gram (m^3/g^3).
        sa_m2cm3_range : tuple (optional)
            Minimum and maximum values for the surface area (SA) in square meters (m^2/cm^3).
        adsorbates: str or list of str (optional)
            The adsorbates included for heat and isotherm analysis.  E.g. ``'Hydrogen'``.
            If not defined, all adsorbates studied are considered.
        database : str (optional)
            The database from which MOF information is retrieved. E.g. ``'CoREMOF 2019'``.
        store_uptake : bool (optional)
            If ``True``, uptake data is stored in ``extras``.
        query_limit : int (optional)
            The maximum number of results to retrieve for the query.
        """
        adsorb_names = {
            "Argon": ("argon", "ar"),
            "CarbonDioxide": ("carbondioxide", "co2"),
            "Hydrogen": ("hydrogen", "h2"),
            "Krypton": ("krypton", "kr"),
            "Methane": ("methane", "ch4"),
            "Nitrogen": ("nitrogen", "n2"),
            "Xenon": ("xenon", "xe"),
        }
        if adsorbates is not None:
            if isinstance(adsorbates, str):
                adsorbates = [adsorbates]
            for idx, ads in enumerate(adsorbates):
                for ads_key, values in adsorb_names.items():
                    if ads.lower() in values:
                        adsorbates[idx] = ads_key
            adsorbates = set(adsorbates)

        if name or mofid or mofkey:
            query_limit = 1
        backend_module = _return_ext_interface_modules("mofxdb")
        structures_collect = backend_module._download_structures(
            name,
            mofid,
            mofkey,
            vf_range,
            lcd_range,
            pld_range,
            sa_m2g_range,
            sa_m2cm3_range,
            adsorbates,
            database,
            store_uptake,
            query_limit,
        )
        self.structures += structures_collect
        return structures_collect

    def import_from_oqmd(
        self, formulas: Union[str, List[str]], query_limit=1000
    ) -> StructureCollection:
        """
        Import from the open quantum materials database.

        Parameters
        ----------
        formulas : str or list of str
            List of chemical formulas or systems that are queried
            from the database. E.g. ``'Fe2O3'`` - defined chemical composition,
            ``'Cs'`` - all entries of elemental phases Cs, ``'Cs-Te'`` - all entries that
            exclusively contain the elements Cs and/or Te.
        query_limit : int (optional)
            Maximum number of crystals that are queried.
        """
        return self._import_from_odb("oqmd", formulas, {"query_limit": query_limit}, {})

    def import_from_optimade(
        self,
        formulas: Union[str, List[str]],
        database_id: str,
        api_version: int = 1,
        optimade_url: str = "https://providers.optimade.org/providers.json",
        timeout: float = 60.0,
    ) -> StructureCollection:
        """
        Import crystal structures using the optimade-API.

        The provider information is queried using the page:
        https://providers.optimade.org/providers.json.

        Parameters
        ----------
        formulas : str or list of str
            List of chemical formulas or systems that are queried
            from the database. E.g. ``'Fe2O3'`` - defined chemical composition,
            ``'Cs'`` - all entries of elemental phases Cs, ``'Cs-Te'`` - all entries that
            exclusively contain the elements Cs and/or Te.
        database_id : str
            Database used to query the data.
        api_version : int (optional)
            Version of the optimade API. The default value is ``1``.
        optimade_url : str (optional)
            Page used to obtain the provider information. The default value is
            ``'https://providers.optimade.org/providers.json'``.
        timeout : float (optional)
            Specifies the time to wait for response from the server. The default value is ``60.0``.
        """
        from warnings import warn

        warn(
            "This method needs to be considered experimental. It seems that the optimade "
            + "interface is unfortunately not yet commonly implemented for all databases.",
            UserWarning,
            2,
        )
        download_kwargs = {
            "optimade_url": optimade_url,
            "api_version": api_version,
            "database_id": database_id,
            "timeout": timeout,
        }
        return self._import_from_odb("optimade", formulas, {}, download_kwargs)

    def return_optimade_database_ids(
        self,
        api_version: int = 1,
        optimade_url: str = "https://providers.optimade.org/providers.json",
        timeout: float = 60.0,
    ) -> list:
        """
        Return a list of all ids of online databases that provide a base-url.

        Parameters
        ----------
        api_version : int (optional)
            Version of the optimade API. The default value is ``1``.
        optimade_url : str (optional)
            Page used to obtain the provider information. The default value is
            ``'https://providers.optimade.org/providers.json'``.
        timeout : float (optional)
            Specifies the time to wait for response from the server. The default value is ``60.0``.

        Returns
        -------
        list
            List of provider-ids.
        """
        backend_module = _return_ext_interface_modules("optimade")
        providers = backend_module._return_database_ids(api_version, optimade_url, timeout)
        return [id0 for id0, attr in providers.items() if attr["base_url"] is not None]

    def generate_random_crystals(
        self,
        formulas: Union[str, List[str]],
        excl_space_groups: list = [],
        tol_tuples: list = None,
        molecular: bool = False,
        dimensions: int = 3,
        bin_size: float = 0.1,
        max_atoms: int = 30,
        max_structures: int = 10,
        max_structures_per_cs: int = 10,
        max_structures_per_sg: int = 5,
        volume_factor: float = 1.0,
    ) -> StructureCollection:
        """
        Generate random crystals using the PyXtaL library.

        Parameters
        ----------
        formulas : str or list of str
            List of chemical formulas or systems that are queried
            from the database. E.g. ``'Fe2O3'`` - defined chemical composition,
            ``'Cs'`` - all entries of elemental phases Cs, ``'Cs-Te'`` - all entries that
            exclusively contain the elements Cs and/or Te.
        excl_space_groups : list (optional)
            Exclude one or more space groups.
        tol_tuples : None or list
            Tolerance tuples used to create the tolerance matrix. The default value is ``None``.
        molecular : bool (optional)
            Whether to generate molecular crystals. The default value is ``False``.
        dimensions : int
            Dimension of the crystal, possible values range from zero to three. The default value
            is ``3``.
        bin_size : float (optional)
            Size of bins that contain a certain number of structures. The default value is ``0.1``.
        max_atoms : int (optional)
            Maximum number of atoms per structure. The default value is ``30``.
        max_structures : int (optional)
            Maximum number of structures that are generated. The default value is ``10``.
        max_structures_per_cs : int (optional)
            Maximum number of structures that are generated per crystal system. The default value
            is ``10``.
        max_structures_per_sg : int (optional)
            Maximum number of structures that are generated per space group. The default value is
            ``5``.
        volume_factor : float (optional)
            Volume factor used to generate the crystal. The default value is ``1.0``.
        """
        if isinstance(formulas, str):
            formulas = [formulas]
        backend_module = _return_ext_interface_modules("pyxtal")
        tol_matrix = backend_module._pyxtal_tolerance_matrix(
            tuples=tol_tuples, molecular=molecular
        )

        structures_collect = StructureCollection()
        for formula in formulas:
            space_group_list = [0] * backend_module.NR_OF_SPACE_GROUPS[dimensions]
            crystal_sys_list = [0] * len(backend_module.SPACE_GROUP_LIMITS[dimensions])

            formula_dict = transform_str_to_dict(formula)
            unspecified_quantity = "-"
            if any(quantity == unspecified_quantity for quantity in formula_dict.values()):
                formula_series = self._create_formula_series(list(formula_dict.keys()), max_atoms)
                structures = backend_module._process_element_set(
                    list(formula_dict.keys()),
                    formula_series,
                    bin_size,
                    tol_matrix,
                    space_group_list,
                    crystal_sys_list,
                    molecular,
                    dimensions,
                    excl_space_groups,
                    max_structures,
                    max_structures_per_cs,
                    max_structures_per_sg,
                    volume_factor,
                )
            else:
                if not self._check_attribute_constraints(
                    {"chem_formula": formula_dict, "label": ""},
                    raise_error=False,
                    print_message=True,
                ):
                    continue
                atoms_per_f_unit = sum(formula_dict.values())
                formulas0 = []
                for counter in range(math.floor(max_atoms / atoms_per_f_unit)):
                    multiple = counter + 1
                    formulas0.append({el: value * multiple for el, value in formula_dict.items()})
                structures = backend_module._create_crystals(
                    formulas0,
                    tol_matrix,
                    space_group_list,
                    crystal_sys_list,
                    molecular,
                    dimensions,
                    excl_space_groups,
                    max_structures,
                    max_structures_per_cs,
                    max_structures_per_sg,
                    volume_factor,
                )
            for strct_idx, structure in enumerate(structures):
                label = "pyxtal_" + uuid.uuid4().hex
                structures_collect._add_structure(label, Structure(**structure), False)
        _update_import_details(self._import_details, "PyXtaL", structures_collect)
        self.structures += structures_collect
        return structures_collect

    def _import_from_odb(
        self,
        provider,
        formulas,
        query_kwargs,
        download_kwargs,
    ):
        backend_module = _return_ext_interface_modules(provider)
        if isinstance(formulas, str):
            formulas = [formulas]

        structures = StructureCollection()
        for formula in formulas:
            queries = self._create_odb_queries(backend_module, formula, **query_kwargs)
            for query in queries:
                time.sleep(0.1)
                entries = backend_module._download_structures(query, **download_kwargs)
                for entry in entries:
                    entry = Structure(**entry)
                    if entry.label in self.structures.labels:
                        print(f"Entry for {entry.label} already imported.")
                        continue
                    if self._apply_constraint_checks(entry, False):
                        structures.append_structure(entry)
                time.sleep(0.1)
        if provider == "optimade":
            provider += "-" + download_kwargs["database_id"]
        _update_import_details(self._import_details, provider, structures)
        self.structures += structures
        return structures

    def _create_odb_queries(self, backend_module, formula_str, **kwargs):
        """Create query arguments."""
        el_phase_query_args = getattr(backend_module, "_elemental_phase_query_args")
        el_set_query_args = getattr(backend_module, "_element_set_query_args")
        formula_query_qrgs = getattr(backend_module, "_formula_query_args")
        queries = []
        formula_dict = transform_str_to_dict(formula_str)

        if "-" in formula_str:
            elements = formula_str.split("-")
            for length in range(len(elements)):
                for subset in itertools.combinations(elements, length + 1):
                    if length == 0 and not self.neglect_elemental_structures:
                        queries.append(el_phase_query_args(subset[0], **kwargs))
                    elif length > 0:
                        queries.append(el_set_query_args(subset, length, **kwargs))
        elif len(formula_dict.keys()) > 1 or not self.neglect_elemental_structures:
            queries.append(formula_query_qrgs(formula_str, **kwargs))
        return queries

    def _create_formula_series(self, elements, max_atoms):
        """
        Create a list of chemical formulas.
        """
        formulas = []
        concentration_list = []
        distances = []

        rng = list(range(max_atoms + 1)) * len(elements)
        for permutation in itertools.permutations(rng, len(elements)):
            if 0 < sum(permutation) <= max_atoms:
                if self.neglect_elemental_structures and any(
                    nr_el == max_atoms for nr_el in permutation
                ):
                    continue
                concentration = tuple([qu_el / sum(permutation) for qu_el in permutation])
                if concentration in concentration_list:
                    continue
                formula = {el: qu_el for el, qu_el in zip(elements, permutation) if qu_el > 0}

                for nat_idx in range(1, max_atoms + 1):
                    if sum(permutation) * nat_idx > max_atoms:
                        break

                    formula = {
                        el: qu_el * nat_idx
                        for el, qu_el in zip(elements, permutation)
                        if qu_el > 0
                    }

                    if not self._check_concentration_constraints(
                        {"chem_formula": formula, "label": ""},
                        print_message=False,
                        raise_error=False,
                    ):
                        continue
                    if not self._check_chem_formula_constraints(
                        {"chem_formula": formula, "label": ""},
                        print_message=False,
                        raise_error=False,
                    ):
                        continue

                    formulas.append(formula)
                    concentration_list.append(concentration)
        zipped = list(zip(concentration_list, formulas))
        zipped.sort(key=lambda point: point[0])
        concentration_list, formulas = zip(*zipped)

        for conc_idx in range(len(concentration_list) - 1):
            conc1 = concentration_list[conc_idx]
            conc2 = concentration_list[conc_idx + 1]
            squared_dist = sum([(c0 - c1) ** 2.0 for c0, c1 in zip(conc1[:-1], conc2[:-1])])
            if squared_dist > 0.0:
                distances.append(math.sqrt(squared_dist))
        print(
            f"Created {len(distances) + 1} different concentrations and "
            f"{len(concentration_list)} formulas."
        )
        print(f"Minimum distance: {round(min(distances), 4)}")
        print(f"Maximum distance: {round(max(distances), 4)}")
        print(f"Average distance: {round(sum(distances)/len(distances), 4)}")
        # return {conc: form for conc, form in zip(concentration_list, formulas)}
        return list(zip(concentration_list, formulas))
