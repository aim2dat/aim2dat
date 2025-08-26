"""Interface to the optimade interface to query structures from online databases."""

# Third party library imports
import requests

# Internal library imports
from aim2dat.chem_f import transform_str_to_dict, transform_dict_to_str
import aim2dat.utils.space_groups as utils_sg


def _formula_query_args(formula):
    """Create formula query for the optimade interface."""
    formula_dict = transform_str_to_dict(formula)
    formula_sorted = transform_dict_to_str(formula_dict, output_type="alphabetic")
    return f'chemical_formula_reduced="{formula_sorted}"'


def _elemental_phase_query_args(element):
    """Create elemental phase query for the optimade interface."""
    return f'elements HAS ALL "{element}" AND nelements=1'


def _element_set_query_args(el_set, length):
    """Create element set query for the optimade interface."""
    subset_str = '", "'.join(el_set)
    return f'elements HAS ALL "{subset_str}" AND nelements={length+1}'


def _download_structures(query, **kwargs):
    """Process query and convert entries for optimade API."""
    providers = _return_database_ids(
        kwargs["api_version"], kwargs["optimade_url"], kwargs["timeout"]
    )
    database_ids = [id0 for id0, attr in providers.items() if attr["base_url"] is not None]
    if kwargs["database_id"] not in database_ids:
        raise Exception(
            f"The database id `{kwargs['database_id']}` is not supported, the "
            f"supported databases are: " + ", ".join(database_ids) + "."
        )

    entries = []

    base_url = providers[kwargs["database_id"]]["base_url"]
    if base_url[-1] != "/":
        base_url += "/"
    link = base_url + f"v{kwargs['api_version']}/structures?filter="

    with requests.Session() as session:
        response = session.get(url=link + query, timeout=kwargs["timeout"])
        response.raise_for_status()
        query_result = response.json()
        for entry in query_result["data"]:
            structure = _parse_entry(entry, kwargs["database_id"])
            if structure is not None:
                entries.append(structure)

    return entries


def _return_database_ids(api_version, url, timeout):
    """
    Update the list of providers.
    """
    providers_url = {}
    with requests.Session() as session:
        response = session.get(url=url, timeout=timeout)
        providers_json = response.json()
        providers_url = {
            provider["id"]: provider["attributes"] for provider in providers_json.get("data")
        }

    providers = {}
    for prov_id, prov_attr in providers_url.items():
        if prov_attr.get("base_url"):
            url = prov_attr["base_url"] + f"/v{api_version}/links"
            with requests.Session() as session:
                try:
                    response = session.get(
                        url=url,
                        timeout=timeout,
                    )
                    provider_json = response.json()
                except (
                    ValueError,
                    requests.exceptions.ConnectTimeout,
                    requests.exceptions.SSLError,
                ):
                    from warnings import warn

                    warn(
                        f"Skipping '{prov_id}' database, cannot retrieve information from: {url}.",
                        UserWarning,
                        2,
                    )

                for database in provider_json["data"]:
                    if (
                        database["attributes"].get("link_type")
                        or database["attributes"].get("type")
                    ) == "child":
                        if database["id"] == prov_id:
                            providers[database["id"]] = database["attributes"]
                        else:
                            providers[f'{prov_id}.{database["id"]}'] = database["attributes"]
    return providers


def _parse_entry(entry, database_id):
    """Parse entry to structure dict."""
    if entry.get("attributes"):
        entry_attr = entry["attributes"]
    else:
        entry_attr = entry

    structure_attributes = ["lattice_vectors", "species_at_sites", "cartesian_site_positions"]
    if all(struct_attr in entry_attr for struct_attr in structure_attributes):
        if entry_attr.get("dimension_types"):
            pbc = [(direction == 1) for direction in entry_attr["dimension_types"]]
        else:
            pbc = [True, True, True]

        entry_id = str(entry["id"])
        label = "optimade-" + database_id + "_" + entry_id
        structure = {
            "label": label,
            "cell": entry_attr["lattice_vectors"],
            "pbc": pbc,
            "elements": entry_attr["species_at_sites"],
            "positions": entry_attr["cartesian_site_positions"],
            "is_cartesian": True,
            "attributes": {
                "source_id": entry_id,
                "source": "optimade-" + database_id,
            },
        }
        if database_id == "oqmd":
            structure["attributes"].update(_convert_extra_properties_oqmd(entry_attr))
        elif database_id == "odbx":
            structure["attributes"].update(_convert_extra_properties_odbx(entry_attr))
        return structure


def _convert_extra_properties_oqmd(entry):
    """Convert additional properties from the open quantum materials database."""
    extra_properties = {
        "formation_energy": float(entry.get("_oqmd_delta_e")),
        "stability": float(entry.get("_oqmd_stability")),
        "band_gap": float(entry.get("_oqmd_band_gap")),
        "space_group": utils_sg.transform_to_nr(entry.get("_oqmd_spacegroup")),
        "oqmd_id": entry.get("_oqmd_entry_id"),
    }
    return extra_properties


def _convert_extra_properties_odbx(entry):
    """Convert additional properties from odbx."""
    extra_properties = {
        "formation_energy": float(entry["_odbx_thermodynamics"].get("formation_energy")),
        "stability": float(entry["_odbx_thermodynamics"].get("hull_distance")),
        "functional": entry["_odbx_dft_parameters"].get("xc_functional"),
        "space_group": int(entry["_odbx_space_group"].get("number")),
    }
    return extra_properties
