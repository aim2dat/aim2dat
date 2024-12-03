"""Interface to the mofx database."""

# Third party library imports
from mofdb_client import fetch
from io import StringIO
from typing import List

# Internal library imports
from aim2dat.strct import Structure, StructureCollection


def _download_structures(
    mofid,
    mofkey,
    vf_min_max,
    lcd_min_max,
    pld_min_max,
    sa_m2g_min_max,
    sa_m2cm3_min_max,
    adsorbates,
    database,
    store_json,
    query_limit,
    pressure_unit,
    loading_unit,
) -> list:
    """Download entries from mofx database."""
    structures_collect = StructureCollection()
    name = None
    telemetry = None
    for entry in fetch(
        mofid,
        mofkey,
        vf_min_max[0],
        vf_min_max[1],
        lcd_min_max[0],
        lcd_min_max[1],
        pld_min_max[0],
        pld_min_max[1],
        sa_m2g_min_max[0],
        sa_m2g_min_max[1],
        sa_m2cm3_min_max[0],
        sa_m2cm3_min_max[1],
        name,
        database,
        telemetry,
        pressure_unit,
        loading_unit,
        query_limit,
    ):
        strct = _parse_entry(adsorbates, store_json, entry)
        if strct is not None:
            structures_collect.append_structure(strct)

    return structures_collect


def _parse_entry(adsorbates, store_json, entry) -> Structure:
    """Parse entry to structure list."""
    if adsorbates is None:
        adsorbates = [adsorb["name"] for adsorb in entry.json_repr["adsorbates"]]
    else:
        if not set(adsorbates) & set([adsorb["name"] for adsorb in entry.json_repr["adsorbates"]]):
            return None
    isotherms, heats = _get_isotherms_heats(adsorbates, entry)
    cif_entry = StringIO(entry.cif)
    # structure = Structure.from_file(cif_entry, file_format="cif", backend="internal")
    structure = Structure.from_file(cif_entry, file_format="cif")
    structure.label = str(entry.name)
    structure._extras = {
        "source_id": entry.id,
        "source": "mofdb",
        "database": entry.database,
        "url": entry.url,
        "name": entry.name,
        "mofid": entry.mofid,
        "mofkey": entry.mofkey,
    }
    if store_json:
        structure._extras.update(entry.json_repr)
    structure._attributes = {
        "void_fraction": entry.void_fraction,
        "surface_area_m2g": entry.surface_area_m2g,
        "surface_area_m2cm3": entry.surface_area_m2cm3,
        "pld": entry.pld,
        "lcd": entry.lcd,
        "pxrd": entry.pxrd,
        "pore_size_distribution": entry.pore_size_distribution,
    }
    if heats:
        structure._attributes.update({"heats": heats})
    if isotherms:
        structure._attributes.update({"isotherms": isotherms})
    return structure


def _get_isotherms_heats(adsorbates, entry) -> List[dict]:
    isotherms = {}
    heats = {}
    isotherms_data = entry.json_repr["isotherms"]
    heats_data = entry.json_repr["heats"]
    for adsorb in adsorbates:
        for heat in heats_data:
            adsorbs_heat = [ads["name"] for ads in heat["adsorbates"]]
            if adsorb in adsorbs_heat:
                heats.setdefault(adsorb, [])
                data = heat["isotherm_data"]
                if len(adsorbs_heat) > 1:
                    frac_adsorb = _get_frac_adsorption(data)
                else:
                    frac_adsorb = None
                    heat_dict = {
                        "adsorbates": adsorbs_heat,
                        "temperature": heat["temperature"],
                        "pressure": [press["pressure"] for press in data],
                        "total_adsorption": [
                            adsorb_data["total_adsorption"] for adsorb_data in data
                        ],
                        "pressureUnits": heat["pressureUnits"],
                        "adsorptionUnits": heat["adsorptionUnits"],
                    }
                if frac_adsorb is not None:
                    heat_dict["frac_adsorption"] = frac_adsorb
                heats[adsorb].append(heat_dict)
        for isotherm in isotherms_data:
            adsorbs_isot = [ads["name"] for ads in isotherm["adsorbates"]]
            if adsorb in adsorbs_isot:
                isotherms.setdefault(adsorb, [])
                data = isotherm["isotherm_data"]
                if len(adsorbs_isot) > 1:
                    frac_adsorb = _get_frac_adsorption(data)
                else:
                    frac_adsorb = None
                iso_dict = {
                    "adsorbates": adsorbs_isot,
                    "temperature": isotherm["temperature"],
                    "pressure": [press["pressure"] for press in data],
                    "total_adsorption": [adsorb_data["total_adsorption"] for adsorb_data in data],
                    "pressureUnits": isotherm["pressureUnits"],
                    "adsorptionUnits": isotherm["adsorptionUnits"],
                }
                if frac_adsorb is not None:
                    iso_dict["frac_adsorption"] = frac_adsorb
                isotherms[adsorb].append(iso_dict)
    return isotherms, heats


def _get_frac_adsorption(data) -> list:
    frac_adsorption = {}
    for d in data:
        for value in d["species_data"]:
            frac_adsorption.setdefault(value["name"], [])
            frac_adsorption[value["name"]].append(value["adsorption"])
    return frac_adsorption
