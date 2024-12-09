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
) -> list:
    """Download entries from mofx database."""
    structures_collect = StructureCollection()
    name = None
    telemetry = None
    pressure_unit = "bar"
    loading_unit = "cm3(STP)/g"
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
            "void_fraction": _value_none_check(entry.void_fraction, ""),
            "surface_area_m2g": _value_none_check(entry.surface_area_m2g, "m2/g"),
            "surface_area_m2cm3": _value_none_check(entry.surface_area_m2cm3, "m2/cm3"),
            "pld": _value_none_check(entry.pld, "angstrom"),
            "lcd": _value_none_check(entry.lcd, "angstrom"),
            "pxrd": _value_none_check(entry.pxrd, "angstrom"),
            "pore_size_distribution": _value_none_check(entry.pore_size_distribution, ""),
        }
    if heats:
        structure._attributes.update({"heats": heats})
    if isotherms:
        structure._attributes.update({"isotherms": isotherms})
    return structure


def _value_none_check(value, unit):
    if value is None:
        return None
    elif isinstance(value, list):
        return {"value": [float(val) for val in value], "unit": unit}
    else:
        return {"value": float(value), "unit": unit}


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
                heat_dict = {
                    "adsorbates": adsorbs_heat,
                    "temperature": _value_none_check(heat["temperature"], "K"),
                    "pressure": _value_none_check(
                        [press["pressure"] for press in data], heat["pressureUnits"]
                    ),
                    "total_adsorption": _value_none_check(
                        [adsorb_data["total_adsorption"] for adsorb_data in data],
                        heat["adsorptionUnits"],
                    ),
                }
                if len(adsorbs_heat) > 1:
                    heat_dict["frac_adsorption"] = _get_frac_adsorption(
                        data, heat["adsorptionUnits"]
                    )
                heats[adsorb].append(heat_dict)
        for isotherm in isotherms_data:
            adsorbs_isot = [ads["name"] for ads in isotherm["adsorbates"]]
            if adsorb in adsorbs_isot:
                isotherms.setdefault(adsorb, [])
                data = isotherm["isotherm_data"]
                iso_dict = {
                    "adsorbates": adsorbs_isot,
                    "temperature": _value_none_check(isotherm["temperature"], "K"),
                    "pressure": _value_none_check(
                        [press["pressure"] for press in data], isotherm["pressureUnits"]
                    ),
                    "total_adsorption": _value_none_check(
                        [adsorb_data["total_adsorption"] for adsorb_data in data],
                        isotherm["adsorptionUnits"],
                    ),
                }
                if len(adsorbs_isot) > 1:
                    iso_dict["frac_adsorption"] = _get_frac_adsorption(
                        data, isotherm["adsorptionUnits"]
                    )
                isotherms[adsorb].append(iso_dict)
    return isotherms, heats


def _get_frac_adsorption(data, unit) -> dict:
    temp_dict = {}
    for d in data:
        for value in d["species_data"]:
            temp_dict.setdefault(value["name"], [])
            temp_dict[value["name"]].append(float(value["adsorption"]))
    frac_adsorption = {}
    for k, v in temp_dict.items():
        frac_adsorption[k] = _value_none_check(v, unit)
    return frac_adsorption
