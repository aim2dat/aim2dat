"""Interface to the mofx database."""

# Third party library imports
from mofdb_client import fetch

# Internal library imports
from aim2dat.strct import Structure, StructureCollection


def _download_structures(
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
) -> list:
    """Download entries from mofx database."""
    structures_collect = StructureCollection()
    for entry in fetch(
        mofid=mofid,
        mofkey=mofkey,
        vf_min=vf_range[0],
        vf_max=vf_range[1],
        lcd_min=lcd_range[0],
        lcd_max=lcd_range[1],
        pld_min=pld_range[0],
        pld_max=pld_range[1],
        sa_m2g_min=sa_m2g_range[0],
        sa_m2g_max=sa_m2g_range[1],
        sa_m2cm3_min=sa_m2cm3_range[0],
        sa_m2cm3_max=sa_m2cm3_range[1],
        name=name,
        database=database,
        telemetry=None,
        pressure_unit="bar",
        loading_unit="mg/g",
        limit=query_limit,
    ):
        strct = _parse_entry(adsorbates, store_uptake, entry)
        if strct is not None:
            structures_collect.append_structure(strct)

    return structures_collect


def _parse_entry(adsorbates, store_uptake, entry) -> Structure:
    """Parse entry to structure list."""
    entry = entry.json_repr
    if adsorbates is None:
        adsorbates = [adsorb["name"] for adsorb in entry["adsorbates"]]
    elif all([adsorb["name"] not in adsorbates for adsorb in entry["adsorbates"]]):
        return None

    attributes = {
        "mofid": entry["mofid"],
        "mofkey": entry["mofkey"],
        "source_id": entry["id"],
        "source": f"MOFX-DB_{entry['mofdb_version']}-{entry['database']}",
        "void_fraction": _value_none_check(entry["void_fraction"], None),
        "surface_area_m2g": _value_none_check(entry["surface_area_m2g"], "m2/g"),
        "surface_area_m2cm3": _value_none_check(entry["surface_area_m2cm3"], "m2/cm3"),
        "pld": _value_none_check(entry["pld"], "angstrom"),
        "lcd": _value_none_check(entry["lcd"], "angstrom"),
        "pxrd": _value_none_check(entry["pxrd"], "angstrom"),
        "pore_size_distribution": _value_none_check(entry["pore_size_distribution"], None),
    }
    extras = {}
    if store_uptake:
        _get_isotherms_heats(adsorbates, entry, extras)
    return Structure.from_file(
        file_path=entry["cif"],
        file_format="cif",
        backend="internal",
        backend_kwargs={"strct_wrap": True},
        label=entry["name"],
        attributes=attributes,
        extras=extras,
    )


def _value_none_check(value, unit):
    if value is None:
        return None
    elif isinstance(value, list):
        return {"value": [float(val) for val in value], "unit": unit}
    else:
        return {"value": float(value), "unit": unit}


def _get_isotherms_heats(adsorbates, entry, extras):
    for data_type in ["isotherms", "heats"]:
        for data_set in entry[data_type]:
            ds_adsorbates = [ads["name"] for ads in data_set["adsorbates"]]
            if any(ads in ds_adsorbates for ads in adsorbates):
                data_set_list = extras.setdefault(data_type, [])
                conv_ds = {
                    "adsorbates": ds_adsorbates,
                    "temperature": _value_none_check(data_set["temperature"], "K"),
                    "pressure": _value_none_check(
                        [press["pressure"] for press in data_set["isotherm_data"]],
                        data_set["pressureUnits"],
                    ),
                    "total_adsorption": _value_none_check(
                        [
                            adsorb_data["total_adsorption"]
                            for adsorb_data in data_set["isotherm_data"]
                        ],
                        data_set["adsorptionUnits"],
                    ),
                    "simin": data_set["simin"],
                }
                if len(ds_adsorbates) > 1:
                    conv_ds["frac_adsorption"] = _get_frac_adsorption(
                        data_set["isotherm_data"], data_set["adsorptionUnits"]
                    )
                data_set_list.append(conv_ds)


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
