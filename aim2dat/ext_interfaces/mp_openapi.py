"""Interface to the materials project online database using the openapi."""

# Standard library imports
import platform
import sys
from typing import List, Tuple
import json
import gzip

# Third party library imports
import requests
import base64
import zlib
import msgpack
import boto3
from botocore import UNSIGNED
from botocore.config import Config


def _formula_query_args(formula: str) -> dict:
    """Create formula query for the mp database."""
    return {"formula": formula}


def _elemental_phase_query_args(element: str) -> dict:
    """Create elemental phase query for the mp database."""
    return {"formula": element}


def _element_set_query_args(el_set: tuple, length) -> dict:
    """Create element set query for the mp database."""
    return {"chemsys": "-".join(el_set)}


def _download_structures(
    query: dict,
    mp_api_key: str,
    inc_structure: str,
    property_data: list,
    conventional_unit_cell: bool,
    compatible_only: bool,
) -> List[tuple]:
    """Process query and convert entries for optimade API."""
    entries = []
    link = "https://api.materialsproject.org/"
    query["_all_fields"] = True
    with requests.Session() as session:
        session.headers = {
            "x-api-key": mp_api_key,
            "user-agent": f"pymatgen/2023.10.4 (Python/{sys.version.split()[0]}"
            + f" {platform.system()}/{platform.release()})",
        }
        response = session.get(link + "heartbeat", timeout=60.0)
        response.raise_for_status()
        mp_result = response.json()
        mp_version = mp_result["db_version"]

        response = session.get(link + "materials/summary", timeout=60.0, params=query)
        response.raise_for_status()
        query_result = response.json()
        for entry in query_result["data"]:
            additional_data = _retrieve_additional_data(
                session, link, entry["material_id"], property_data
            )
            structure = _parse_entry(entry, additional_data, mp_version, inc_structure)
            if structure is not None:
                entries.append(structure)
    return entries


def _download_structure_by_id(
    mp_id: str, mp_api_key: str, structure_type: str, property_data: list
) -> dict:
    session, mp_version, link = _open_requests_session(mp_api_key)
    response = session.get(
        link + "materials/summary/",
        timeout=60.0,
        params={"material_ids": mp_id, "_all_fields": True},
    )
    entry = response.json()
    if "error" in entry:
        raise ValueError(entry["error"])
    entry = entry["data"][0]
    additional_data = _retrieve_additional_data(session, link, entry["material_id"], property_data)
    return _parse_entry(entry, additional_data, mp_version, structure_type)


def _open_requests_session(mp_api_key: str) -> Tuple[requests.Session, str, str]:
    link = "https://api.materialsproject.org/"
    with requests.Session() as session:
        session.headers = {
            "x-api-key": mp_api_key,
            "user-agent": f"pymatgen/2023.10.4 (Python/{sys.version.split()[0]}"
            + " {platform.system()}/{platform.release()})",
        }
        response = session.get(link + "heartbeat", timeout=60.0)
        response.raise_for_status()
        mp_result = response.json()
        return session, mp_result["db_version"], link


def _parse_entry(
    entry: dict, additional_data: dict, mp_version: str, structure_type: str
) -> tuple:
    if structure_type == "initial":
        mp_structure = additional_data.pop("initial_structures")[-1]
    else:
        mp_structure = entry["structure"]

    structure = {
        "label": "mp_" + entry["material_id"],
        "cell": mp_structure["lattice"]["matrix"],
        "pbc": [True, True, True],
        "elements": [site["species"][0]["element"] for site in mp_structure["sites"]],
        "positions": [site["xyz"] for site in mp_structure["sites"]],
        "is_cartesian": True,
        "attributes": {
            "source_id": entry["material_id"],
            "source": "MP_" + mp_version,
            "theoretical": entry["theoretical"],
            "formation_energy": {
                "value": float(entry["formation_energy_per_atom"]),
                "unit": "eV/atom",
            },
            "stability": {
                "value": float(entry["energy_above_hull"]),
                "unit": "eV/atom",
            },
            # "functional": thermo_data["energy_type"],
            "band_gap": {
                "value": float(entry["band_gap"]),
                "unit": "eV",
            },
            "space_group": entry["symmetry"]["number"],
            # "magnetic_moment": None,
        },
        "extras": {},
    }
    for extra_prop, data in additional_data.items():
        convert_fct = globals()["_convert_" + extra_prop]
        structure["extras"][extra_prop] = convert_fct(data)
    return structure


def _retrieve_additional_data(
    session: requests.Session, link: str, mp_id: str, property_data: list
) -> dict:
    def retrieve_mp_object(session, link, link_suffix, params, unpack_obj):
        params.update({"_all_fields": True})
        response = session.get(
            link + link_suffix,
            params=params,
        )
        response.raise_for_status()
        mp_obj = response.json()
        if unpack_obj:
            b64_bytes = base64.b64decode(mp_obj["data"][0], validate=True)
            packed_bytes = zlib.decompress(b64_bytes)
            return msgpack.unpackb(packed_bytes, raw=False)
        else:
            return mp_obj["data"]

    def retrieve_mp_s3_object(prefix, task_id):
        s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
        obj = s3.Object("materialsproject-parsed", f"{prefix}/{task_id}.json.gz")
        with gzip.GzipFile(fileobj=obj.get()["Body"]) as gzipfile:
            content = gzipfile.read()
        return json.loads(content)

    additional_data = {}
    el_structure_data = {"bandstructure": None, "dos": None}
    if "el_band_structure" in property_data or "el_dos" in property_data:
        response = session.get(
            link + "materials/electronic_structure/",
            params={"_fields": "bandstructure,dos", "material_ids": mp_id},
        )
        response.raise_for_status()
        el_structure_data = response.json()["data"][0]
    if "initial_structure" in property_data:
        structure_data = retrieve_mp_object(
            session, link, "materials/core/", {"material_ids": mp_id}, False
        )
        additional_data["initial_structures"] = structure_data[0]["initial_structures"]
    if "el_band_structure" in property_data and el_structure_data["bandstructure"] is not None:
        task_id = None
        for bs_data in el_structure_data["bandstructure"].values():
            if bs_data is not None:
                task_id = bs_data["task_id"]
                break
        if task_id is not None:
            additional_data["el_band_structure"] = retrieve_mp_s3_object("bandstructures", task_id)
    if "ph_band_structure" in property_data or "ph_dos" in property_data:
        phonon_data = retrieve_mp_object(
            session, link, "materials/phonon/", {"material_ids": mp_id}, False
        )
        if phonon_data[0].get("phonon_bandstructure"):
            additional_data["ph_band_structure"] = phonon_data[0]["phonon_bandstructure"]
        if phonon_data[0].get("phonon_dos"):
            additional_data["ph_dos"] = phonon_data[0]["phonon_dos"]
    if "el_dos" in property_data and el_structure_data["dos"] is not None:
        additional_data["el_dos"] = retrieve_mp_s3_object("dos", task_id)
    if "xas_spectra" in property_data:
        additional_data["xas_spectra"] = retrieve_mp_object(
            session, link, "materials/xas", {"material_ids": mp_id}, False
        )
    return additional_data


def _convert_el_band_structure(band_structure: dict) -> dict:
    bands_data = band_structure["data"]
    bands_dict = {
        "kpoints": [],
        "path_labels": [],
        "unit_y": "eV",
    }

    # Parse k-points:
    label_keys = list(bands_data["labels_dict"].keys())
    label_values = list(bands_data["labels_dict"].values())
    for kpoint_idx, kpoint in enumerate(bands_data["kpoints"]):
        bands_dict["kpoints"].append(kpoint)
        if kpoint in label_values:
            bands_dict["path_labels"].append([kpoint_idx, label_keys[label_values.index(kpoint)]])

    # Parse bands:
    if bands_data["is_spin_polarized"]:
        print("Spin-polarized band structures are not yet supported.")
    else:
        bands_dict["bands"] = [
            [
                float(bands_data["bands"]["1"][band][kpoint]) - bands_data["efermi"]
                for band in range(len(bands_data["bands"]["1"]))
            ]
            for kpoint in range(len(bands_data["bands"]["1"][0]))
        ]
    return bands_dict


def _convert_ph_band_structure(band_structure: dict) -> dict:
    band_structure["is_spin_polarized"] = False
    band_structure["kpoints"] = band_structure.pop("qpoints")
    band_structure["efermi"] = 0.0
    band_structure["bands"] = {"1": band_structure.pop("bands")}
    return _convert_el_band_structure({"data": band_structure})


def _convert_el_dos(dos: dict) -> dict:
    energy = [e0 - dos["data"]["efermi"] for e0 in dos["data"]["energies"]]
    dos_dict = {
        "pdos": {"energy": energy, "pdos": []},
        "tdos": {
            "energy": energy,
            "tdos": dos["data"]["densities"]["1"],
            "unit_x": "eV",
        },
    }
    chem_formula = {}
    for site, pdos_data in zip(dos["data"]["structure"]["sites"], dos["data"]["pdos"]):
        single_pdos = {"element": site["species"][0]["element"]}
        if single_pdos["element"] in chem_formula:
            chem_formula[single_pdos["element"]] += 1
        else:
            chem_formula[single_pdos["element"]] = 1
        single_pdos["kind"] = single_pdos["element"] + str(chem_formula[single_pdos["element"]])
        if isinstance(pdos_data, dict):
            for orbital, orb_data in pdos_data.items():
                single_pdos[orbital] = orb_data["densities"]["1"]
        else:
            single_pdos["total"] = pdos_data
        dos_dict["pdos"]["pdos"].append(single_pdos)
    return dos_dict


def _convert_ph_dos(dos: dict) -> dict:
    dos["energies"] = dos.pop("frequencies")
    dos["densities"] = {"1": dos.pop("densities")}
    dos["efermi"] = 0.0
    return _convert_el_dos({"data": dos})


def _convert_xas_spectra(xas_spectra: dict) -> dict:
    spectra_dict = {}
    for xas_spectrum in xas_spectra:
        spectra_dict[
            xas_spectrum["spectrum"]["absorbing_element"]
            + "_"
            + xas_spectrum["spectrum"]["edge"]
            + "_"
            + xas_spectrum["spectrum"]["spectrum_type"]
        ] = {
            "x_values": xas_spectrum["spectrum"]["x"],
            "y_values": xas_spectrum["spectrum"]["y"],
            "unit_x": "eV",
        }
    return spectra_dict
