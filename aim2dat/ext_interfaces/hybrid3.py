"""interface to the Hybrid 3 database."""

# Standard library imports
import os
import tempfile
import zipfile
import time

# Third party library imports
from requests import Session, ReadTimeout, HTTPError

# Internal library imports
from aim2dat.strct import Structure, SamePositionsError


def get_data(link, timeout=60.0, **kwargs):
    """Get data from REST API."""
    data = None
    counter = 0
    with Session() as session:
        while data is None and counter < 3:
            try:
                response = session.get(link, timeout=timeout, params=kwargs)
                response.raise_for_status()
                data = response
            except ReadTimeout:
                time.sleep(3)
    return data


def _get_all_datasets(timeout=60.0, page_size=5000):
    data = get_data(
        f"https://materials.hybrid3.duke.edu/materials/datasets/summary/?page_size={page_size}",
        timeout=timeout,
    ).json()
    results = data["results"]
    page_counter = 1
    while data["next"] is not None:
        data = get_data(data["next"], timeout=timeout).json()
        results += data["results"]
        page_counter += 1
        print(f"page {page_counter}/{len(results)}")
    return results


def _get_entry_data(pk, timeout=60.0):
    return get_data(
        f"https://materials.hybrid3.duke.edu/materials/datasets/{pk}/", timeout=timeout
    ).json()


def _get_entry_structures(pk, timeout=60.0):
    try:
        files = get_data(
            f"https://materials.hybrid3.duke.edu/materials/datasets/{pk}/files/", timeout=timeout
        )
    except HTTPError:
        return []

    structures = []
    with tempfile.TemporaryDirectory(prefix="hybrid3_", suffix="_tmp") as temp_dir:
        with open(temp_dir + "/files.zip", "wb") as f:
            f.write(files.content)
        with zipfile.ZipFile(temp_dir + "/files.zip", "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        for f in os.listdir(temp_dir + "/files"):
            if f.endswith(".cif"):
                kwargs = {"file_format": "cif", "strct_check_chem_formula": False}
            elif f.endswith(".in"):
                kwargs = {"file_format": "fhiaims_geometry"}
            else:
                continue

            try:
                structures.append(Structure.from_file(temp_dir + f"/files/{f}", **kwargs))
            except (ValueError, SamePositionsError):
                pass
    return structures
