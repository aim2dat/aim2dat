"""Interface to the materials project online database."""

# Standard library imports
from typing import List, Tuple

# Third party library imports
import pymatgen.ext.matproj as pymatgen_matproj

# Internal library imports
from aim2dat.strct.strct import Structure


def _formula_query_args(formula: str) -> str:
    """Create formula query for the mp database."""
    return formula


def _elemental_phase_query_args(element: str) -> str:
    """Create elemental phase query for the mp database."""
    return element


def _element_set_query_args(el_set: tuple, length) -> str:
    """Create element set query for the mp database."""
    return "-".join(el_set)


def _download_structures(query: dict, **kwargs) -> List[Tuple[str, Structure]]:
    """Process query and convert entries for the materials project API."""
    mp_key = kwargs.pop("mp_api_key")

    with pymatgen_matproj.MPRester(mp_key) as mp_rester:
        mp_version = mp_rester.get_database_version()
        print(f"connected with Materials Project Version {mp_version}.")

        rester_args = kwargs
        rester_args["chemsys_formula_id_criteria"] = query
        query_result = mp_rester.get_entries(**kwargs)
        entries = []

        # Convert entries
        for entry in query_result:
            entries.append(_parse_entry(entry, mp_version))
    return entries


# Add entry type
def _parse_entry(entry, mp_version: str) -> Tuple[str, Structure]:
    """Parse entry to structure dict."""
    functional = entry.parameters["pseudo_potential"]["functional"]
    if entry.parameters["is_hubbard"]:
        functional += "+U"
    calc_icsd = entry.data.get("icsd_ids") if entry.data.get("icsd_ids") is not None else []

    structure_pymatgen = entry.structure
    cell = structure_pymatgen.lattice.matrix
    positions = []
    elements = []
    for ii_at in range(len(structure_pymatgen)):
        elements.append(str(structure_pymatgen.sites[ii_at].specie))
        positions.append(structure_pymatgen.sites[ii_at].coords)
    structure = {
        "label": "mp_" + str(entry.entry_id),
        "cell": cell,
        "pbc": [True, True, True],
        "elements": elements,
        "positions": positions,
        "is_cartesian": True,
        "attributes": {
            "source_id": str(entry.entry_id),
            "source": "MP_" + mp_version,
            "formation_energy": {
                "value": float(entry.data.get("formation_energy_per_atom")),
                "unit": "eV/atom",
            },
            "stability": {"value": float(entry.data.get("e_above_hull")), "unit": "eV/atom"},
            "icsd_ids": list(set(calc_icsd + entry.data.get("icsd_ids"))),
            "functional": functional,
            "direct_band_gap": {"value": float(entry.data.get("band_gap")), "unit": "eV"},
            "space_group": int(entry.data.get("spacegroup").get("number")),
            "magnetic_moment": float(entry.data.get("total_magnetization")),
        },
    }
    return Structure(**structure)
