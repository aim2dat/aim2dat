"""Interface to the open quantum materials database."""

# Third party library imports
import qmpy_rester as qr

# Internal library imports
from aim2dat.chem_f import transform_str_to_dict
import aim2dat.utils.space_groups as utils_sg


def _formula_query_args(formula, query_limit):
    """Create formula query for the oqmd database."""
    return {"limit": str(query_limit), "composition": formula}


def _elemental_phase_query_args(element, query_limit):
    """Create elemental phase query for the oqmd database."""
    return _formula_query_args(element, query_limit)


def _element_set_query_args(el_set, length, query_limit):
    """Create element set query for the oqmd database."""
    return {
        "limit": str(query_limit),
        "ntypes": str(len(el_set)),
        "element_set": ",".join(el_set),
    }


def _download_structures(query):
    """Process query and convert entries for oqmd API."""
    entries = []
    with qr.QMPYRester() as oqmd_rester:
        query_result = oqmd_rester.get_oqmd_phases(**query, verbose=False)
        if query_result is not None and "data" in query_result:
            for entry in query_result["data"]:
                entries.append(_parse_entry(entry))
    return entries


def _parse_entry(entry):
    """Parse entry to structure dict."""

    def is_plus_u(formula):
        plus_u_candidates = [
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Th",
            "U",
            "Np",
            "Pu",
        ]
        return any(element in plus_u_candidates for element in formula.keys())

    entry_id = entry.get("entry_id")
    icsd_ids = [entry.get("icsd_id")] if entry.get("icsd_id") is not None else []
    formula = transform_str_to_dict(entry.get("composition"))

    functional = "PBE"
    if is_plus_u(formula):
        functional += "+U"

    positions_scaled = []
    elements = []
    # cell_np = np.transpose(entry.get("unit_cell"))

    for atom in entry.get("sites"):
        symbol, position_str = atom.split("@", 1)
        elements.append(symbol.strip())
        positions_scaled.append([float(x) for x in position_str.split()])

    structure = {
        "label": "OQMD_" + str(entry_id),
        "cell": entry.get("unit_cell"),
        "pbc": [True, True, True],
        "elements": elements,
        "positions": positions_scaled,
        "is_cartesian": False,
        "attributes": {
            "source_id": str(entry_id),
            "source": "OQMD",
            "icsd_ids": icsd_ids,
            "formation_energy": {"value": entry.get("delta_e", None), "unit": "eV/atom"},
            "stability": {"value": entry.get("stability", None), "unit": "eV/atom"},
            "functional": functional,
            "band_gap": {"value": entry.get("band_gap", None), "unit": "eV"},
            "space_group": utils_sg.transform_to_nr(entry.get("spacegroup", None)),
        },
    }
    return structure
