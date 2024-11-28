"""Interface to the mofx database."""

# Third party library imports
from mofdb_client import fetch
from io import StringIO

# Internal library imports
from aim2dat.strct.strct import Structure


def _download_structures(
    mofid: str = None,
    mofkey: str = None,
    vf_min: float = None,
    vf_max: float = None,
    lcd_min: float = None,
    lcd_max: float = None,
    pld_min: float = None,
    pld_max: float = None,
    sa_m2g_min: float = None,
    sa_m2g_max: float = None,
    sa_m2cm3_min: float = None,
    sa_m2cm3_max: float = None,
    name: str = None,
    database: str = None,
    telemetry: bool = True,
    pressure_unit: str = None,
    loading_unit: str = None,
    limit: int = None,
) -> list:
    """Download entries from mofx database."""
    structures = []
    for entry in fetch(
        mofid,
        mofkey,
        vf_min,
        vf_max,
        lcd_min,
        lcd_max,
        pld_min,
        pld_max,
        sa_m2g_min,
        sa_m2g_max,
        sa_m2cm3_min,
        sa_m2cm3_max,
        name,
        database,
        telemetry,
        pressure_unit,
        loading_unit,
        limit,
    ):
        structures.append(_parse_entry(entry))

    return structures


def _parse_entry(entry) -> Structure:
    """Parse entry to structure list."""
    cif_entry = StringIO(entry.cif)
    structure = Structure.from_file(cif_entry, file_format="cif")
    structure.label = str(entry.name)
    structure._attributes = {
        "source_id": entry.id,
        "source": "mofdb-" + str(entry.id),
        "isotherms": entry.isotherms,
        "heats": entry.heats,
        "void_fraction": entry.void_fraction,
        "surface_area_m2g": entry.surface_area_m2g,
        "surface_area_m2cm3": entry.surface_area_m2cm3,
        "pld": entry.pld,
        "lcd": entry.lcd,
        "pxrd": entry.pxrd,
        "pore_size_distribution": entry.pore_size_distribution,
        "database": entry.database,
        "url": entry.url,
        "adsorbates": entry.adsorbates,
        "mofid": entry.mofid,
        "mofkey": entry.mofkey,
        "batch_number": entry.batch_number,
        "entry_repr": entry,
    }
    return structure
