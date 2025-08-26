"""
Methods acting on a ``Structure`` object to calculate structural properties and features.
"""

from aim2dat.strct.ext_analysis.fragmentation import (
    calc_molecular_fragments,
)
from aim2dat.strct.ext_analysis.graphs import calc_graph
from aim2dat.strct.ext_analysis.prdf import calc_prdf
from aim2dat.strct.ext_analysis.ffprint_order_p import (
    calc_ffingerprint_order_p,
)
from aim2dat.strct.ext_analysis.warren_cowley_order_parameters import (
    calc_warren_cowley_order_p,
)
from aim2dat.strct.ext_analysis.planes import calc_planes
from aim2dat.strct.ext_analysis.dscribe_descriptors import (
    calc_interaction_matrix,
    calc_acsf_descriptor,
    calc_soap_descriptor,
    calc_mbtr_descriptor,
)
from aim2dat.strct.ext_analysis.h_bonds import calc_hydrogen_bonds


__all__ = [
    "calc_molecular_fragments",
    "calc_graph",
    "calc_prdf",
    "calc_ffingerprint_order_p",
    "calc_warren_cowley_order_p",
    "calc_planes",
    "calc_hydrogen_bonds",
    "calc_interaction_matrix",
    "calc_acsf_descriptor",
    "calc_soap_descriptor",
    "calc_mbtr_descriptor",
]
