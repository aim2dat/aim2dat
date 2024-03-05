"""
Methods acting on a ``Structure`` object to calculate structural properties and features.
"""

from aim2dat.strct.ext_analysis.fragmentation import (
    determine_molecular_fragments,
)
from aim2dat.strct.ext_analysis.graphs import create_graph
from aim2dat.strct.ext_analysis.prdf import calculate_prdf
from aim2dat.strct.ext_analysis.ffprint_order_p import (
    calculate_ffingerprint_order_p,
)
from aim2dat.strct.ext_analysis.warren_cowley_order_parameters import (
    calculate_warren_cowley_order_p,
)
from aim2dat.strct.ext_analysis.planes import calculate_planes
from aim2dat.strct.ext_analysis.dscribe_descriptors import (
    calculate_interaction_matrix,
    calculate_acsf_descriptor,
    calculate_soap_descriptor,
    calculate_mbtr_descriptor,
)


__all__ = [
    "determine_molecular_fragments",
    "create_graph",
    "calculate_prdf",
    "calculate_ffingerprint_order_p",
    "calculate_warren_cowley_order_p",
    "calculate_planes",
    "calculate_interaction_matrix",
    "calculate_acsf_descriptor",
    "calculate_soap_descriptor",
    "calculate_mbtr_descriptor",
]
