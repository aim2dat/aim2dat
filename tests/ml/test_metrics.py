"""Test custom metrics for scikit learn models."""

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import StructureOperations
from aim2dat.ml.transformers import StructureFFPrintTransformer
from aim2dat.ml.metrics import ffprint_cosine


@pytest.mark.parametrize(
    "structure1,structure2,use_weights",
    [
        ("Cs2Te_62_prim", "CsK2Sb_225", False),
        ("Cs2Te_62_prim", "Cs2Te_194_prim", False),
        ("Cs2Te_62_prim", "Cs2Te_194_prim", True),
    ],
)
def test_ffprint_cosine(create_structure_collection_object, structure1, structure2, use_weights):
    """Test ffprint_cosine metric."""
    r_max = 5.0
    delta_bin = 0.1
    sigma = 0.5
    strct_c, _ = create_structure_collection_object([structure1, structure2])
    transf = StructureFFPrintTransformer(
        r_max=r_max, delta_bin=delta_bin, sigma=sigma, add_header=True, use_weights=use_weights
    )

    fingerprints = transf.fit_transform(strct_c)
    comp = StructureOperations(structures=strct_c).compare_structures_via_ffingerprint(
        structure1,
        structure2,
        r_max=r_max,
        delta_bin=delta_bin,
        sigma=sigma,
        use_weights=use_weights,
    )
    assert abs(ffprint_cosine(fingerprints[0], fingerprints[1]) - comp) < 1.0e-5
