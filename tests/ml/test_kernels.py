"""Test custom kernels for scikit learn models."""

# Third party library imports
import pytest

# Internal library imports
from aim2dat.ml.transformers import StructureFFPrintTransformer
from aim2dat.ml.kernels import krr_ffprint_cosine, krr_ffprint_laplace


@pytest.mark.parametrize(
    "structure1,structure2,use_weights,gamma,ref_cosine,ref_laplace",
    [
        ("Cs2Te_62_prim", "CsK2Sb_225", False, None, -1.0, 0.980199),
        ("Cs2Te_62_prim", "Cs2Te_194_prim", False, 0.5, 0.256416, 0.830360),
        ("Cs2Te_62_prim", "Cs2Te_194_prim", True, None, 0.171190, 0.991746),
    ],
)
def test_ffprint_kernels(
    create_structure_collection_object,
    structure1,
    structure2,
    use_weights,
    gamma,
    ref_cosine,
    ref_laplace,
):
    """Test ffprint kernels for the kernel ridge regression model."""
    r_max = 5.0
    delta_bin = 0.1
    sigma = 0.5
    strct_c, _ = create_structure_collection_object([structure1, structure2])
    transf = StructureFFPrintTransformer(
        r_max=r_max, delta_bin=delta_bin, sigma=sigma, add_header=True, use_weights=use_weights
    )
    fingerprints = transf.fit_transform(strct_c)
    assert abs(krr_ffprint_cosine(fingerprints[0], fingerprints[1]) - ref_cosine) < 1.0e-5
    assert (
        abs(krr_ffprint_laplace(fingerprints[0], fingerprints[1], gamma=gamma) - ref_laplace)
        < 1.0e-5
    )
