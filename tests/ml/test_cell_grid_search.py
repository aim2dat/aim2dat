"""Test CellGridSearch class."""

# Standard library imports
import os

# Third party library imports
import pytest
from sklearn.neighbors import KNeighborsRegressor

# Internal library imports
from aim2dat.strct import Structure
from aim2dat.io import read_yaml_file
from aim2dat.ml.cell_grid_search import CellGridSearch
from aim2dat.ml.transformers import StructureFFPrintTransformer

REF_PATH = os.path.dirname(__file__) + "/cell_grid_search_ref/"


@pytest.mark.parametrize(
    "system", ["Cs6Te31_2", "CsTe65_3", "Cs2Te44_38", "Te13_123", "Te45_152", "Te10_229"]
)
def test_cell_grid_search_target_structure(system, structure_comparison):
    """Test CellGridSearch class using a target structure."""
    ref = read_yaml_file(REF_PATH + system + ".yaml")

    cell_grid_search = CellGridSearch(**ref["attributes"])
    cell_grid_search.set_initial_structure(Structure(**ref["initial_structure"]))
    cell_grid_search.set_target_structure(Structure(**ref["target_structure"]))
    assert abs(cell_grid_search.return_initial_score() - ref["initial_score"]) < 1e-5
    max_score, max_params = cell_grid_search.fit()
    assert abs(max_score - ref["max_score"]) < 1e-5
    assert max_params == ref["max_parameters"]
    cell_grid_search.return_search_space()


def test_cell_grid_search_model(create_structure_collection_object, structure_comparison):
    """Test CellGridSearch class using a model and transformer."""
    r_max = 5.0
    delta_bin = 0.1
    sigma = 0.5

    strct_c, _ = create_structure_collection_object(
        os.path.dirname(__file__)
        + "/train_test_split_crystals_ref/PBE_CSP_Cs-Te_crystal-preopt_wo_dup.h5"
    )
    structure = strct_c[300]
    strct_c = strct_c[:100]
    targets = [strct["attributes"]["stability"] for strct in strct_c]

    transf = StructureFFPrintTransformer(
        r_max=r_max, delta_bin=delta_bin, sigma=sigma, add_header=False, use_weights=True
    )
    features = transf.fit_transform(strct_c)

    sklearn_model = KNeighborsRegressor()
    sklearn_model.fit(features, targets)

    grid_search = CellGridSearch()
    grid_search.set_model(sklearn_model, transformer=transf)
    grid_search.set_initial_structure(structure)
    assert abs(grid_search.return_initial_score() - 0.1178956) < 1.0e-5
    score, params = grid_search.fit()
    assert abs(score - 0.0688410) < 1.0e-5
    assert params == [1.0, 0.8, 0.8, 1.0, 0.9, 1.0]
    structure_comparison(
        grid_search.get_optimized_structure(),
        dict(read_yaml_file(REF_PATH + "model_ref_structure.yaml")),
    )
