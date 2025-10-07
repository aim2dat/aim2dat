"""Test calculate-functions of the Structure class."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports:
from aim2dat.strct import Structure, SurfaceGeneration
from aim2dat.strct.surface_utils import _transform_slab_to_primitive
from aim2dat.strct.analysis.brillouin_zone_2d import _get_kpath
from aim2dat.io import read_yaml_file

STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
REF_PATH = os.path.dirname(__file__) + "/analysis/"


def test_calc_distance_angle_dihedral_errors():
    """Test correct error handling on site indices input."""
    strct = Structure.from_file(STRUCTURES_PATH + "Benzene.yaml", backend="internal")
    with pytest.raises(ValueError) as error:
        strct.calc_distance([1] * 3, [0, 2, 20])
    assert str(error.value) == "`site_index` needs to be smaller than the number of sites."
    with pytest.raises(TypeError) as error:
        strct.calc_distance([1.0] * 3, [0, 2, 3])
    assert str(error.value) == "`site_index` must be of type int, list, tuple, np.ndarray or None."
    with pytest.raises(TypeError) as error:
        strct.calc_distance([0, 2.0, 3], [1] * 3)
    assert str(error.value) == "`site_index` must be of type int, list, tuple, np.ndarray or None."
    with pytest.raises(TypeError) as error:
        strct.calc_distance([1], "test")
    assert str(error.value) == "`site_index` must be of type int, list, tuple, np.ndarray or None."


@pytest.mark.parametrize(
    "structure, file_suffix",
    [("Benzene", "yaml"), ("ZIF-8", "cif"), ("MOF-303_3xH2O_flawed", "xsf")],
)
def test_calc_distance(structure, file_suffix):
    """Test calc_distance function."""
    ref_outputs = read_yaml_file(REF_PATH + "geometry_" + structure + ".yaml")
    backend = "internal" if structure == "Benzene" else "ase"
    strct = Structure.from_file(STRUCTURES_PATH + structure + "." + file_suffix, backend=backend)
    dist = strct.calc_distance(**ref_outputs["distance"]["function_args"])
    if isinstance(ref_outputs["distance"]["reference"], list):
        assert [
            abs(dist[(idx0, idx1)] - val) < 1e-5
            for (idx0, idx1, val) in zip(
                ref_outputs["distance"]["function_args"]["site_index1"],
                ref_outputs["distance"]["function_args"]["site_index2"],
                ref_outputs["distance"]["reference"],
            )
        ], "Wrong distance."
    else:
        assert abs(dist - ref_outputs["distance"]["reference"]) < 1e-5, "Wrong distance."


def test_calc_distance_pbc():
    """Test correct handling of periodic boundary conditions in the calc_distance function."""
    strct = Structure(
        elements=["H", "H"],
        positions=[[1.0, 1.0, 1.0], [1.0, 9.0, 1.0]],
        cell=[[2.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 2.0]],
        pbc=True,
    )
    assert abs(strct.calc_distance(0, 1) - 2.0) < 1e-5
    strct._attributes = {}
    strct._extras = {}
    strct._function_args = {}
    strct.pbc = [True, False, True]
    assert abs(strct.calc_distance(0, 1) - 8.0) < 1e-5


@pytest.mark.parametrize("structure, file_suffix", [("ZIF-8", "cif"), ("ScBDC", "cif")])
def test_calc_distance_sc(structure, file_suffix):
    """Test calc_distance function using the super cell."""
    ref_outputs = read_yaml_file(REF_PATH + "geometry_" + structure + ".yaml")
    strct = Structure.from_file(
        STRUCTURES_PATH + structure + "." + file_suffix,
        backend="internal",
        backend_kwargs={"strct_check_chem_formula": False},
    )
    dist = strct.calc_distance(**ref_outputs["distance_sc"]["function_args"])
    if ref_outputs["distance_sc"]["reference"] is None:
        assert dist is None
    else:
        if isinstance(dist, tuple):
            dist, pos = dist
            for idx0, pos_list in enumerate(ref_outputs["distance_sc"]["reference_pos"]):
                site_idx = (
                    ref_outputs["distance_sc"]["function_args"]["site_index1"][idx0],
                    ref_outputs["distance_sc"]["function_args"]["site_index2"][idx0],
                )
                for pos_idx, pos_ref in enumerate(pos_list):
                    assert all(
                        [abs(p0 - p1) < 1e-5 for p0, p1 in zip(pos[site_idx][pos_idx], pos_ref)]
                    ), f"Position {site_idx}/{pos_idx} is wrong."
        for idx0, dist_list in enumerate(ref_outputs["distance_sc"]["reference"]):
            if isinstance(dist_list, list):
                site_index1, site_index2 = (
                    ref_outputs["distance_sc"]["function_args"]["site_index1"][idx0],
                    ref_outputs["distance_sc"]["function_args"]["site_index2"][idx0],
                )
                for dist_idx, dist_ref in enumerate(dist_list):
                    assert (
                        abs(dist[(site_index1, site_index2)][dist_idx] - dist_ref) < 1e-5
                    ), f"Distance {(site_index1, site_index2)}/{dist_idx} is wrong."
            else:
                assert abs(dist[idx0] - dist_list) < 1e-5, f"Distance {idx0} is wrong."


@pytest.mark.parametrize("structure, file_suffix", [("Benzene", "yaml"), ("ZIF-8", "cif")])
def test_calc_angle(structure, file_suffix):
    """Test calc_angle function."""
    ref_outputs = read_yaml_file(REF_PATH + "geometry_" + structure + ".yaml")
    backend = "internal" if structure == "Benzene" else "ase"
    strct = Structure.from_file(STRUCTURES_PATH + structure + "." + file_suffix, backend=backend)
    angles = strct.calc_angle(**ref_outputs["angle"]["function_args"])
    if isinstance(ref_outputs["angle"]["reference"], list):
        for angle, ref in zip(angles.values(), ref_outputs["angle"]["reference"]):
            assert abs(angle - ref) < 1e-3, "Wrong angle."
    else:
        assert abs(angles - ref_outputs["angle"]["reference"]) < 1e-3, "Wrong angle."


@pytest.mark.parametrize("structure, file_suffix", [("ScBDC", "cif")])
def test_calc_dihedral_angle(structure, file_suffix):
    """Test calc_dihedral_angle function."""
    ref_outputs = read_yaml_file(REF_PATH + "geometry_" + structure + ".yaml")
    strct = Structure.from_file(
        STRUCTURES_PATH + structure + "." + file_suffix,
        backend="internal",
        backend_kwargs={"strct_check_chem_formula": False},
    )
    angles = strct.calc_dihedral_angle(**ref_outputs["dihedral_angle"]["function_args"])
    assert len(angles) == len(ref_outputs["dihedral_angle"]["reference"])
    for angle, ref in zip(angles.values(), ref_outputs["dihedral_angle"]["reference"]):
        assert abs(angle - ref) < 1e-3, "Wrong angle."


def test_cn_analysis_error():
    """Test method validation of coordination analysis."""
    strct = Structure(**dict(read_yaml_file(STRUCTURES_PATH + "GaAs_216_conv.yaml")))
    with pytest.raises(ValueError) as error:
        strct.calc_coordination(method="test")
    assert (
        str(error.value)
        == "Method 'test' is not supported. Supported methods are: 'minimum_distance', "
        "'n_nearest_neighbours', 'atomic_radius', 'econ', 'voronoi'."
    )
    with pytest.raises(ValueError) as error:
        strct.calc_coordination(method="voronoi", voronoi_weight_type="test")
    assert str(error.value) == "`weight_type` 'test' is not supported."


@pytest.mark.parametrize(
    "structure_label, method",
    [
        ("GaAs_216_conv", "minimum_distance"),
        ("GaAs_216_conv", "n_nearest_neighbours"),
        ("GaAs_216_conv", "atomic_radius"),
        ("GaAs_216_conv", "econ"),
        ("GaAs_216_conv", "okeeffe"),
        ("Cs2Te_62_prim", "minimum_distance"),
        ("Cs2Te_62_prim", "n_nearest_neighbours"),
        ("Cs2Te_62_prim", "atomic_radius"),
        ("Cs2Te_62_prim", "econ"),
        ("Cs2Te_62_prim", "okeeffe"),
        ("Cs2Te_62_prim", "voronoi_no_weights"),
        ("Cs2Te_62_prim", "voronoi_radius"),
        ("Cs2Te_62_prim", "voronoi_area"),
    ],
)
def test_cn_analysis(nested_dict_comparison, structure_label, method):
    """
    Test the different methods to determine the coordination number of atomic sites.
    """
    inputs = dict(read_yaml_file(STRUCTURES_PATH + structure_label + ".yaml"))
    inputs.pop("kinds", None)
    ref = dict(
        read_yaml_file(REF_PATH + "coordination_" + structure_label + "_" + method + ".yaml")
    )
    structure = Structure(**inputs)
    output = structure.calc_coordination(**ref["parameters"], indices=list(range(len(structure))))
    sites = output.pop("sites")
    sites_ref = ref["ref"].pop("sites")
    assert len(sites) == len(sites_ref)
    nested_dict_comparison(output, ref["ref"])
    for site_idx, (site, site_ref) in enumerate(zip(sites, sites_ref)):
        neighs = site.pop("neighbours")
        neighs_ref = site_ref.pop("neighbours")
        assert len(neighs) == len(neighs_ref)
        nested_dict_comparison(site, site_ref)
        used_indices = []
        for neigh in neighs:
            for idx, neigh_ref in enumerate(neighs_ref):
                match = True
                for key in ["element", "kind", "site_index"]:
                    if neigh[key] != neigh_ref[key]:
                        match = False
                if "weight" in neigh_ref and abs(neigh["weight"] - neigh_ref["weight"]) > 1e-5:
                    match = False
                if abs(neigh["distance"] - neigh_ref["distance"]) > 1e-5:
                    match = False
                if any(
                    abs(v0 - v1) > 1e-5 for v0, v1 in zip(neigh["position"], neigh_ref["position"])
                ):
                    match = False
                if match and idx not in used_indices:
                    used_indices.append(idx)
                    break
            assert match, f"Neighbour {neigh} of site {site_idx} not found in reference data."


@pytest.mark.parametrize(
    "structure, kwargs",
    [
        ("ZIF-8.cif", {"symprec": 1e-5}),
        ("GaAs_216_prim.yaml", {}),
        ("Cs2Te_62_prim.yaml", {"symprec": 0.01, "return_sym_operations": True}),
    ],
)
def test_calc_space_group(nested_dict_comparison, structure, kwargs):
    """Test calc_space_group function."""
    ref_outputs = dict(read_yaml_file(REF_PATH + "symmetry_" + structure.split(".")[0] + ".yaml"))
    strct = Structure.from_file(STRUCTURES_PATH + structure, backend="internal")
    sg_dict = strct.calc_space_group(**kwargs)
    nested_dict_comparison(sg_dict, ref_outputs)
    assert strct["attributes"]["space_group"] == sg_dict["space_group"]["number"]


@pytest.mark.parametrize(
    "structure",
    [
        "CHBrClF",
        "HClO",
        "C4H6O6",
        "HCN",
        "C2H2",
        "N2H4",
        "H3PO4",
        "C2H2Cl2",
        "H2O",
        "NH3",
        "C2H4",
        "PCl5",
        "C5H5",
        "Benzene",
        "C7H7",
        "C8H8",
        "C3H4",
        "CH4",
        "SF6",
    ],
)
def test_point_group_determination(nested_dict_comparison, structure):
    """
    Test the method to determine the point group.
    """
    ref = dict(read_yaml_file(REF_PATH + "symmetry_" + structure + ".yaml"))
    if structure in ["H2O", "NH3"]:
        structure = Structure.from_str(structure)
    else:
        structure = Structure.from_file(STRUCTURES_PATH + structure + ".yaml", backend="internal")
    nested_dict_comparison(structure.calc_point_group(), ref)
    assert structure["attributes"]["point_group"] == ref["point_group"]


@pytest.mark.parametrize(
    "bulk_crystal, miller_indices, ter",
    [
        ("GaAs_216_prim", (1, 0, 0), 1),
        # ("GaAs_216_prim", (1, 1, 0), 1),
        # ("Cs2Te_62_prim", (1, 0, 0), 1),
        # ("Cs2Te_62_prim", (1, 1, 1), 1),
        # ("Cs2Te_194_prim", (1, 0, 1), 1),
        # ("Cs2Te_194_prim", (5, 1, 2), 1),
    ],
)
def test_2d_brillouin_zone_kpath(
    structure_comparison, nested_dict_comparison, bulk_crystal, miller_indices, ter
):
    """Test k-path generation in 2d Brillouin zone."""
    mil_ind_str = "".join(str(mil_idx) for mil_idx in miller_indices)
    ref_outputs = read_yaml_file(
        REF_PATH + f"brillouin_zone_2d_{bulk_crystal}_{mil_ind_str}_{ter}.yaml"
    )
    bulk_structure = Structure(**read_yaml_file(STRUCTURES_PATH + bulk_crystal + ".yaml"))
    bulk_structure.attributes["label"] = bulk_crystal
    surf_gen = SurfaceGeneration(bulk_structure)

    surf_c = surf_gen.generate_surface_slabs(miller_indices=miller_indices)
    prim_slab, layer_group = _transform_slab_to_primitive(surf_c[ter - 1], 0.005, -1, 0)
    kpath = _get_kpath(
        prim_slab["cell"],
        2,
        layer_group,
        0.015,
        1.0e-5,
    )
    structure_comparison(
        prim_slab,
        ref_outputs["prim_slab"],
        tolerance=1e-3,
    )
    assert layer_group == ref_outputs["layer_group"], "Wrong layer group."
    nested_dict_comparison(kpath, ref_outputs["kpath"])
