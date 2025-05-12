"""Test calculate-functions via the StructureOperations class."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports:
from aim2dat.strct import StructureCollection, StructureOperations
from aim2dat.strct.ext_analysis import (
    calc_prdf,
    calc_ffingerprint_order_p,
    calc_warren_cowley_order_p,
    calc_planes,
)
from aim2dat.io import read_yaml_file


STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
PRDF_PATH = os.path.dirname(__file__) + "/prdf_functions/"
WC_LIKE_ORDER_PATH = os.path.dirname(__file__) + "/warren_cowley_like_order_p/"
PLANES_PATH = os.path.dirname(__file__) + "/planes/"
STABILITIES_PATH = os.path.dirname(__file__) + "/stabilities/"


@pytest.mark.parametrize(
    "structure,ref_order_p",
    [
        ("GaAs_216_prim", (742.1268, (495.7578, 495.7578))),
        (
            "Cs2Te_62_prim",
            (
                115.6221,
                (
                    155.4298,
                    113.9550,
                    114.2768,
                    153.4972,
                    153.4300,
                    113.8181,
                    113.7066,
                    153.9680,
                    62.4293,
                    62.2036,
                    62.2604,
                    62.6054,
                ),
            ),
        ),
    ],
)
def test_ffingerprint_order_parameters(structure, ref_order_p):
    """Test order F-fingerprint order parameters."""
    strct_collect = StructureCollection()
    inputs = dict(read_yaml_file(STRUCTURES_PATH + structure + ".yaml"))
    # inputs["structure_label"] = structure
    strct_collect.append(structure, **inputs)
    strct_ops = StructureOperations(strct_collect)
    output = strct_ops[structure].perform_analysis(
        method=calc_ffingerprint_order_p,
        kwargs={"r_max": 20.0, "delta_bin": 0.005, "sigma": 10.0},
    )
    assert (
        abs(output["order_p"] - ref_order_p[0]) < 1e-3
    ), f"Total order parameter of structure {structure} is wrong."
    assert len(output["site_order_p"]) == len(
        ref_order_p[1]
    ), f"Wrong number of site order parameters for structure {structure}."
    for idx, (site_order_p0, ref_site_order_p0) in enumerate(
        zip(output["site_order_p"], ref_order_p[1])
    ):
        assert (
            abs(site_order_p0 - ref_site_order_p0) < 1e-3
        ), f"Order parameter of site {idx} of structure {structure} is wrong."


@pytest.mark.parametrize("structure", ["Cs2Te_62_prim", "GaAs_216_conv", "GaAs_216_prim"])
def test_prdf_functions(structure, nested_dict_comparison):
    """Test partial radial distribution calculation."""
    ref = read_yaml_file(PRDF_PATH + structure + "_ref.yaml")
    strct_c = StructureCollection()
    inputs = dict(read_yaml_file(STRUCTURES_PATH + structure + ".yaml"))
    strct_c.append(structure, **inputs)
    strct_ops = StructureOperations(strct_c)
    element_prdf, atomic_prdf = strct_ops[0].perform_analysis(
        method=calc_prdf, kwargs=ref["parameters"]
    )

    assert len(element_prdf) == len(ref["element_prdf"]), "Wrong number of el-pairs."
    for el_pair, prdf in element_prdf.items():
        assert el_pair in ref["element_prdf"], f"Element pair {el_pair} not in reference output."
        assert all(
            [
                abs(val0 - ref["element_prdf"][el_pair][idx]) < 1.0e-5
                for idx, val0 in enumerate(prdf)
            ]
        ), f"Element prdf is wrong for element pair {el_pair}."
    assert len(atomic_prdf) == len(ref["atomic_prdf"]), "Wrong number of sites."
    for site_idx, site in enumerate(atomic_prdf):
        assert len(site) == len(
            ref["atomic_prdf"][site_idx]
        ), f"Wrong number of element pairs for site {site_idx}."
        for el_pair, prdf in site.items():
            assert (
                el_pair in ref["atomic_prdf"][site_idx]
            ), f"Element pair {el_pair} not in atomic prdf."
            assert all(
                [
                    abs(val0 - ref["atomic_prdf"][site_idx][el_pair][idx]) < 1.0e-5
                    for idx, val0 in enumerate(prdf)
                ]
            ), f"Atomic prdf is wrong for element pair {el_pair} at site {site_idx}."


@pytest.mark.parametrize(
    "structure, r_max, max_shells",
    [
        ("Al_225_conv", 5.0, 3),
        ("NaCl_225_prim", 5.0, 3),
        ("Cs2Te_19_prim", 5.0, 2),
        ("Cs2Te_62_prim", 5.0, 2),
        ("GaAs_216_prim", 5.0, 3),
        ("GaAs_216_conv", 5.0, 3),
    ],
)
def test_warren_cowley_like_order_parameters(nested_dict_comparison, structure, r_max, max_shells):
    """Test calculation of warren cowley order parameters."""
    inputs = dict(read_yaml_file(STRUCTURES_PATH + structure + ".yaml"))
    ref = dict(read_yaml_file(WC_LIKE_ORDER_PATH + structure + "_ref.yaml"))
    strct_c = StructureCollection()
    strct_c.append(structure, **inputs)
    strct_ops = StructureOperations(strct_c)
    output = strct_ops[0].perform_analysis(
        method=calc_warren_cowley_order_p,
        kwargs={"r_max": r_max, "max_shells": max_shells},
    )
    nested_dict_comparison(output, ref)


@pytest.mark.parametrize(
    "structure,file_type", [("Imidazole", "xyz"), ("ZIF-8_complex", "xyz"), ("ZIF-8", "cif")]
)
def test_planes(structure, file_type):
    """
    Test finding planes.
    """
    ref = read_yaml_file(PLANES_PATH + structure + "_ref.yaml")
    strct_collect = StructureCollection()
    strct_collect.append_from_file("structure", STRUCTURES_PATH + structure + "." + file_type)
    strct_ops = StructureOperations(strct_collect)
    planes = strct_ops["structure"].perform_analysis(method=calc_planes, kwargs=ref["parameters"])

    for plane, ref_plane in zip(planes, ref["reference"]):
        assert len(plane["site_indices"]) == len(
            ref_plane["site_indices"]
        ), "Number of site indices do not match."
        for site_idx in ref_plane["site_indices"]:
            assert site_idx in plane["site_indices"], f"Site index {site_idx} not found."
        for vect, ref_vect in zip(plane["plane"], ref_plane["plane"]):
            for coord, ref_coord in zip(vect, ref_vect):
                assert abs(coord - ref_coord) < 1e-4, "Vectors do not match."
        for proj_pos, ref_proj_pos in zip(plane["proj_positions"], ref_plane["proj_positions"]):
            assert (
                proj_pos["label"] == ref_proj_pos["label"]
            ), "Wrong label for projected position."
            for coord in "xy":
                assert (
                    abs(proj_pos[coord] - ref_proj_pos[coord]) < 1e-4
                ), "Coordinates of projected positions is wrong."


def test_stabilities(create_structure_collection_object):
    """Test calc_stabilities function."""
    ref = read_yaml_file(STABILITIES_PATH + "MOFs_ref.yaml")
    strct_c, _ = create_structure_collection_object(["GaAs_216_conv"])
    for idx0, strct in enumerate(ref["input"]):
        strct_c.append("test_" + str(idx0), **strct)
    strct_ops = StructureOperations(strct_c)
    f_e, st = strct_ops.calc_stabilities(unit="eV")
    f_e = f_e[1:]
    st = st[1:]
    assert all(
        abs(val - ref_val) < 1.0e-4 for val, ref_val in zip(f_e, ref["ref"]["formation_energies"])
    ), "Formation energies are wrong."
    assert all(
        abs(val - ref_val) < 1.0e-4 for val, ref_val in zip(st, ref["ref"]["stabilities"])
    ), "Stabilities are wrong."
