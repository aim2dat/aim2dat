"""k-path in 2d Brillouin zones."""

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.strct.analysis.geometry import _calc_reciprocal_cell
from aim2dat.utils.maths import calc_angle


P_OBL_LGS = [1, 2, 3, 4, 5, 6, 7]
C_RECT_LGS = [10, 13, 18, 22, 26, 35, 36, 47, 48]
P_RECT_LGS = (
    [8, 9, 11, 12]
    + list(range(14, 18))
    + [19, 20, 21, 23, 24, 25]
    + list(range(27, 35))
    + list(range(37, 47))
)  # 34 Centered?
P_SQU_LGS = list(range(49, 65))
P_TRIG_LGS = [65, 68, 70, 74, 79]
P_HEX_LGS = [66, 67, 69, 71, 72, 73, 75, 76, 77, 78, 80]


def _get_kpath(cell, aperiodic_dir, layer_group, reference_distance, threshold):
    # TODO make general for different aperiodic directions..
    path_parameters = {"explicit_segments": [], "point_coords": {"GAMMA": [0.0, 0.0, 0.0]}}
    periodic_dirs = [idx0 for idx0 in range(3) if idx0 != aperiodic_dir]
    cell_lengths = [np.linalg.norm(cell[idx0]) for idx0 in periodic_dirs]
    angle = calc_angle(cell[periodic_dirs[0]], cell[periodic_dirs[1]])
    path_parameters["reciprocal_primitive_lattice"] = _calc_reciprocal_cell(cell)
    rec_cell = np.array(path_parameters["reciprocal_primitive_lattice"])
    if layer_group in P_OBL_LGS:
        path_parameters["point_coords"].update(
            {
                "A": [0.5, -0.5, 0.0],
                "B": [0.5, 0.0, 0.0],
                "Y": [0.0, 0.5, 0.0],
            }
        )
        path_parameters["path"] = [["A", "GAMMA"], ["GAMMA", "B"], ["GAMMA", "Y"]]
    elif layer_group in C_RECT_LGS:
        path_parameters["point_coords"]["S"] = [0.0, 0.5, 0.0]
        if angle == np.pi / 3.0 or angle == np.pi / 2.0 or angle == np.pi * 2.0 / 3.0:
            raise ValueError("Symmetry is c2 but the angle between cell vectors is wrong.")
        elif angle < np.pi / 2.0:  # case with a > b
            path_parameters["point_coords"].update(
                {
                    "Y": [0.5, 0.5, 0.0],
                    "F_0": _c_rectangular_calc_f0_point(rec_cell),
                    "DELTA_0": _c_rectangular_calc_delta0_point(rec_cell),
                }
            )
            path_parameters["path"] = [
                ["GAMMA", "Y"],
                ["Y", "F_0"],
                ["F_0", "S"],
                ["S", "DELTA_0"],
                ["DELTA_0", "GAMMA"],
                ["GAMMA", "S"],
            ]
        else:  # case with a < b
            path_parameters["point_coords"].update(
                {
                    "Y": [-0.5, 0.5, 0.0],
                    "C_0": _c_rectangular_calc_c0_point(rec_cell),
                    "SIGMA_0": _c_rectangular_calc_simga0_point(rec_cell),
                }
            )
            path_parameters["path"] = [
                ["GAMMA", "Y"],
                ["Y", "C_0"],
                ["C_0", "S"],
                ["S", "SIGMA_0"],
                ["SIGMA_0", "GAMMA"],
                ["GAMMA", "S"],
            ]
    elif layer_group in P_RECT_LGS:
        if abs(angle - np.pi / 2.0) > threshold:
            raise ValueError(
                "Symmetry is p2 but the angle between cell vectors is not 90 degrees."
            )
        path_parameters["point_coords"].update(
            {
                "S": [0.5, 0.5, 0.0],
                "X": [0.5, 0.0, 0.0] if cell_lengths[1] > cell_lengths[0] else [0.0, 0.5, 0.0],
                "Y": [0.0, 0.5, 0.0] if cell_lengths[1] > cell_lengths[0] else [0.5, 0.0, 0.0],
            }
        )
        path_parameters["path"] = [
            ["GAMMA", "X"],
            ["X", "S"],
            ["S", "Y"],
            ["Y", "GAMMA"],
            ["GAMMA", "S"],
        ]
    elif layer_group in P_SQU_LGS:
        if abs(cell_lengths[0] - cell_lengths[1]) > threshold:
            raise ValueError("Symmetry is p4 but cell vectors have different lengths.")
        if abs(angle - np.pi / 2.0) > threshold:
            raise ValueError(
                "Symmetry is p4 but the angle between cell vectors is not 90 degrees."
            )
        path_parameters["point_coords"].update({"M": [0.5, 0.5, 0.0], "X": [0.5, 0.0, 0.0]})
        path_parameters["path"] = [["GAMMA", "X"], ["X", "M"], ["M", "GAMMA"]]
    elif layer_group in P_TRIG_LGS + P_HEX_LGS:
        if abs(cell_lengths[0] - cell_lengths[1]) > threshold:
            raise ValueError("Symmetry is p6 but cell vectors have different lengths.")
        if abs(angle - 2.0 * np.pi / 3.0) > threshold:
            raise ValueError(
                "Symmetry is p6 but the angle between cell vectors is not 120 degrees."
            )
        path_parameters["point_coords"].update(
            {"K": [1.0 / 3.0, 1.0 / 3.0, 0.0], "M": [0.5, 0.0, 0.0]}
        )
        if layer_group in P_TRIG_LGS:
            path_parameters["point_coords"]["KA"] = [2.0 / 3.0, -1.0 / 3.0, 0.0]
            path_parameters["path"] = [
                ["GAMMA", "K"],
                ["K", "KA"],
                ["KA", "GAMMA"],
                ["GAMMA", "M"],
            ]
        else:
            path_parameters["path"] = [["GAMMA", "K"], ["K", "M"], ["M", "GAMMA"]]
    total_points = 0
    for p_points in path_parameters["path"]:
        kpoints = [
            np.dot(rec_cell.T, path_parameters["point_coords"][label]).T for label in p_points
        ]
        seg_points = int(round(np.linalg.norm(kpoints[1] - kpoints[0]) / reference_distance))
        path_parameters["explicit_segments"].append([total_points, total_points + seg_points])
        total_points += seg_points
    return path_parameters


def _c_rectangular_calc_f0_point(rec_cell):
    idx0 = 0
    idx1 = 1
    if rec_cell[idx1][0] == 0.0:
        u = 0.5 * rec_cell[idx0][1] / (rec_cell[idx0][1] - rec_cell[idx1][1])
    else:
        a = -0.5 * (rec_cell[idx0][0] + rec_cell[idx0][1] * rec_cell[idx1][1] / rec_cell[idx1][0])
        b = (
            rec_cell[idx1][0]
            - rec_cell[idx0][0]
            + rec_cell[idx1][1] * (rec_cell[idx1][1] - rec_cell[idx0][1]) / rec_cell[idx1][0]
        )
        u = a / b
    return [float(0.5 - u), float(0.5 + u), 0.0]


def _c_rectangular_calc_delta0_point(rec_cell):
    idx0 = 0
    idx1 = 1
    if rec_cell[idx1][1] == 0.0:
        u = 0.5 * rec_cell[idx1][0] / (rec_cell[idx1][0] - rec_cell[idx0][0])
    else:
        a = 0.5 * (rec_cell[idx1][1] + (rec_cell[idx1][0] ** 2.0) / rec_cell[idx1][1])
        b = (
            rec_cell[idx1][1]
            - rec_cell[idx0][1]
            - rec_cell[idx1][0] * (rec_cell[idx1][0] - rec_cell[idx0][0]) / rec_cell[idx1][1]
        )
        u = a / b
    return [float(-u), float(u), 0.0]


def _c_rectangular_calc_c0_point(rec_cell):
    if rec_cell[1][0] == 0.0:
        u = 0.5 * rec_cell[0][1] / (rec_cell[0][1] + rec_cell[1][1])
    else:
        v = rec_cell[0][1] + rec_cell[1][1] * rec_cell[0][0] / rec_cell[1][0]
        u = v / (v + 4.0 * rec_cell[1][1])
    return [float(u - 0.5), float(u + 0.5), 0.0]


def _c_rectangular_calc_simga0_point(rec_cell):
    if rec_cell[1][1] == 0.0:
        u = 0.5 * rec_cell[1][0] / (rec_cell[0][0] + rec_cell[1][0])
    else:
        a = 0.5 * (rec_cell[1][0] ** 2.0 / rec_cell[1][1] + rec_cell[1][1])
        b = (
            rec_cell[0][1]
            + rec_cell[1][1]
            + (rec_cell[0][0] + rec_cell[1][0]) * rec_cell[1][0] / rec_cell[1][1]
        )
        u = a / b
    return [float(u), float(u), 0.0]
