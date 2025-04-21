"""Functions to read plot-files of xmgrace."""

# Standard library imports
from typing import Union

# Third party library imports
import numpy as np
import re

# Internal library imports
from aim2dat.io.utils import read_band_structure


def read_xmgrace_file(file_path: str) -> Union[list, tuple]:
    """
    Read xmgrace plot filies.

    Up to now the functionality is very limited. xy-data is read and grouped by the line-color.

    Notes
    -----
        TODO: Increase functionality and generality.

    Parameters
    ----------
    file_path : str
        Path to the xmgrace file.

    Returns
    -------
    x_values : list
        Nested list of the x-values.
    y_values : list
        Nested list of the y-values.
    tick_labels : tuple
        Tuple of dictionaries containing the position on the x-axis ("pos") and the label
        ("label").
    """

    def parse_header(fobj):
        """Parse header information of the xmgrace file."""
        plot_categories = {}
        tick_labels = {}

        pattern_tick_major = re.compile(
            r"^@\s*xaxis\s*tick\s*major\s*(?P<tick_nr>\d+),"
            r"\s*(?P<tick_pos>([-+]?[0-9]*\.?[0-9]*))$"
        )
        pattern_tick_label = re.compile(
            r'^@\s*xaxis\s*ticklabel\s*(?P<tick_nr>\d+),\s*"(?P<tick_label>.*)"$'
        )
        pattern_line_color = re.compile(
            r"^@\s*s(?P<plot_nr>\d+)\s*line\s*color\s*(?P<color>\d+)?$"
        )
        str_stop = "@target"

        with open(file_path, "r") as fobj:
            for line_idx, line in enumerate(fobj):
                if pattern_line_color.match(line):
                    match = pattern_line_color.match(line)
                    group_dict = match.groupdict()
                    plot_categories[int(group_dict["plot_nr"])] = int(group_dict["color"])
                elif pattern_tick_label.match(line):
                    match = pattern_tick_label.match(line)
                    group_dict = match.groupdict()
                    tick_nr = int(group_dict["tick_nr"])
                    if tick_nr not in tick_labels:
                        tick_labels[tick_nr] = {"label": group_dict["tick_label"]}
                    else:
                        tick_labels[tick_nr]["label"] = group_dict["tick_label"]
                elif pattern_tick_major.match(line):
                    match = pattern_tick_major.match(line)
                    group_dict = match.groupdict()
                    tick_nr = int(group_dict["tick_nr"])
                    if tick_nr not in tick_labels:
                        tick_labels[tick_nr] = {"pos": float(group_dict["tick_pos"])}
                    else:
                        tick_labels[tick_nr]["pos"] = float(group_dict["tick_pos"])
                if str_stop in line:
                    break
        nr_of_categories = len(set(plot_categories.values()))

        # Sort tick labels:
        tick_labels = list(tick_labels.values())
        zipped = list(zip([tick_label["pos"] for tick_label in tick_labels], tick_labels))
        zipped.sort(key=lambda point: point[0])
        _, tick_labels = zip(*zipped)

        return nr_of_categories, plot_categories, line_idx, tick_labels

    str_start_set = "@type xy"
    start_pattern = re.compile(r"^@target\s*G\d+.S(\d+)?$")
    str_end_set = "&"
    in_set = False
    nr_of_categories, plot_categories, start_idx, tick_labels = parse_header(file_path)

    with open(file_path, "r") as fobj:
        x_values = [[] for idx in range(nr_of_categories)]
        y_values = [[] for idx in range(nr_of_categories)]

        lines = fobj.readlines()[start_idx:]
        for line in lines:
            if not in_set:
                match = start_pattern.match(line)
                if match:
                    category_idx = plot_categories[int(match.group(1))] - nr_of_categories + 1
                    x_values[category_idx].append([])
                    y_values[category_idx].append([])
                elif str_start_set in line:
                    in_set = True
            elif str_end_set in line:
                in_set = False
            elif in_set:
                x_values[category_idx][-1].append(float(line.split()[0]))
                y_values[category_idx][-1].append(float(line.split()[1]))
    return x_values, y_values, tick_labels


@read_band_structure(r".*\.agr$")
def read_xmgrace_band_structure(file_path: str, kpoints: list) -> list:
    """
    Read xmgrace band structure file.

    Parameters
    ----------
    file_path : str
        Path to the xmgrace file.
    kpoints : list
        List of tuples containing the label and the k-point.

    Returns
    -------
    band_structures : list
        List of band structures.
    """
    band_structures = []
    x_values, y_values, tick_labels = read_xmgrace_file(file_path)
    x_values = x_values[0][0]
    path = []

    # Construct segments and reverse-engineer k-path:
    labels = []
    x_values_indices = [0, 0]
    for segment_idx in range(len(tick_labels) - 1):
        # Determine number of points:
        for idx, value in enumerate(x_values[x_values_indices[0] :]):
            if abs(value - tick_labels[segment_idx + 1]["pos"]) < 1e-4:
                x_values_indices[1] = x_values_indices[0] + idx
                break

        # Check labels and find k-points:
        segment_labels = (tick_labels[segment_idx]["label"], tick_labels[segment_idx + 1]["label"])
        segment_kpoints = []
        for x_values_idx, segment_label in zip(x_values_indices, segment_labels):
            if segment_label == "\\xG\\f{}":
                segment_kpoints.append(kpoints["Gamma"])
                labels.append((x_values_idx, "Gamma"))
            else:
                segment_kpoints.append(kpoints[segment_label])
                labels.append((x_values_idx, segment_label))

        # Calculate directional vector
        vector = np.subtract(segment_kpoints[1], segment_kpoints[0])

        # Create k-path:
        length = x_values_indices[1] - x_values_indices[0]
        for point_nr in range(length + 1):
            path.append(np.add(segment_kpoints[0], point_nr * vector / (length)).tolist())
        x_values_indices[0] = x_values_indices[1] + 1

    for bands_cat in y_values:
        band_structures.append(
            {
                "kpoints": path,
                "bands": np.asarray(bands_cat).transpose().tolist(),
                "path_labels": labels,
            }
        )
    return band_structures
