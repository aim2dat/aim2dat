"""Functions to determine the different types of hulls for function values."""

# Third party library imports
from scipy.spatial import ConvexHull


def get_convex_hull(points, lower_hull=True, upper_hull=True, tolerance=0.001):
    """
    Get convex hull from a list of n-dimensional points using scipy.

    So far this method only works for 2-dimensional data.

    Parameters
    ----------
    points : list
        List of points with shape (npoints, ndim).
    lower_hull : bool
        Whether to include the lower part of the hull (values below zero).
    upper_hull : bool
        Whether to include the upper part of the hull (values above zero).
    tolerance : float
        Tolerance parameter for lower and upper hull.
    """
    if not lower_hull and not upper_hull:
        raise ValueError("Either `lower_hull` or `upper_hull` need to be set to `True`.")
    convex_hull = [[], []]
    hull = ConvexHull(points)
    convex_hull_dict = {}
    for vertix in hull.vertices:
        if lower_hull and upper_hull:
            convex_hull[0].append(points[vertix][0])
            convex_hull[1].append(points[vertix][1])
        elif upper_hull and points[vertix][1] > -1.0 * tolerance:
            if points[vertix][0] in convex_hull_dict:
                convex_hull_dict[points[vertix][0]] = max(
                    points[vertix][1], convex_hull_dict[points[vertix][0]]
                )
            else:
                convex_hull_dict[points[vertix][0]] = points[vertix][1]
        elif lower_hull and points[vertix][1] < tolerance:
            if points[vertix][0] in convex_hull_dict:
                convex_hull_dict[points[vertix][0]] = min(
                    points[vertix][1], convex_hull_dict[points[vertix][0]]
                )
            else:
                convex_hull_dict[points[vertix][0]] = points[vertix][1]

    if len(convex_hull_dict) > 0:
        convex_hull = [list(convex_hull_dict.keys()), list(convex_hull_dict.values())]
    zipped = list(zip(convex_hull[0], convex_hull[1]))
    zipped.sort(key=lambda point: point[0])
    if len(zipped) > 0:
        convex_hull[0], convex_hull[1] = zip(*zipped)
    return convex_hull


def get_minimum_maximum_points(points):
    """
    Get minimum and maximum values for each x-value.

    Parameters
    ----------
    points : list
        List of points with shape (npoints, ndim).

    Returns
    -------
    x_values : list
        x-values.
    min_values : list
        Minimum y-value for each point.
    max_values : list
        Maximum y-value for each point.
    """
    min_points = {}
    max_points = {}
    x_values = []
    for pt in points:
        x_val = pt[:-1]
        y_val = pt[-1]
        if len(pt) == 2:
            x_val = pt[0]
        if x_val not in x_values:
            x_values.append(x_val)
        for method, dict0 in zip((min, max), (min_points, max_points)):
            if x_val in dict0:
                dict0[x_val] = method(y_val, dict0[x_val])
            else:
                dict0[x_val] = y_val
    min_values = []
    max_values = []
    x_values.sort()
    for x_val in x_values:
        min_values.append(min_points[x_val])
        max_values.append(max_points[x_val])
    return x_values, min_values, max_values
