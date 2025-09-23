"""Module that contains custom mathematical functions."""

# Standard library imports
import math
import numpy as np


def calc_angle(vector1, vector2):
    """
    Calculate the angle between two vectors.

    Parameters
    ----------
    vector1 : list
        List containing three numbers.
    vector2 : list
        List containing three numbers.

    Returns
    -------
    angle : float
        Angle in radian.
    """
    return math.acos(
        np.clip(
            np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)),
            -1.0,
            1.0,
        )
    )


def calc_plane_equation(point1, point2, point3):
    """
    Calculate the plane from 3 given points in the form ``a*x + b*y + c*z = d``.

    Parameters
    ----------
    point1 : list or np.array
        Point in 3-dimensional space.
    point2 : list or np.array
        Point in 3-dimensional space.
    point3 : list or np.array
        Point in 3-dimensional space.

    Returns
    -------
    a : float
        plane parameter.
    b : float
        plane parameter.
    c : float
        plane parameter.
    d : float
        plane parameter.
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    vector1 = point2 - point1
    vector2 = point3 - point1

    plane_normal = np.cross(vector1, vector2)
    a = plane_normal[0]
    b = plane_normal[1]
    c = plane_normal[2]
    d = -1.0 * sum(plane_normal * point1)

    return a, b, c, d


def calc_solid_angle(center, points):
    """
    Calculate the solid angle between a center point and points that span a polyhedron.

    Parameters
    ----------
    center : list
        Center point.
    points : list
        Nested list of points.

    Returns
    -------
    solid_angle : float
        Solid angle of the polyhedron.
    """
    solid_angle = 0.0

    # Transform to numpy and center points:
    points_np = [np.subtract(np.array(point), np.array(center)) for point in points]
    norms = [np.linalg.norm(point) for point in points_np]

    # Split area into Tetrahedrons and calculate based on the method of Van Oosterom and Strackee:
    for tr_idx in range(len(points_np) - 2):
        # Point indices:
        idx1 = tr_idx + 1
        idx2 = tr_idx + 2
        numerator = np.abs(
            np.linalg.det(np.array([points_np[0], points_np[idx1], points_np[idx2]]))
        )
        denomintor = (
            norms[0] * norms[idx1] * norms[idx2]
            + np.dot(points_np[0], points_np[idx1]) * norms[idx2]
            + np.dot(points_np[0], points_np[idx2]) * norms[idx1]
            + np.dot(points_np[idx1], points_np[idx2]) * norms[0]
        )
        if denomintor == 0.0:
            solid_angle += math.pi
        else:
            angle = np.arctan(numerator / denomintor)
            solid_angle += 2.0 * (angle if angle > 0.0 else angle + math.pi)
    return solid_angle


def calc_polygon_area(vertices):
    """
    Calculate the area of a polygon.

    Parameters
    ----------
    vertices : list or np.array
        (n x 3) list of the vertices.

    Returns
    -------
    area : float
        Area of the polygon.
    """
    vertices = np.array([np.array(vertix) for vertix in vertices])

    # Reference to first point:
    vertices_ref = vertices[1:] - vertices[0:1]

    # calculate cross-products:
    cross_products = 0.5 * np.cross(vertices_ref[:-1], vertices_ref[1:], axis=1)

    # Add cross-products:
    added = np.zeros(3)
    for cross_product in cross_products:
        added = np.add(added, cross_product)

    # Calculate norm
    area = np.linalg.norm(added)
    return area


def calc_circular_segment_area(radius, distance):
    """
    Calculate the circular segment.

    See: https://en.wikipedia.org/wiki/Circular_segment.

    Parameters
    ----------
    radius : float
        Radius of the circle.
    distance : float
        Distance of the segment from the circle center.

    Returns
    -------
    segment_area : float
        Area of the segment.
    """
    if distance > radius:
        raise ValueError("The height of the segment cannot be larger than the radius.")

    # Calculate angle:
    theta = 2.0 * np.arccos(distance / radius)
    return 0.5 * radius**2.0 * (theta - math.sin(theta))


def calc_reflection_matrix(n_vector):
    """
    Calculate the 3d reflection matrix normal to the input vector.

    Parameters
    ------------
    n_vector : list or np.array
        Normal vector of the reflection plane.

    Returns
    -------
    : np.array
        Reflection matrix.
    """
    n_vector = np.array(n_vector)
    n_vector /= np.linalg.norm(n_vector)
    sigma = np.zeros((3, 3))
    for dir0 in range(3):
        sigma[dir0, dir0] = 1.0 - 2.0 * n_vector[dir0] ** 2.0
        sigma[dir0, dir0 - 2] = -2.0 * n_vector[dir0] * n_vector[dir0 - 2]
        sigma[dir0 - 2, dir0] = -2.0 * n_vector[dir0] * n_vector[dir0 - 2]
    return sigma


def create_lin_ind_vector(vector):
    """
    Create linearly independent vector with reference to the input vector.
    The method adds one to the first element of the vector which is zero.
    In case all elements are non-zero, one is added to the first element of
    the vector.

    Parameters
    ----------
    vector : list, tuple or np.array
        Input vector

    Returns
    -------
    : np.array
        Linearly independent vector.
    """
    vector = np.array(vector)
    is_ind = False
    for i, v in enumerate(vector):
        if math.isclose(v, 0.0, rel_tol=0.0, abs_tol=1e-05):
            vector[i] += 1.0
            is_ind = True
            break
    if not is_ind:
        vector[0] += 1
    return vector


def gaussian_function(x, mu, sigma):
    """
    Calculate the Gaussian function with a certain sigma-value.

    Parameters
    ----------
    x : float
        x value for which the y value is calculated.
    mu : float
        Expected value.
    sigma : float
        Variance.

    Returns
    -------
    y : float
        y value.
    """
    return 1.0 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-np.power((x - mu) / sigma, 2.0) / 2.0)
