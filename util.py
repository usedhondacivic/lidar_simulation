import numpy as np


# chatgpt <3
def ray_line_seg_intersection(ray_origin, ray_direction, line_point1, line_point2):
    """
    Finds the intersection point of a ray and a line segment.

    Args:
        ray_origin: The origin of the ray (numpy array)
        ray_direction: The direction vector of the ray (numpy array)
        line_point1: One point on the line segment (numpy array)
        line_point2: Another point on the line segment (numpy array)

    Returns:
        Intersection point if exists, otherwise None
    """

    # Calculate the direction vector of the line
    line_direction = line_point2 - line_point1

    # Check if the ray and line are parallel
    if np.cross(ray_direction, line_direction).any() == 0:
        return None  # Parallel

    # Solve the system of equations for the intersection point
    A = np.array([ray_direction, -line_direction]).T
    b = line_point1 - ray_origin

    try:
        t, u = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None  # No intersection

    # Check if the intersection point lies on the line segment
    if 0 <= u <= 1 and t >= 0:
        intersection_point = ray_origin + t * ray_direction
        return intersection_point
    else:
        return None  # Intersection point lies outside the line segment


def ray_line_intersection(ray_origin, ray_direction, line_point1, line_point2):
    """
    Finds the intersection point of a ray and a line.

    Args:
        ray_origin: The origin of the ray (numpy array)
        ray_direction: The direction vector of the ray (numpy array)
        line_point1: One point on the line (numpy array)
        line_point2: Another point on the line segment (numpy array)

    Returns:
        Intersection point if exists, otherwise None
    """

    # Calculate the direction vector of the line
    line_direction = line_point2 - line_point1

    # Check if the ray and line are parallel (in 2D, np.cross returns a scalar)
    cross_product = np.cross(ray_direction, line_direction)
    if cross_product == 0:
        return None  # Parallel

    # Solve the system of equations for the intersection point
    A = np.array([ray_direction, -line_direction]).T
    b = line_point1 - ray_origin

    try:
        t, u = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None  # No intersection (singular matrix)

    # Check if the intersection point lies on the ray (t >= 0)
    if t >= 0:
        intersection_point = ray_origin + t * ray_direction
        return intersection_point
    else:
        return None  # Intersection is behind the ray origin


def ray_ray_intersection(ray_origin1, ray_direction1, ray_origin2, ray_direction2):
    """
    Finds the intersection point of two rays in 2D.

    Args:
        ray_origin1: The origin of ray 1 (numpy array)
        ray_direction1: The direction vector of ray 1 (numpy array)
        ray_origin2: The origin of ray 2 (numpy array)
        ray_direction2: The direction vector of ray 2 (numpy array)

    Returns:
        Intersection point if it exists (both rays intersect in the forward direction), otherwise None
    """

    # Check if the rays are parallel (cross product in 2D is a scalar)
    cross_product = np.cross(ray_direction1, ray_direction2)
    if cross_product == 0:
        return None  # Parallel rays, no intersection

    # Solve the system of equations for the intersection point
    A = np.array([ray_direction1, -ray_direction2]).T
    b = ray_origin2 - ray_origin1

    try:
        t1, t2 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None  # No intersection (singular matrix)

    # Check if the intersection point lies on both rays (both t1 and t2 should be >= 0)
    if t1 >= 0 and t2 >= 0:
        intersection_point = ray_origin1 + t1 * ray_direction1
        return intersection_point
    else:
        return None  # Intersection point lies behind one or both ray origins
