def quadratic_bezier(t, p0, p1, p2):
    u = 1 - t
    return u**2 * p0 + 2 * u * t * p1 + t**2 * p2


def generate_curve_points(p0, p1, p2, num_points=20000):
    curve_points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        point = quadratic_bezier(t, p0, p1, p2)
        curve_points.append(point)
    curve_points = [[p[0], p[1]] for p in curve_points]
    return curve_points
