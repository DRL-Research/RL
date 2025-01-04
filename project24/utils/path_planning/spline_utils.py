import numpy as np
from scipy import interpolate
from project24.utils import tracker_utils


class PathSpline:
    def __init__(self, x_points, y_points):
        self.xb = x_points  # Base points' x values
        self.yb = y_points  # Base points' y values
        self.xi = None  # Interpolated points' x values
        self.yi = None  # Interpolated points' y values
        self.curvature = None  # Menger curvature of interpolated data
        self.meters_per_index = 0.0
        self.path_length = 0.0  # meters
        self.array_length = int(0)
        self.last_idx = int(0)

    def generate_spline(self, amount, meters=True, smoothing=0, summation=10000):
        """
        Parameters:
        - amount (float or int): If meters is True, represents the distance between 2 points (indices) on the spline.
                                 If meters is False, represents the actual number of indices the spline will have.
        - meters (bool): If True, 'amount' is interpreted as the distance between 2 points in meters.
                         If False, 'amount' is the actual number of indices.
        - smoothing (float): Controls the smoothness of the spline. A higher value results in a smoother spline. Default is 0.
        - summation (int): Number of points used to measure the path length. Default is 10000.

        Sets the following attributes:
        - xi (numpy.ndarray): Array of x-coordinates of the interpolated spline.
        - yi (numpy.ndarray): Array of y-coordinates of the interpolated spline.
        - curvature (numpy.ndarray): Array of curvature values corresponding to each point on the spline.
        - meters_per_index (float): Distance between two consecutive indices on the spline.
        - path_length (float): Total length of the generated path in meters.
        - array_length (int): Number of points in the interpolated spline.
        - last_idx (int): Index of the last point in the interpolated spline.
        """
        # High-level API function to be used for generating a path spline.
        # If meters is True, amount will be roughly the distance between 2 points (indices),
        # Otherwise, amount is the actual number of indices the spline will have.

        # Measure the path length using "summation" number of points regardless of the "amount" choice:
        x_temp, y_temp, c_temp = self._calculate_spline(summation, smoothing)
        total_length = 0.0
        x_diff = np.diff(x_temp) # Computes the differences between consecutive x-coordinates
        y_diff = np.diff(y_temp) # Computes the differences between consecutive y-coordinates

        #Iterates through the differences between consecutive points and computes the Euclidean distance between them.
        for idx in range(summation - 1):
            total_length += np.sqrt(x_diff[idx] ** 2 + y_diff[idx] ** 2) # The distances are summed up to calculate the total length of the spline.

        if meters:
            num_idx = int(total_length / amount) # Calculates num_idx by dividing the total length of the spline by the specified distance between two points (amount).
            self.meters_per_index = total_length / float(num_idx) # set to the actual distance between two consecutive indices by dividing the total length by the calculated num_idx.

        else: # If meters is False, it means that amount directly represents the desired number of indices for the interpolated path.
            num_idx = int(amount)
            self.meters_per_index = total_length / amount

        """
        This logic allows flexibility in how the user can specify the path generation parameters.
        If meters is True, the user provides a target distance between two points on the path, and the code calculates the corresponding number of indices and distance between indices.
        If meters is False, the user directly specifies the desired number of indices, and the code calculates the corresponding distance between indices."""

        #Assigns the returned arrays for x-coordinates (xi), y-coordinates (yi), and curvature (curvature) to the corresponding attributes of the PathSpline instance.
        self.xi, self.yi, self.curvature = self._calculate_spline(num_idx, smoothing)

        # update the attributes
        self.array_length = len(self.xi)
        self.last_idx = self.array_length - 1
        self.path_length = total_length

    def _calculate_spline(self, num_idx, smoothing=0):
        """
        Parameters:
        - num_idx (int): The number of indices or points desired in the interpolated spline.
        - smoothing (float): Controls the smoothness of the spline. A higher value results in a smoother spline. Default is 0.

        Returns:
        - xi (numpy.ndarray): Array of x-coordinates of the interpolated spline.
        - yi (numpy.ndarray): Array of y-coordinates of the interpolated spline.
        - ci (numpy.ndarray): Array of curvature values corresponding to each point on the spline.
        """
        # Fits splines to x=f(u) and y=g(u), treating both as periodic. Also note that s=0
        # is used only to force the spline fit to pass through all the input points.
        # interpolate.splprep fits splines to the given set of base points.
        # s=smoothing controls the smoothness of the spline. A value of 0 is used to force the spline fit to pass through all input points.
        # per=True treats the data as periodic, which means that the spline will be forced to be periodic (closed) in both dimensions.
        # quiet=1 suppresses warnings.
        # tck - A tuple, (t,c,k) containing the vector of knots, the B-spline coefficients, and the degree of the spline.
        # u is an array of the parameters used for the spline interpolation. we dont use it in the code.
        tck, u = interpolate.splprep([self.xb, self.yb], s=smoothing, per=False, quiet=1)

        # Evaluate the spline fits for 10000 evenly spaced distance values:
        xi, yi = interpolate.splev(np.linspace(0, 1, num_idx), tck) # the x and y coordinates of the interpolated path
        # xi, yi = self.xb, self.yb  # the x and y coordinates of the interpolated path

        # Evaluate the curvature for each and every point using the Menger curvature formula
        #t he Menger curvature function (self.menger_curvature) is called for each triplet of consecutive points.
        ci = np.zeros(num_idx)
        for idx in range(num_idx - 2):
            if idx == 0:
                pass
            else:
                first_p = np.array([xi[idx - 1], yi[idx - 1]])
                second_p = np.array([xi[idx], yi[idx]])
                third_p = np.array([xi[idx + 1], yi[idx + 1]])
                ci[idx] = self.menger_curvature(first_p, second_p, third_p)

        # Handle Curvature at End Points
        ci[0] = ci[1]
        ci[-1] = ci[-2]

        # Returns the x and y coordinates of the interpolated path, along with the calculated curvature (ci) at each point.
        return xi, yi, ci

    def find_closest_point(self, target_point):
        """
        Parameters:
        - target_point (tuple): A tuple containing the (x, y) coordinates of the target point.

        Returns:
        - closest_idx (int): Index of the closest point on the interpolated spline.
        - closest_vector (numpy.ndarray): Vector pointing from the target point to the closest point on the spline.
        - closest_tangent (numpy.ndarray): Tangent vector at the closest point on the spline.
        """
        # Finding the closest point using "brute force", no optimization.
        # Minimum for norm will be minimum for norm squared as well, dropping the sqrt():

        distances = (self.xi - target_point[0])**2 + (self.yi - target_point[1])**2 # Computes the squared distances from the target point to each point on the interpolated spline.
        closest_idx = distances.argmin()

        # Handle Special Cases for First and Last Points
        if closest_idx == 0:
            prev_idx = self.last_idx
            next_idx = closest_idx + 1
        elif closest_idx == self.last_idx:
            prev_idx = closest_idx - 1
            next_idx = 0
        else:
            prev_idx = closest_idx - 1
            next_idx = closest_idx + 1

        closest_vector = np.array([self.xi[closest_idx] - target_point[0], self.yi[closest_idx] - target_point[1]]) # Computes the vector from the target point to the closest point on the spline
        closest_tangent = np.array([self.xi[next_idx] - self.xi[prev_idx], self.yi[next_idx] - self.yi[prev_idx]]) # Computes the tangent vector at the closest point using the neighboring points
        if np.linalg.norm(closest_tangent) > 1e-6:
            closest_tangent /= np.linalg.norm(closest_tangent)  # Normalize to a length of 1
        # else ?

        return closest_idx, closest_vector, closest_tangent

    @staticmethod
    def menger_curvature(prev_p, curr_p, next_p):
        """
        Calculate the Menger curvature at a given point on a curve defined by three consecutive points.

        Parameters:
        - prev_p (numpy.ndarray): Coordinates of the previous point.
        - curr_p (numpy.ndarray): Coordinates of the current point.
        - next_p (numpy.ndarray): Coordinates of the next point.

        Returns:
        - curvature (float): Menger curvature at the current point.
        """
        prev_unit_vec = (curr_p - prev_p) / np.linalg.norm(curr_p - prev_p)
        next_unit_vec = (next_p - curr_p) / np.linalg.norm(next_p - curr_p)
        # Ensure dot product is within valid range for arccos
        dot_product = np.clip(np.dot(prev_unit_vec, next_unit_vec), -1.0, 1.0)

        base_angle = np.arccos(dot_product)
        base_length = np.linalg.norm(next_p - prev_p)
        curvature = 2 * np.sin(base_angle) / base_length
        if curvature < 1e-4:  # Meaningless value for our purpose
            curvature = 0
        return curvature


def generate_path_points(mapping_data):
    """
    Parameters:
    - mapping_data (list): List of tracked cone objects.

    Returns:
    - track_points (numpy.ndarray): Array of 2D coordinates representing the generated path points.
    """
    # Uses the list of tracked cones in order to generate a valid spline:
    left_points = np.ndarray(shape=(0, 2))
    right_points = np.ndarray(shape=(0, 2))
    unknown_points = np.ndarray(shape=(0, 2))

    for tracked_obj in mapping_data: # Iterate through each tracked cone in the mapping_data
        if tracked_obj.active:
            point = tracked_obj.position[0:2]
            if tracked_obj.color == tracker_utils.ConeTracker.COLOR_BLUE:
                left_points = np.append(left_points, point.reshape(1, 2), axis=0)
            elif tracked_obj.color == tracker_utils.ConeTracker.COLOR_YELLOW:
                right_points = np.append(right_points, point.reshape(1, 2), axis=0)
            else:
                # Qualify unknown cones as the closest color (potentially dangerous):
                left_dist = np.linalg.norm(left_points - point, axis=1)
                right_dist = np.linalg.norm(right_points - point, axis=1)
                if np.min(left_dist) < np.min(right_dist):
                    left_points = np.append(left_points, point.reshape(1, 2), axis=0) # array([x1,y1],
                                                                                            # [x2,y2]...)
                else:
                    right_points = np.append(right_points, point.reshape(1, 2), axis=0)
                unknown_points = np.append(unknown_points, point.reshape(1, 2), axis=0)

    # Assuming somewhat equal cone detections on each side!
    # To ensure that there are an equal number of points on both the left and right sides, the minimum length of left_points and right_points arrays is calculated (min_length).
    min_length = min(left_points.shape[0], right_points.shape[0])

    # track_points is then calculated as the average of these selected points.
    track_points = (left_points[:min_length, :] + right_points[:min_length, :]) / 2 # array([x1,y1],
                                                                                          # [x2,y2]...)

    return track_points




