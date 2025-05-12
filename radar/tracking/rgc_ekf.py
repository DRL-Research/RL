import math
import numpy as np

class RGC_EKF:
    """
    Rectangular Geometric Constraints based Extended Kalman Filter (RGC-EKF).

    Tracks the kinematic state and rectangular shape (length, width) of an extended object
    using radar measurements, based on the paper:
    "Rectangular Geometric Constraints based Extended Object Tracking with Rigorous Shape Estimation"
    by Jiaye Yang et al.
    """

    def __init__(self, initial_state, initial_covariance, motion_noise, measurement_noise, radar_model="polar"):
        """
        Initialize the RGC-EKF filter.

        Args:
            initial_state (list or np.ndarray): Initial state vector [x, y, v, ψ, ψ̇, a, b]
            initial_covariance (np.ndarray): Initial covariance matrix (7x7)
            motion_noise (np.ndarray): Process noise covariance (7x7)
            measurement_noise (np.ndarray): Measurement noise covariance (2x2)
            radar_model (str): Coordinate system of radar ('polar' or 'cartesian')
        """
        self.x = np.array(initial_state, dtype=np.float32)   # state: position, velocity, yaw, shape
        self.P = np.array(initial_covariance, dtype=np.float32)
        self.Q = np.array(motion_noise, dtype=np.float32)
        self.R = np.array(measurement_noise, dtype=np.float32)
        self.model = radar_model

    def predict(self, dt):
        """
        Perform the EKF prediction step using the CTRV motion model.

        Args:
            dt (float): Time difference in seconds
        """
        px, py, v, psi, psi_dot, a, b = self.x

        # Predict position using CTRV model
        if abs(psi_dot) > 1e-3:
            px += (v / psi_dot) * (math.sin(psi + psi_dot * dt) - math.sin(psi))
            py += (v / psi_dot) * (-math.cos(psi + psi_dot * dt) + math.cos(psi))
        else:
            px += v * math.cos(psi) * dt
            py += v * math.sin(psi) * dt
        psi += psi_dot * dt

        # Update predicted state
        self.x[:5] = [px, py, v, psi, psi_dot]
        self.P[:5, :5] += self.Q[:5, :5]  # only update kinematic part

    def fr(self, theta_local, a, b):
        """
        RGC-based radial function: returns the distance from center to contour in direction theta.

        Args:
            theta_local (float): Local angle in object frame
            a (float): Length of the rectangle
            b (float): Width of the rectangle

        Returns:
            float: Distance to contour at angle θ
        """
        theta_threshold = math.atan2(b, a)
        theta_local = (theta_local + np.pi) % (2 * np.pi) - np.pi  # normalize to [-π, π]

        if abs(theta_local) >= theta_threshold and abs(theta_local) <= np.pi - theta_threshold:
            return b / (2 * abs(np.sin(theta_local)))  # vertical edges
        else:
            return a / (2 * abs(np.cos(theta_local)))  # horizontal edges

    def cartesian_to_polar(self, x, y):
        """Convert Cartesian (x, y) to polar (r, theta)."""
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r, theta

    def polar_to_cartesian(self, r, theta):
        """Convert polar (r, theta) to Cartesian (x, y)."""
        return r * np.cos(theta), r * np.sin(theta)

    def h(self, measurement, s=1.0):
        """
        Measurement model: projects radar measurement into expected Cartesian location
        on the rectangle edge using RGC radial function.

        Args:
            measurement (np.ndarray): A radar measurement [x, y]
            s (float): Scale factor (1.0 for contour, <1 for surface)

        Returns:
            np.ndarray: Expected measurement from current state
        """
        x, y, v, psi, psi_dot, a, b = self.x

        # Global angle of measurement
        r, theta_G = self.cartesian_to_polar(measurement[0] - x, measurement[1] - y)
        theta_L = theta_G - psi  # local angle

        fr_val = self.fr(theta_L, a, b)
        px = x + s * fr_val * np.cos(theta_G)
        py = y + s * fr_val * np.sin(theta_G)
        return np.array([px, py])

    def update(self, z_list, s=1.0):
        """
        EKF update step using RGC-based measurement model.

        Args:
            z_list (list of np.ndarray): List of radar measurements [x, y]
            s (float): Scale factor for scattering model (1.0 for contour, <1 for surface)
        """
        for z in z_list:
            hx = self.h(z, s)                  # Expected measurement
            y = z - hx                         # Innovation
            H = np.eye(7)[:2, :]               # Approximate Jacobian (only position part)
            S = H @ self.P @ H.T + self.R      # Innovation covariance
            K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain
            self.x += K @ y
            self.P = (np.eye(7) - K @ H) @ self.P

    def get_state(self):
        """
        Returns the current estimated state and covariance.

        Returns:
            (np.ndarray, np.ndarray): state vector, covariance matrix
        """
        return self.x, self.P