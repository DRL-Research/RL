import numpy as np

from radar.utils.geometry import normalize_angle


def generate_synthetic_radar_data(x_true, a, b, num_points=15, noise_std=0.3):
    """
    Generate synthetic radar points from a rectangular object with known state.

    Args:
        x_true (np.ndarray): Ground truth [x, y, v, psi, psi_dot]
        a (float): Length of the rectangle
        b (float): Width of the rectangle
        num_points (int): Number of points to simulate
        noise_std (float): Standard deviation of noise to add

    Returns:
        np.ndarray: Array of radar points (N x 2)
    """
    xc, yc, _, psi, _ = x_true
    angles = np.linspace(0, 2 * np.pi, num_points)
    measurements = []

    for theta in angles:
        # RGC distance function (approx)
        theta = normalize_angle(theta - psi)
        threshold = np.arctan2(b, a)

        if abs(theta) >= threshold and abs(theta) <= np.pi - threshold:
            r = b / (2 * abs(np.sin(theta)))
        else:
            r = a / (2 * abs(np.cos(theta)))

        x = xc + r * np.cos(theta + psi)
        y = yc + r * np.sin(theta + psi)

        # Add Gaussian noise
        x_noisy = x + np.random.normal(0, noise_std)
        y_noisy = y + np.random.normal(0, noise_std)
        measurements.append([x_noisy, y_noisy])

    return np.array(measurements)
