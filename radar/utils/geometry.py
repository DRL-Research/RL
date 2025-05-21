import numpy as np

def polar_to_cartesian(r, theta):
    """Convert polar coordinates (r, θ) to Cartesian (x, y)."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def cartesian_to_polar(x, y):
    """Convert Cartesian coordinates (x, y) to polar (r, θ)."""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def normalize_angle(angle):
    """Normalize angle to the range [-π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def mahalanobis_distance(x, mean, cov):
    """Compute Mahalanobis distance between a point and a Gaussian distribution."""
    diff = x - mean
    return np.sqrt(diff.T @ np.linalg.inv(cov) @ diff)
