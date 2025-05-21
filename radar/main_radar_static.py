import numpy as np
import matplotlib.pyplot as plt
import airsim
from tracking.rgc_ekf import RGC_EKF
from utils.geometry import normalize_angle
from utils.radar_data import generate_synthetic_radar_data


def get_initial_state_from_airsim(client, vehicle_name="Car1", length=5.0, width=2.0):
    pose = client.simGetObjectPose(vehicle_name)
    yaw = airsim.to_eularian_angles(pose.orientation)[2]
    return [
        pose.position.x_val,
        pose.position.y_val,
        0.0,  # initial velocity
        yaw,
        0.0,  # yaw rate
        length,
        width
    ]

# --- Visualize tracker state ---
def compute_box_corners(x, y, psi, a, b):
    dx = a / 2
    dy = b / 2
    corners_local = np.array([
        [dx, dy],
        [dx, -dy],
        [-dx, -dy],
        [-dx, dy],
        [dx, dy]  # close the box
    ])
    R = np.array([
        [np.cos(psi), -np.sin(psi)],
        [np.sin(psi), np.cos(psi)]
    ])
    return corners_local @ R.T + np.array([x, y])

def main():
    client = airsim.CarClient()
    client.confirmConnection()

    # === Get initial state from Car1's actual visual pose ===
    car_length = 5.0
    car_width = 2.0
    initial_state = get_initial_state_from_airsim(client, "Car1", car_length, car_width)

    initial_cov = np.eye(7) * 0.5
    motion_noise = np.eye(7) * 0.1
    measurement_noise = np.eye(2) * 0.2
    tracker = RGC_EKF(initial_state, initial_cov, motion_noise, measurement_noise)

    # === Ground Truth Initialization ===
    true_state = np.array(initial_state[:5])  # [x, y, v, psi, psi_dot]
    dt = 0.1

    trajectory = []
    shape_dims = []

    for step in range(30):
        # Simulate motion
        x, y, v, psi, psi_dot = true_state
        x += v * np.cos(psi) * dt
        y += v * np.sin(psi) * dt
        psi += psi_dot * dt
        true_state = np.array([x, y, v, psi, psi_dot])

        # Generate radar points from updated position
        measurements = generate_synthetic_radar_data(true_state, car_length, car_width)

        tracker.predict(dt)
        tracker.update(measurements)
        x_est, _ = tracker.get_state()

        trajectory.append((x_est[0], x_est[1]))
        shape_dims.append((x_est[5], x_est[6]))

    # === Plot Estimated Trajectory ===
    trajectory = np.array(trajectory)
    shape_dims = np.array(shape_dims)



    # EKF box (red)
    est_box = compute_box_corners(x_est[0], x_est[1], x_est[3], x_est[5], x_est[6])
    est_box_airsim = [airsim.Vector3r(pt[0], pt[1], 0.1) for pt in est_box]
    client.simPlotLineStrip(est_box_airsim, color_rgba=[1, 0, 0, 1], thickness=8.0, is_persistent=False)

    # Ground truth box (blue)
    gt_box = compute_box_corners(true_state[0], true_state[1], true_state[3], car_length, car_width)
    gt_box_airsim = [airsim.Vector3r(pt[0], pt[1], 0.1) for pt in gt_box]
    client.simPlotLineStrip(gt_box_airsim, color_rgba=[0, 0, 1, 1], thickness=6.0, is_persistent=False)

    # EKF center dot (green)
    client.simPlotPoints([airsim.Vector3r(float(x_est[0]), float(x_est[1]), 0.1)],
                         color_rgba=[0, 1, 0, 1], size=15.0, is_persistent=False)

    print(
        f"[{step:02d}] Tracker: x={x_est[0]:.2f}, y={x_est[1]:.2f}, yaw={np.degrees(x_est[3]):.1f}°, a={x_est[5]:.2f}, b={x_est[6]:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], label="Estimated Trajectory", marker='o')
    plt.title("RGC-EKF Estimated Trajectory (from Car1 visual pose)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()

    # === Plot Shape Estimates ===
    plt.figure(figsize=(8, 4))
    plt.plot(shape_dims[:, 0], label="Estimated Length (a)")
    plt.plot(shape_dims[:, 1], label="Estimated Width (b)")
    plt.title("Estimated Shape Dimensions Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Dimension (m)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
