import airsim
import numpy as np
import matplotlib.pyplot as plt
import time
from tracking.rgc_ekf import RGC_EKF
from utils.radar_data import generate_synthetic_radar_data

def get_initial_state_from_airsim(client, vehicle_name="Car1", length=5.0, width=2.0):
    pose = client.simGetObjectPose(vehicle_name)
    yaw = airsim.to_eularian_angles(pose.orientation)[2]
    return [
        pose.position.x_val,
        pose.position.y_val,
        0.0,  # v
        yaw,
        0.0,  # psi_dot
        length,
        width
    ]

def compute_box_corners(x, y, psi, a, b):
    dx = a / 2
    dy = b / 2
    corners_local = np.array([
        [dx, dy],
        [dx, -dy],
        [-dx, -dy],
        [-dx, dy],
        [dx, dy]
    ])
    R = np.array([
        [np.cos(psi), -np.sin(psi)],
        [np.sin(psi), np.cos(psi)]
    ])
    return corners_local @ R.T + np.array([x, y])

def main(follow, learn):
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True, learn)
    controls = airsim.CarControls()
    controls.throttle = 0.2 # Range: 0 to 1
    controls.steering = 0.0  # Range: -1 (left) to 1 (right)
    controls.brake = 0.0
    client.setCarControls(controls, vehicle_name=learn)
    time.sleep(2.0)

    car_length = 5.0
    car_width = 2.0
    dt = 0.1

    initial_state = get_initial_state_from_airsim(client, follow, car_length, car_width)
    tracker = RGC_EKF(
        initial_state,
        initial_covariance=np.eye(7) * 0.5,
        motion_noise=np.eye(7) * 0.1,
        measurement_noise=np.eye(2) * 0.2
    )

    trajectory = []
    shape_dims = []
    true_states = []

    for step in range(100):
        client.simFlushPersistentMarkers()
        # Get updated pose after move
        pose = client.simGetObjectPose(follow)
        yaw = airsim.to_eularian_angles(pose.orientation)[2]
        car_x = pose.position.x_val
        car_y = pose.position.y_val
        car_z = pose.position.z_val
        true_state = np.array([car_x, car_y, 0.0, yaw, 0.0])
        measurements = generate_synthetic_radar_data(true_state, car_length, car_width)

        tracker.predict(dt)
        tracker.update(measurements)
        x_est, _ = tracker.get_state()

        trajectory.append((x_est[0], x_est[1]))
        shape_dims.append((x_est[5], x_est[6]))
        true_states.append(true_state.copy())

        # --- AirSim visual ---
        z = car_z + 0.2
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

        print(f"[{step:02d}] Tracker: x={x_est[0]:.2f}, y={x_est[1]:.2f}, yaw={np.degrees(x_est[3]):.1f}°, "
              f"a={x_est[5]:.2f}, b={x_est[6]:.2f}")
        time.sleep(0.1)

    # === Stop Car ===
    controls.throttle = 0.0
    controls.steering = 0.0
    controls.brake = 1.0
    client.setCarControls(controls, vehicle_name=learn)

    # --- 1. Distance between EKF and Ground Truth box centers ---
    distances = [
        np.linalg.norm(np.array(trajectory[i]) - np.array([true_states[i][0], true_states[i][1]]))
        for i in range(len(trajectory))
    ]

    plt.figure(figsize=(10, 4))
    plt.plot(distances, label="Center Distance (EKF vs GT)", color='purple')
    plt.xlabel("Time Step")
    plt.ylabel("Distance (meters)")
    plt.title("EKF vs Ground Truth Center Distance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 2. 10 Subplots showing red and blue box positions (1 box per 10 steps) ---
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(10):
        ax = axes[i]
        j = i * 10  # only the first step in each 10-step block

        # --- Red (EKF) box ---
        x_est_step = trajectory[j]
        yaw_est = true_states[j][3]  # you can also store and use x_est yaws if needed
        est_box = compute_box_corners(*x_est_step, yaw_est, *shape_dims[j])
        est_box = np.vstack([est_box, est_box[0]])  # close the box loop
        est_x, est_y = zip(*est_box)
        ax.plot(est_x, est_y, 'r-', linewidth=2, label='EKF Estimate')

        # --- Blue (GT) box ---
        gt_x, gt_y, gt_yaw = true_states[j][0], true_states[j][1], true_states[j][3]
        gt_box = compute_box_corners(gt_x, gt_y, gt_yaw, car_length, car_width)
        gt_box = np.vstack([gt_box, gt_box[0]])  # close the box loop
        gt_xs, gt_ys = zip(*gt_box)
        ax.plot(gt_xs, gt_ys, 'b--', linewidth=2, label='Ground Truth')

        ax.set_title(f"Step {j}")
        ax.axis("equal")
        ax.grid(True)
        ax.legend()

    plt.suptitle("Red vs Blue Box Placement Every 10 Steps")
    plt.tight_layout()
    plt.show()

    # === 3. Center Distance Over Time ===
    plt.plot(distances)
    plt.axhline(5.0, color='gray', linestyle='--', label='5 meters reference')
    plt.legend()
    plt.figure(figsize=(10, 4))
    plt.plot(distances, label="Tracking Error (EKF vs GT)", color='blue')
    plt.axhline(5.0, color='gray', linestyle='--', label='5 meters reference')
    plt.xlabel("Time Step")
    plt.ylabel("Center Distance (meters)")
    plt.title("EKF vs Ground Truth Center Distance Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # the learning car is moving, the following car is standing still
    main(follow="Car2", learn="Car1")
