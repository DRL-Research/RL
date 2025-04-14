import airsim
import numpy as np
import time
from collections import defaultdict

client = airsim.CarClient()
client.confirmConnection()

vehicle_name = "Car1"
sensor_name = "RadarSim"

print("Radar simulation started...")

# Previous frame's points: dict[tuple[int azimuth]] -> list of (range, elevation, x, y, z)
previous_points = {}
previous_time = time.time()

def cartesian_to_polar(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    return r, azimuth, elevation

while True:
    lidar_data = client.getLidarData(lidar_name=sensor_name, vehicle_name=vehicle_name)

    if len(lidar_data.point_cloud) < 3:
        print("No points detected")
        time.sleep(0.1)
        continue

    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    current_time = time.time()
    dt = current_time - previous_time if previous_time else 1e-3
    previous_time = current_time

    angle_bins = defaultdict(list)
    velocities = {}

    for x, y, z in points:
        r, azimuth, elevation = cartesian_to_polar(x, y, z)
        azimuth_bin = int(round(azimuth))
        angle_bins[azimuth_bin].append((r, elevation, x, y, z))

        # Try matching previous point to calculate velocity
        if azimuth_bin in previous_points:
            prev = previous_points[azimuth_bin][0]  # Assume closest is always at index 0
            _, _, px, py, pz = prev
            vx = (x - px) / dt
            vy = (y - py) / dt
            vz = (z - pz) / dt
            velocity = np.sqrt(vx**2 + vy**2 + vz**2)
            velocities[azimuth_bin] = velocity

    print(f"\n--- Frame at t={current_time:.2f}, dt={dt:.2f}s ---")
    for az in sorted(angle_bins.keys())[:10]:
        r, elevation, x, y, z = min(angle_bins[az], key=lambda p: p[0])
        v = velocities.get(az, 0.0)
        print(f"Azimuth {az}° | Range={r:.2f}m | Elev={elevation:.1f}° | Vel={v:.2f} m/s")

    # Save this frame’s closest points
    previous_points = {az: [min(angle_bins[az], key=lambda p: p[0])] for az in angle_bins}

    time.sleep(0.5)
