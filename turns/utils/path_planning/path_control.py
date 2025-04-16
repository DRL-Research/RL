import multiprocessing
import struct
import time
from multiprocessing import shared_memory
import numpy as np
from turns.utils.path_planning.discrete_plant_emulator import DiscretePlant
from turns.utils.path_planning.pidf_controller import PidfControl


class StanleyFollower:
    EPSILON = 1e-4  # For numerical stability

    def __init__(self, path_spline, max_velocity=10, min_velocity=5, lookahead=5.0, k_steer=10.0):

        self.path = path_spline

        # Max steering MUST be the same as your simulated vehicle's settings within Unreal.
        self.max_velocity = max_velocity  # m/s
        self.min_velocity = min_velocity # m/s
        self.max_steering = np.deg2rad(80.0)  # radians

        # Velocity coefficient defaults to the difference between max and min speeds,
        # divided by the maximum path curvature:
        self.k_vel = (self.max_velocity - self.min_velocity) / (self.path.curvature.max() + self.EPSILON)
        self.lookahead = lookahead  # meters
        self.k_steer = k_steer  # Stanley steering coefficient

    def calc_ref_speed_steering(self, car_pos, car_vel, heading): ## function to handle the speed and steering
        """
        Parameters:
        - car_pos (numpy.ndarray): Current vehicle position (x, y).
        - car_vel (float): Current vehicle velocity.
        - heading (float): Current vehicle heading angle.

        Returns:
        - speed (float): Reference speed for the vehicle.
        - steering_angle (float): Reference steering angle for the vehicle.
        """
        # First we match the path's tangent angle [rad]:
        closest_idx, closest_vector, closest_tangent = self.path.find_closest_point(car_pos)
        path_direction = np.arctan2(closest_tangent[1], closest_tangent[0])
        theta_e = heading - path_direction #theta_e represents the difference between the current heading of the vehicle and the tangent angle of the desired path at the closest point.
        if theta_e > 1.5 * np.pi:
            theta_e -= 2.0 * np.pi
        if theta_e < -1.5 * np.pi:
            theta_e += 2.0 * np.pi

        # Second we calculate the closing steering angle [rad]:
        closest_dist = np.linalg.norm(closest_vector)
        ego_rotation = np.array([[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]])
        ego_closest = np.matmul(ego_rotation, closest_vector)  # Closest vector in vehicle frame of reference

        theta_f = -np.arctan(self.k_steer * closest_dist / (np.abs(car_vel) + self.EPSILON))*np.sign(ego_closest[1]) #theta_f represents a closing angle based on the distance to the closest point on the path and the vehicle's velocity.

        # Combine them together and saturate:
        steering_angle = np.clip(theta_e + theta_f, -self.max_steering, self.max_steering) # ensures that the resulting steering angle is within the specified steering limits, which are defined by self.max_steering and -self.max_steering.

        # Then we handle the desired speed [m/s]:
        # lookahead_idx is calculated by taking the index of the closest point on the path (closest_idx) and adding an offset based on the lookahead distance.
        # The division by (self.path.meters_per_index + self.EPSILON) is likely to convert the lookahead distance from meters to path indices.
        lookahead_idx = closest_idx + int(np.around(self.lookahead / (self.path.meters_per_index + self.EPSILON)))

        # This block of code ensures that if the lookahead index exceeds the length of the path array, it wraps around to the beginning of the path. This is relevant when the vehicle is close to the end of the path.
        if lookahead_idx >= self.path.array_length:
            lookahead_idx = lookahead_idx - self.path.array_length


        # The desired speed (speed) is calculated based on the maximum velocity (self.max_velocity) minus a term that depends on the curvature of the path at the lookahead point.
        # self.k_vel is a coefficient that scales the effect of curvature on speed. The higher the curvature, the lower the desired speed.
        speed = self.max_velocity - self.k_vel * self.path.curvature[lookahead_idx]

        # The calculated speed is then clipped to ensure it does not fall below a minimum velocity (self.min_velocity). This is done to prevent the vehicle from slowing down too much.
        speed = max(speed, self.min_velocity)

        return speed, steering_angle


class PursuitFollower:
    EPSILON = 1e-4  # For numerical stability

    def __init__(self, min_distance, max_distance):

        self.min_lookahead = min_distance
        self.max_lookahead = max_distance # this maximum lookahead distance sets a limit on how far into the future the vehicle should plan its path based on those cones.
        self.previous_angle = 0.0  # Optional as fallback angle instead of moving straight ahead.
        self.k_steer = 0.5
        self.max_steering = np.deg2rad(80.0)  # Radians
        self.pursuit_points = [np.array([0.0, 0.0, 0.0])]

    def calc_ref_steering(self, tracked_cones, map_to_vehicle):
        """
        Parameters:
        - tracked_cones (list): List of tracked cone objects.
        - map_to_vehicle (numpy.ndarray): Transformation matrix from map to vehicle coordinates.

        Returns:
        - steering_angle (float): Reference steering angle for the vehicle.
        """
        # Manage pursuit points list:
        last_blue = None
        last_yellow = None
        for curr_cone in reversed(tracked_cones): # Iterate through the tracked cones to find the most recent blue and yellow cones.
            if curr_cone.active:
                if last_blue is None and curr_cone.color == curr_cone.COLOR_BLUE:
                    last_blue = curr_cone.position
                if last_yellow is None and curr_cone.color == curr_cone.COLOR_YELLOW:
                    last_yellow = curr_cone.position
                if last_blue is not None and last_yellow is not None:
                    last_average = (last_blue + last_yellow) / 2.0
                    pursuit_diff = np.linalg.norm(self.pursuit_points[-1] - last_average)

                    # Only add the current point to the list if it is not the same as the last one:
                    if pursuit_diff > 1.0:
                        self.pursuit_points.append(last_average)
                    break

        # Scanning pursuit points from newest (and farthest) backwards.
        # Keep scanning as long as the point is farther than the max lookahead distance.
        # If we reached a point that is too close, keep moving straight until (hopefully) a new one gets within range.
        if len(self.pursuit_points) > 2:
            for point in reversed(self.pursuit_points): #  scanning pursuit points from the newest to the oldest
                # Compute pursuit point in vehicle coordinates:
                pursuit_map = np.append(point, 1)
                pursuit_vehicle = np.matmul(map_to_vehicle, pursuit_map)[:3]
                pursuit_distance = np.linalg.norm(pursuit_vehicle)
                if pursuit_distance < self.max_lookahead: #This ensures that the pursuit behavior focuses on cones that are within a certain distance ahead of the vehicle, and points beyond this threshold are not considered for steering guidance.
                    break
            if pursuit_distance > self.min_lookahead:
                steering_angle = -self.k_steer * np.arctan(pursuit_vehicle[1] / (pursuit_vehicle[0] + self.EPSILON))
                self.previous_angle = np.clip(steering_angle, -self.max_steering, self.max_steering)
            else:
                # If the pursuit point is not within the acceptable range, the vehicle is instructed to move straight
                steering_angle = 0.0
                # steering_angle = self.previous_angle  # Optional fallback.
        else:
            steering_angle = 0.0  # Move straight if we haven't picked up any points yet.

        steering_angle = np.clip(steering_angle, -self.max_steering, self.max_steering)

        return steering_angle


class SteeringProcManager:
    shmem_active = None
    shmem_setpoint = None
    shmem_output = None

    steer_emulator = None
    steer_controller = None
    steering_thread = None

    memories_exist = False

    def __init__(self):
        if not self.is_multiprocessing():
            self.create_steering_procedure()
        else:
            print("Running in multiprocessing context, skipping steering procedure creation.")

    def __del__(self):
        self.terminate_steering_procedure()

    @classmethod
    def create_steering_procedure(cls):
        # Everything below is hardcoded, changes can be made by adding arguments:
        if not cls.memories_exist:
            cls.memories_exist = True
            # Create or pick up existing shared memory objects:
            try:
                cls.shmem_active = shared_memory.SharedMemory(name='active_state', create=True, size=1)
                cls.shmem_setpoint = shared_memory.SharedMemory(name='input_value', create=True, size=8)
                cls.shmem_output = shared_memory.SharedMemory(name='output_value', create=True, size=8)
            except FileExistsError:
                cls.shmem_active = shared_memory.SharedMemory(name='active_state', create=False, size=1)
                cls.shmem_setpoint = shared_memory.SharedMemory(name='input_value', create=False, size=8)
                cls.shmem_output = shared_memory.SharedMemory(name='output_value', create=False, size=8)

            # Initialize default values:
            is_active = True
            desired_steer = 0.0
            real_steer = 0.0
            # These values are then packed into the corresponding shared memory buffers using struct.pack.
            cls.shmem_active.buf[:1] = struct.pack('?', is_active)
            cls.shmem_setpoint.buf[:8] = struct.pack('d', desired_steer)
            cls.shmem_output.buf[:8] = struct.pack('d', real_steer)

            # Create an emulated plant and a controller:
            sample_time = 0.001 # The time between consecutive samples
            cls.steer_emulator = DiscretePlant(sample_time, 10, 4)
            cls.steer_controller = PidfControl(sample_time)
            cls.steer_controller.set_pidf(1000.0, 0.0, 15.0, 0.0)
            cls.steer_controller.set_extrema(0.01, 1.0)
            cls.steer_controller.alpha = 0.01

            if not cls.is_multiprocessing():
                # Initiate a new process for steering:
                cls.steering_thread = multiprocessing.Process(target=cls.steer_emulator.async_steering,
                                                              args=(sample_time, cls.steer_controller),
                                                              daemon=True)
                cls.steering_thread.start()
                time.sleep(2.0)  # New process takes a lot of time to "jumpstart"

    @classmethod
    def retrieve_shared_memories(cls):
        if not cls.is_multiprocessing():
            cls.create_steering_procedure()

        shmem_active = shared_memory.SharedMemory(name='active_state', create=False)
        shmem_setpoint = shared_memory.SharedMemory(name='input_value', create=False)
        shmem_output = shared_memory.SharedMemory(name='output_value', create=False)

        return shmem_active, shmem_setpoint, shmem_output

    @classmethod
    def detach_shared_memories(cls, shared_memories_list=None): # close shared memories
        if shared_memories_list is None:
            cls.shmem_active.buf[:1] = struct.pack('?', False)
            cls.shmem_active.close()
            cls.shmem_setpoint.close()
            cls.shmem_output.close()
        else:
            for shmem in shared_memories_list:
                shmem.close()

    @classmethod
    def terminate_steering_procedure(cls):
        # Should only be used if all processes have called detach_shared_memories!
        if cls.memories_exist:
            cls.shmem_active.buf[:1] = struct.pack('?', False)
            cls.shmem_active.unlink() #  This removes the shared memory objects, making them unavailable for further use.
            cls.shmem_setpoint.unlink()
            cls.shmem_output.unlink()
            if cls.steering_thread is not None:
                cls.steering_thread.terminate()
            cls.memories_exist = False

    @classmethod
    def is_multiprocessing(cls):
        return multiprocessing.current_process().name != "MainProcess"


# The function below was not used and left for reference only:
def calc_dead_reckoning(car_pos, car_speed, heading, yaw_rate, delta_time):
    updated_heading = heading + delta_time * yaw_rate
    updated_pos = car_pos + delta_time * car_speed * np.array([np.cos(updated_heading), np.sin(updated_heading)])
    return updated_pos, updated_heading

