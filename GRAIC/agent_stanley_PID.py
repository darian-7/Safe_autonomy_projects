import carla
import time
import numpy as np
import numpy as np
from matplotlib import pyplot as plt
from collections import deque
import math
import numpy as np
from scipy.interpolate import CubicSpline
from agents.tools.misc import get_speed

class StanleyController:
    """
    Stanley controller for lateral control
    Assumes rear axle model of the car
    """
    def __init__(self, x=0, y=0, yaw=0, v=0, delta=0, max_steering_angle=1.22, L=2.875, K=0.0):
        # States
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

        # Steering angle
        self.delta = delta
        self.max_steering_angle = max_steering_angle

        # Wheel base
        self.L = L

        # Control gain
        self.K = K

    def update_position(self, x, y):
        self.x = x
        self.y = y

    def update_speed(self, v):
        self.v = v

    def update_yaw(self, yaw):
        self.yaw = yaw

    def update_steering_angle(self, steer):
        self.delta = steer * self.max_steer_angle

    def get_steer_input(self, x, y, yaw, v, waypoints):
        self.update_position(x, y)
        self.update_yaw(yaw)
        self.update_speed(v)

        x_f = self.x + self.L * np.cos(self.yaw)
        y_f = self.y + self.L * np.sin(self.yaw)

        cx = waypoints[0]
        cy = waypoints[1]
        # cyaw = waypoints[2]

        distances = np.sum(( np.array([[x_f], [y_f]]) - np.stack((cx, cy)) )**2, axis=0)
        idx = np.argmin(distances)
        cte = distances[idx]

        if idx != len(waypoints[0]):
            desired_heading = np.arctan2(cy[idx+1] - cy[idx], cx[idx+1] - cx[idx])
        else:
            desired_heading = np.arctan2(cy[idx] - cy[idx-1], cx[idx] - cx[idx-1])
        # desired_heading = cyaw[idx]
        heading_error = desired_heading - self.yaw
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        target_x, target_y = cx[idx], cy[idx]

        ##### Method 1 #############
        # dx = x_f - target_x
        # dy = y_f - target_y
        # front_vec = [-np.cos(self.yaw + np.pi / 2),
        #               - np.sin(self.yaw + np.pi / 2)]
        # cte = np.dot([dx, dy], front_vec)

        ##### Method 2 #############
        yaw_ct2vehicle = np.arctan2(y_f - target_y, x_f - target_x)
        yaw_ct2heading = desired_heading - yaw_ct2vehicle
        yaw_ct2heading = np.arctan2(np.sin(yaw_ct2heading), np.cos(yaw_ct2heading))
        cte *= np.sign(yaw_ct2heading)

        steer = heading_error + np.arctan2(self.K * cte, self.v)
        self.delta = steer
        print(steer)
        return steer


class LongitudinalPID:
    """
    PID controller for longitudinal control
    """
    def __init__(self, v=0, L=2.875, Kp=0.01, Kd=0.0, Ki=0.0,
                 integrator_min=None, integrator_max=None):
        # States
        self.v = v
        self.prev_error = 0
        self.sum_error = 0

        # Wheel base
        self.L = L

        # Control gain
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integrator_min = integrator_min
        self.integrator_max = integrator_max

    def update_speed(self, v):
        self.v = v

    def get_throttle_input(self, v, dt, target_speed):
        self.update_speed(v)

        error = target_speed - self.v
        self.sum_error += error * dt
        if self.integrator_min is not None:
            self.sum_error = np.fmax(self.sum_error,
                                     self.integrator_min)
        if self.integrator_max is not None:
            self.sum_error = np.fmin(self.sum_error,
                                     self.integrator_max)

        throttle = self.Kp * error + \
            self.Ki * self.sum_error + \
            self.Kd * (error - self.prev_error) / dt
        self.prev_error = error

        return throttle
def get_speed(velocity):
    velocity = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    # print(velocity)
    return velocity

class Agent():

    def __init__(self, target_speed=30.0):
        # Initialize controllers
        self.stanley_controller = StanleyController()
        self.longitudinal_pid = LongitudinalPID()

        # Target speed (m/s)
        self.target_speed = target_speed

    def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary):
        """
        Execute one step of navigation.

        Args:
        filtered_obstacles
            - Type:        List[carla.Actor(), ...]
            - Description: All actors except for EGO within sensoring distance
        waypoints 
            - Type:         List[[x,y,z], ...] 
            - Description:  List All future waypoints to reach in (x,y,z) format
        vel
            - Type:         carla.Vector3D 
            - Description:  Ego's current velocity in (x, y, z) in m/s
        transform
            - Type:         carla.Transform 
            - Description:  Ego's current transform
        boundary 
            - Type:         List[List[left_boundry], List[right_boundry]]
            - Description:  left/right boundary each consists of 20 waypoints,
                            they defines the track boundary of the next 20 meters.

        Return: carla.VehicleControl()
        """
        # Actions to take during each simulation step
        # Feel Free to use carla API; however, since we already provide info to you, using API will only add to your delay time
        # Currently the timeout is set to 10s

        # 
        # print(waypoints)
        print("Reach Customized Agent")


        current_x, current_y, current_yaw = transform.location.x, transform.location.y, transform.rotation.yaw
        current_speed = get_speed(vel)  # Assuming get_speed is a utility function to convert velocity to speed

        # Update Stanley Controller state
        if waypoints:
            waypoint_xs = [point[0] for point in waypoints]
            waypoint_ys = [point[1] for point in waypoints]
            
            steer = self.stanley_controller.get_steer_input(current_x, current_y, current_yaw, current_speed, [waypoint_xs, waypoint_ys])
        else:
            steer = 0.0

        # Update Longitudinal PID state
        dt = 1/60  # Assuming a time step of 0.05 seconds
        throttle = self.longitudinal_pid.get_throttle_input(current_speed, dt, self.target_speed)

        # Formulate Vehicle Control
        control = carla.VehicleControl()
        control.steer = np.clip(steer, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.brake = 0.0  # Add logic for braking if necessary

        return control
