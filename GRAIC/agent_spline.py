import carla
import math
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
import math
import csv

def save_waypoints_to_csv(waypoints, filename='path.txt'):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write heade
        # Write waypoint coordinates
        for waypoint in waypoints:
            writer.writerow([waypoint[0], waypoint[1]])

def save_curr_to_csv(loc, filename='curr.txt'):
    with open(filename, mode='a', newline='') as file:  # 'a' mode to append to the file
        writer = csv.writer(file)
        writer.writerow([loc[0], loc[1]])




def get_speed(velocity):
    velocity = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    # print(velocity)
    return velocity

class Agent():
    def __init__(self, vehicle=None, L=2.875):
        self.vehicle = vehicle
        self.L = L  # Wheelbase of the vehicle
        # all_filtered_obstacles = []
    
    def calculate_curvature(self,path):
        # Calculate the curvature of a path
        dx_dt = np.gradient(path[:, 0])
        dy_dt = np.gradient(path[:, 1])
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvature = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / ((dx_dt ** 2 + dy_dt ** 2) ** 1.5)
        return curvature
    
    def smooth_path(self,inner_border, outer_border,waypoints):
        inner_border = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in inner_border])
        outer_border = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in outer_border])
        midpoints = (inner_border + outer_border) / 2
        # print(waypoints)
        # midpoints = (midpoints + waypoints[:,:])/ 2
        num_points = len(midpoints)

        # Fit a cubic spline
        t = np.linspace(0, 1, num_points)
        # print(len(midpoints[0]),len(midpoints[1]))

        cs_x = interp1d(t, midpoints[:, 0], kind='cubic')
        cs_y = interp1d(t, midpoints[:, 1], kind='cubic')

        # Generate more points for a smoother path
        num_interpolated_points = 2500  # Increase this number for more resolution
        t_smooth = np.linspace(0, 1, num_interpolated_points)
        x_smooth = cs_x(t_smooth)
        y_smooth = cs_y(t_smooth)

        sigma = 75 # Adjust the sigma value as needed
        x_smooth = gaussian_filter1d(x_smooth, sigma)
        y_smooth = gaussian_filter1d(y_smooth, sigma)

        # Clipping points to stay within track limits (optional, based on your track requirements)
        min_x, max_x = np.min([inner_border[:, 0].min(), outer_border[:, 0].min()]), np.max([inner_border[:, 0].max(), outer_border[:, 0].max()])
        min_y, max_y = np.min([inner_border[:, 1].min(), outer_border[:, 1].min()]), np.max([inner_border[:, 1].max(), outer_border[:, 1].max()])
        x_smooth = np.clip(x_smooth, min_x, max_x)
        y_smooth = np.clip(y_smooth, min_y, max_y)



        return np.column_stack((x_smooth, y_smooth))

    
    # def find_optimal_racing_line(self, inner_border, outer_border):
    #     # Convert Waypoint objects to numpy arrays of coordinates
    #     inner_coords = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in inner_border])
    #     outer_coords = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in outer_border])
    #     midline = (inner_coords + outer_coords) / 2

    #     curvature = self.calculate_curvature(midline)
    #     optimal_line = np.copy(midline)

    #     for i in range(len(optimal_line)):
    #         # Dynamic factor based on curvature
    #         curvature_factor = np.clip(np.abs(curvature[i]), 0.1, 10)
    #         direction = np.array([-midline[i, 1], midline[i, 0]])
    #         if curvature[i] < 0:
    #             direction = np.array([-midline[i, 1], midline[i, 0]])
    #         elif curvature[i] > 0:
    #             direction = np.array([midline[i, 1], -midline[i, 0]])

    #         direction /= np.linalg.norm(direction)
    #         movement = direction * curvature_factor * 30  # Adjust 0.5 to a suitable scaling factor
    #         new_point = midline[i] + movement

    #         # #Ensure the new point is within track limits
    #         if np.linalg.norm(new_point - inner_coords[i]) < np.linalg.norm(inner_coords[i] - outer_coords[i]) and np.linalg.norm(new_point - outer_coords[i]) < np.linalg.norm(inner_coords[i] - outer_coords[i]):
    #             optimal_line[i] = new_point

    #     # Smoothing the line
    #     x_spline = UnivariateSpline(range(len(optimal_line[:, 0])), optimal_line[:, 0], s=1.5)
    #     y_spline = UnivariateSpline(range(len(optimal_line[:, 1])), optimal_line[:, 1], s=1.5)
    #     for i in range(len(optimal_line)):
    #         optimal_line[i, 0] = x_spline(i)
    #         optimal_line[i, 1] = y_spline(i)

    #     return optimal_line


    def control(self, curr_x, curr_y, curr_vel, curr_yaw, waypoints, filtered_obstacles):
        prev_error = 0.0  # Consider storing this as a class variable if you want to use the previous error in the PD controller
        # print(len(waypoints))
        lookahead_distance = 15.0
        kp = 1.2
        kd = 0.5
        max_speed = 0.8
        min_speed = 0.3
        max_angle_error = 65  # degrees
        sharp_turn_threshold = 20  # degrees
        high_speed_threshold = 0.7
        
        max_possible_speed = 30
        brake = 0
        
        
        curr_vel = min(curr_vel / max_possible_speed, 1.0)
        # print(curr_vel)
        # Find lookahead point
        lookahead_point = None
        for waypoint in waypoints:
            if math.dist((curr_x, curr_y), waypoint[:2]) > lookahead_distance:
                lookahead_point = waypoint
                break
        if lookahead_point is None:
            return 0.0, 0.0  # If no lookahead point, return zero speed and steering

        # Calculate heading error
        angle_to_waypoint = math.atan2(lookahead_point[1] - curr_y, lookahead_point[0] - curr_x)
        heading_error = math.atan2(math.sin(angle_to_waypoint - curr_yaw), math.cos(angle_to_waypoint - curr_yaw))
        
        # Sharp turn and high-speed logic
        if abs(math.degrees(heading_error)) > sharp_turn_threshold or curr_vel > high_speed_threshold:
            target_velocity = max(min_speed, curr_vel - min_speed)  # Target slower speed for sharp turn/high speed
        else:
            target_velocity = min(max_speed, curr_vel + 0.05)  # Target higher speed otherwise

        # Decide whether to apply brake or throttle
        if target_velocity < curr_vel:
            # Apply brake if current velocity is greater than target velocity
            brake = max(0.0, min(0.7*target_velocity, (curr_vel - target_velocity) / curr_vel))
            target_velocity = 0.0
        else:
            # Apply throttle if current velocity is less than target velocity
            brake = 0.0
            target_velocity = min(1.0, target_velocity / max_possible_speed)
        
        # PD control for steering
        pd_steering = kp * heading_error + kd * (heading_error - prev_error)
        target_steering = np.arctan(2 * self.L * np.sin(pd_steering) / lookahead_distance)

        # Adjust target velocity based on heading error
        angle_error = abs(math.degrees(heading_error))  # Use heading error for velocity adjustment
        angle_error = angle_error % 360
        # print(angle_error)
        # print(angle_error / max_angle_error)
        normalized_angle_error = min(angle_error / max_angle_error, 1.0)
        target_velocity = max_speed - ((max_speed - min_speed) * normalized_angle_error)  # Decrease speed with increasing angle error




        # Obstacle Avoidane
        
        # obs = np.array([[ob.transform.location.x, ob.transform.location.y] for ob in filtered_obstacles])
        # print(obs)
        # print(filtered_obstacles)


        return target_velocity, target_steering, brake
    



    
    def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary):
        # print(filtered_obstacles)
        current_velocity = get_speed(vel)
        curr_x = transform.location.x
        curr_y = transform.location.y
        curr_yaw = math.radians(transform.rotation.yaw)
        obs = []
        for obs in filtered_obstacles:
            location = obs.get_location()
            obs.append((location.x, location.y))

        # print(obs)

        path = self.smooth_path(boundary[0], boundary[1],waypoints)
        # print(path)
        save_waypoints_to_csv(path)
        target_velocity, steer, brake = self.control(curr_x, curr_y, current_velocity ,curr_yaw, path,filtered_obstacles)
        # Create the control command
        control = carla.VehicleControl()
        control.throttle = target_velocity
        # control.throttle = 0.0
        control.steer = steer
        # print(brake)
        control.brake = float(brake)
        loc = [curr_x, curr_y]
        save_curr_to_csv(loc)
        return control
