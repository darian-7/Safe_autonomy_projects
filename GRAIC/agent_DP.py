import carla
import math
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
import csv

def save_waypoints_to_csv(waypoints, filename='path_dp.txt'):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write heade
        # Write waypoint coordinates
        for waypoint in waypoints:
            writer.writerow([waypoint[0], waypoint[1]])

def save_curr_to_csv(loc, filename='curr_dp.txt'):
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


    def divide_track(self,inner_border, outer_border, divisions):
        divided_points = []
        for inner, outer in zip(inner_border, outer_border):
            segment = [inner + (outer - inner) * i / (divisions - 1) for i in range(divisions)]
            divided_points.append(segment)
        return np.array(divided_points)  # Shape will be (n, divisions, 2)

    def generate_waypoints(self,divided_points):
        num_segments = len(divided_points)
        # Assuming each segment has the same number of lanes
        num_lanes = len(divided_points[0])
        # Create a 2D grid of waypoints with shape (num_segments, num_lanes, 2)
        waypoints_grid = np.zeros((num_segments, num_lanes, 2))
        for i in range(num_segments):
            for j in range(num_lanes):
                waypoints_grid[i, j] = divided_points[i][j]
        return waypoints_grid

    def get_closest_border_point(self,waypoint, border_points):
        distances = np.sum((border_points - waypoint) ** 2, axis=1)
        # Find the index of the minimum distance
        closest_index = np.argmin(distances)
        # Return the closest border point
        return border_points[closest_index]


    def calculate_cost(self, waypoint, next_point, inner_border, outer_border, border_threshold, obstacles,loc):
        base_cost = np.linalg.norm(next_point - waypoint)
        cost = base_cost
        obstacle_threshold = 5
        distance_to_inner = np.linalg.norm(waypoint - inner_border)
        distance_to_outer = np.linalg.norm(waypoint - outer_border)
        if distance_to_inner < border_threshold:
            cost += (border_threshold - distance_to_inner) ** 2
        if distance_to_outer < border_threshold:
            cost += (border_threshold - distance_to_outer) ** 2



        # Enhanced turning radius cost
        if waypoint.shape == next_point.shape and len(waypoint) >= 3:
            angle_diff = np.arctan2(next_point[1] - waypoint[1], next_point[0] - waypoint[0])
            turning_radius = self.calculate_turning_radius(angle_diff)
            if turning_radius < self.L * 2:
                cost += (self.L * 2 / turning_radius - 1) * base_cost  # Scale cost based on turning radius

        # Enhanced obstacle avoidance cost
        distance_to_obstacle = self.distance_to_nearest_obstacle(loc, obstacles)
        if distance_to_obstacle < obstacle_threshold:
            cost += ((obstacle_threshold - distance_to_obstacle) / obstacle_threshold) ** 2 * base_cost

        return cost
    def distance_to_nearest_obstacle(self, loc, obstacles):

        if not obstacles:
            return float('inf')
        distances = [np.linalg.norm(np.array(loc) - np.array(obstacle)) for obstacle in obstacles]
        return min(distances)

    def dynamic_programming(self,waypoints_grid, inner_border, outer_border, border_threshold, obstacles,loc):
        num_segments, lanes, _ = waypoints_grid.shape
        optimal_costs = np.full((num_segments, lanes), float('inf'))
        optimal_previous = np.full((num_segments, lanes), -1, dtype=int)
        
        optimal_costs[0, :] = 0
        
        for segment in range(1, num_segments):
            for lane in range(lanes):
                current_waypoint = waypoints_grid[segment, lane]
                closest_inner_border_point = self.get_closest_border_point(current_waypoint, inner_border)
                closest_outer_border_point = self.get_closest_border_point(current_waypoint, outer_border)
                for previous_lane in range(lanes):
                    cost = self.calculate_cost(waypoints_grid[segment-1, previous_lane],current_waypoint,closest_inner_border_point,closest_outer_border_point,border_threshold,obstacles,loc)
                    if optimal_costs[segment-1, previous_lane] + cost < optimal_costs[segment, lane]:
                            optimal_costs[segment, lane] = optimal_costs[segment-1, previous_lane] + cost
                            optimal_previous[segment, lane] = previous_lane
        
        optimal_path_indices = []
        current_lane = np.argmin(optimal_costs[-1])
        for segment in range(num_segments - 1, -1, -1):
            optimal_path_indices.append((segment, current_lane))
            current_lane = optimal_previous[segment, current_lane]
        
        optimal_path = np.array([waypoints_grid[segment, lane] for segment, lane in reversed(optimal_path_indices)])
        
        return optimal_path


    def control(self, curr_x, curr_y, curr_vel, curr_yaw, waypoints, obstacles):
        prev_error = 0.0  
        lookahead_distance = 10.0
        kp = 0.95
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
        # if abs(math.degrees(heading_error)) > sharp_turn_threshold or curr_vel>high_speed_threshold:
        #     target_velocity = max(min_speed, curr_vel - min_speed) # Slow down if it's a sharp turn and at high speed
        #     brake = 0.65*curr_vel
        #     # brake = min(brake / 1, 1.0)
        #     # print(brake)
        # else:
        #     target_velocity = min(max_speed, curr_vel + 0.05)  
        
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
        normalized_angle_error = min(angle_error / max_angle_error, 1.0)
        target_velocity = max_speed - ((max_speed - min_speed) * normalized_angle_error)  # Decrease speed with increasing angle error
        
        distance_to_obstacle = self.distance_to_nearest_obstacle((curr_x, curr_y), obstacles)
        print(distance_to_obstacle)
        if distance_to_obstacle < 20:
            target_velocity *= (distance_to_obstacle / 20)  # Reduce speed proportionally to obstacle distance
            brake = min(0.7, 2 * (1 - (distance_to_obstacle / 20)))  # Increase brake force as obstacle gets closer



        return target_velocity, target_steering, brake
    



    
    def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary):
        current_velocity = get_speed(vel)
        curr_x = transform.location.x
        curr_y = transform.location.y
        curr_yaw = math.radians(transform.rotation.yaw)
        inner_border = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in boundary[0]])
        outer_border = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in boundary[1]])
        divisions = 6
        border_threshold = 3
        obstacles = []
        for vehicle in filtered_obstacles:
            ob = vehicle.get_location()
            obstacles.append((ob.x,ob.y))        
        loc = [curr_x, curr_y]
        divided_track = self.divide_track(inner_border, outer_border, divisions)
        waypoints_grid = self.generate_waypoints(divided_track)

        optimal_path = self.dynamic_programming(waypoints_grid, inner_border, outer_border, border_threshold,obstacles,loc)

        target_velocity, steer, brake = self.control(curr_x, curr_y, current_velocity, curr_yaw, optimal_path, obstacles)
        control = carla.VehicleControl()
        control.throttle = target_velocity
        control.steer = steer
        # print(brake)
        control.brake = brake
        # save_curr_to_csv(loc)
        # save_waypoints_to_csv(optimal_path)
        return control
