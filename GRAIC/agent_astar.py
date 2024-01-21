import carla
import math
import numpy as np
import matplotlib.pyplot as plt
import csv

show_animation = False

class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


def get_speed(velocity):
    velocity = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    # print(velocity)
    return velocity

class Agent():
    def __init__(self, vehicle=None, L=2.875):
        self.vehicle = vehicle
        self.L = L  # Wheelbase of the vehicle
        # all_filtered_obstacles = []

    def longitudinal_controller(self, curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints):
        straight_speed = 0.8    
        turn_speed = 0.3
        brake = 0.0
        if len(future_unreached_waypoints) >= 2:
            first_waypoint = future_unreached_waypoints[0]
            second_waypoint = future_unreached_waypoints[1]
            angle = math.atan2(second_waypoint[1] - first_waypoint[1], second_waypoint[0] - first_waypoint[0])
            # Ensure the angle is within the range [0, 2*pi]
            angle = (angle + 2 * math.pi) % (2 * math.pi)
        else:
            angle = 0.0
        angle_error = abs(math.degrees(curr_yaw) - math.degrees(angle))
        angle_error = angle_error % 360
        if 340 > angle_error > 10: 
            target_velocity = turn_speed
            brake = 0.0
        else:
            target_velocity = straight_speed
        # print(target_velocity,math.degrees(angle),math.degrees(curr_yaw),angle_error)
        return target_velocity, brake

    def pure_pursuit_lateral_controller(self, curr_x, curr_y, curr_yaw, waypoints):
        lookahead_distance = 7.0
        lookahead_point = None

        # Find the first waypoint that is at least lookahead_distance away
        for waypoint in waypoints:
            if math.dist((curr_x, curr_y), waypoint[:2]) > lookahead_distance:
                lookahead_point = waypoint
                break

        if lookahead_point is None:
            return 0.0  # No steering if no lookahead point is found

        # Calculate the angle to the lookahead point
        angle_to_waypoint = math.atan2(lookahead_point[1] - curr_y, lookahead_point[0] - curr_x)

        # Calculate the target steering angle using the pure pursuit formula
        target_steering = np.arctan(2 * self.L * np.sin(angle_to_waypoint - curr_yaw) / lookahead_distance)
        # print(target_steering)
        return target_steering
    
    # def stanley_controller(self, curr_x, curr_y, curr_yaw, waypoints, vel):
    #     # Parameters
    #     k = 1.0  # control gain
    #     k_soft = 0.5  # softening constant
    #     k_yaw = 1.0  # yaw correction gain

    #     # Find the nearest point on the path
    #     min_dist = float('inf')
    #     nearest_point = None
    #     nearest_point_index = 0
    #     for i, waypoint in enumerate(waypoints):
    #         dist = math.sqrt((curr_x - waypoint[0])**2 + (curr_y - waypoint[1])**2)
    #         if dist < min_dist:
    #             min_dist = dist
    #             nearest_point = waypoint
    #             nearest_point_index = i

    #     # Calculate cross-track error (assuming waypoint[0] is x and waypoint[1] is y)
    #     if nearest_point is not None:
    #         cross_track_error = math.sin(math.atan2(nearest_point[1] - curr_y, nearest_point[0] - curr_x) - curr_yaw) * min_dist
    #     else:
    #         cross_track_error = 0

    #     # Calculate heading error
    #     path_direction = math.atan2(waypoints[nearest_point_index + 1][1] - waypoints[nearest_point_index][1],
    #                                 waypoints[nearest_point_index + 1][0] - waypoints[nearest_point_index][0])
    #     heading_error = path_direction - curr_yaw

    #     # Normalize heading error to [-pi, pi]
    #     heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

    #     # Calculate the steering angle
    #     steer = k_yaw * heading_error + math.atan2(k * cross_track_error, k_soft + vel)

    #     # Normalize steering angle to [-1, 1] assuming that the steering angle is within [-pi, pi]
    #     steer = max(-1.0, min(1.0, steer / math.pi))

    #     return steer





    def process_boundaries(self, boundary):
        obstacles = []
        for boundary_side in boundary:  # boundary_side is either left_boundary or right_boundary
            for waypoint in boundary_side:
                # Access the x, y coordinates of the waypoint
                x = waypoint.transform.location.x
                y = waypoint.transform.location.y
                # Create an obstacle tuple with the x, y coordinates and a predefined radius
                obstacle = (x, y) 
                obstacles.append(obstacle)
        return obstacles
    
    def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary):
        
        # save_waypoints_to_csv(waypoints)
        # save_boundary_to_csv(boundary)

        # Get the current velocity in m/s
        # print(vel)
        current_velocity = get_speed(vel)
        # print(current_velocity)
        # Determine the target velocity
        # target_velocity = self.longitudinal_controller(current_velocity, target_velocity)

        obstacle_list = self.process_boundaries(boundary)

        # print(obstacle_list)
        # Get the current position and orientation
        curr_x = transform.location.x
        curr_y = transform.location.y
        curr_yaw = math.radians(transform.rotation.yaw)
        gx, gy = waypoints[7][0], waypoints[7][1]
        ox, oy = zip(*obstacle_list) if obstacle_list else ([], [])
        grid_size = 1.0
        # Create an instance of the AStarPlanner with the current obstacle list
        cs = 1.0
        a_star = AStarPlanner(ox, oy, grid_size , cs)

        # Get the path from the current position to the goal
        path_x, path_y = a_star.planning(curr_x, curr_y, gx, gy)
        # print(path_x,path_y)
        # Use the A* path as the waypoints
        a_star_waypoints = list(zip(path_x, path_y))

        # print(a_star_waypoints)
        # Calculate the steering angle
        target_velocity, brake= self.longitudinal_controller(curr_x, curr_y, current_velocity, curr_yaw, waypoints)
        # steer = self.stanley_controller(curr_x, curr_y, curr_yaw, waypoints, current_velocity)
        steer = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, a_star_waypoints)
        # steer = np.clip(steer,-1.0,1.0)

        # print("STEEER:",steer)
        # Create the control command
        control = carla.VehicleControl()
        control.throttle = target_velocity
        # control.throttle = 0.0
        control.steer = steer
        control.brake = 0.0  # Set the brake to 0 for now

        return control
