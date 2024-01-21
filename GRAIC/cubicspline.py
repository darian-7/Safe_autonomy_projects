import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.special import comb

import re
from scipy.interpolate import UnivariateSpline
from scipy.special import fresnel




def divide_track(inner_border, outer_border, divisions):
    divided_points = []
    for inner, outer in zip(inner_border, outer_border):
        segment = [inner + (outer - inner) * i / (divisions - 1) for i in range(divisions)]
        divided_points.append(segment)
    return np.array(divided_points)  # Shape will be (n, divisions, 2)

def generate_waypoints(divided_points):
    num_segments = len(divided_points)
    # Assuming each segment has the same number of lanes
    num_lanes = len(divided_points[0])
    # Create a 2D grid of waypoints with shape (num_segments, num_lanes, 2)
    waypoints_grid = np.zeros((num_segments, num_lanes, 2))
    for i in range(num_segments):
        for j in range(num_lanes):
            waypoints_grid[i, j] = divided_points[i][j]
    return waypoints_grid

def get_closest_border_point(waypoint, border_points):
    """
    Find the point on the border that is closest to the given waypoint.

    :param waypoint: A numpy array representing the (x, y) coordinates of the waypoint.
    :param border_points: A numpy array of shape (n, 2), representing the (x, y) coordinates of the border points.
    :return: A numpy array representing the (x, y) coordinates of the closest border point.
    """
    # Calculate the squared Euclidean distance from the waypoint to each border point
    distances = np.sum((border_points - waypoint) ** 2, axis=1)
    # Find the index of the minimum distance
    closest_index = np.argmin(distances)
    # Return the closest border point
    return border_points[closest_index]


def calculate_cost(waypoint, next_point, inner_border, outer_border, border_threshold):
    # Basic cost based on Euclidean distance
    cost = np.sum((next_point - waypoint) ** 2)
    
    # Calculate the distance to the closest border for penalty
    distance_to_inner = np.linalg.norm(waypoint - inner_border)
    distance_to_outer = np.linalg.norm(waypoint - outer_border)

    # Apply a penalty if the waypoint is too close to either border
    border_penalty = 1000  # This value should be tuned to your specific needs
    if distance_to_inner < border_threshold or distance_to_outer < border_threshold:
        cost += border_penalty
    
    return cost

def dynamic_programming(waypoints_grid, inner_border, outer_border, border_threshold):
    num_segments, lanes, _ = waypoints_grid.shape
    optimal_costs = np.full((num_segments, lanes), float('inf'))
    optimal_previous = np.full((num_segments, lanes), -1, dtype=int)
    
    # Initialize the starting line costs
    optimal_costs[0, :] = 0
    
    # Compute the costs for each segment
    for segment in range(1, num_segments):
        for lane in range(lanes):
            current_waypoint = waypoints_grid[segment, lane]
            closest_inner_border_point = get_closest_border_point(current_waypoint, inner_border)
            closest_outer_border_point = get_closest_border_point(current_waypoint, outer_border)
            for previous_lane in range(lanes):
                cost = calculate_cost(waypoints_grid[segment-1, previous_lane],current_waypoint,closest_inner_border_point,closest_outer_border_point,border_threshold)
                if optimal_costs[segment-1, previous_lane] + cost < optimal_costs[segment, lane]:
                        optimal_costs[segment, lane] = optimal_costs[segment-1, previous_lane] + cost
                        optimal_previous[segment, lane] = previous_lane
    
    # Backtrack to find the optimal path
    optimal_path_indices = []
    current_lane = np.argmin(optimal_costs[-1])
    for segment in range(num_segments - 1, -1, -1):
        optimal_path_indices.append((segment, current_lane))
        current_lane = optimal_previous[segment, current_lane]
    
    # Extract the actual (x, y) points for the optimal path
    optimal_path = np.array([waypoints_grid[segment, lane] for segment, lane in reversed(optimal_path_indices)])
    
    return optimal_path



# LAST WORKING
def smooth_path(inner_border, outer_border):
    # inner_border = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in inner_border])
    # outer_border = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in outer_border])
    racing_line_weight=0.2
    num_points = len(inner_border)

    # Fit a cubic spline for inner and outer borders
    t = np.linspace(0, 1, num_points)  # Normalize parameter t
    cs_inner_x = interp1d(t, inner_border[:, 0], kind='cubic')
    cs_inner_y = interp1d(t, inner_border[:, 1], kind='cubic')
    cs_outer_x = interp1d(t, outer_border[:, 0], kind='cubic')
    cs_outer_y = interp1d(t, outer_border[:, 1], kind='cubic')

    # Generate more points for a smoother path
    num_interpolated_points = 2000  # Increase this number for more resolution
    t_smooth = np.linspace(0, 1, num_interpolated_points)

    # Interpolate inner and outer borders to get smoother racing line
    inner_smooth_x = cs_inner_x(t_smooth)
    inner_smooth_y = cs_inner_y(t_smooth)
    outer_smooth_x = cs_outer_x(t_smooth)
    outer_smooth_y = cs_outer_y(t_smooth)

    # Calculate racing line by combining inner and outer borders
    racing_line_x = racing_line_weight * inner_smooth_x + (1 - racing_line_weight) * outer_smooth_x
    racing_line_y = racing_line_weight * inner_smooth_y + (1 - racing_line_weight) * outer_smooth_y

    return np.column_stack((racing_line_x, racing_line_y))

def generate_smooth_path1(inner_border, outer_border):
    # inner_border = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in inner_border])
    # outer_border = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in outer_border])


    
    
    midpoints = (inner_border + outer_border) / 2
    num_points = len(midpoints)

    # Fit a cubic spline
    t = np.linspace(0, 1, num_points)  # Normalize parameter t
    cs_x = interp1d(t, midpoints[:, 0], kind='cubic')
    cs_y = interp1d(t, midpoints[:, 1], kind='cubic')

    # Generate more points for a smoother path
    num_interpolated_points = 2000  # Increase this number for more resolution
    t_smooth = np.linspace(0, 1, num_interpolated_points)
    x_smooth = cs_x(t_smooth)
    y_smooth = cs_y(t_smooth)

    # Apply Gaussian smoothing
    sigma = 15  # Adjust the sigma value as needed
    x_smooth = gaussian_filter1d(x_smooth, sigma)
    y_smooth = gaussian_filter1d(y_smooth, sigma)

    # Clipping points to stay within track limits (optional, based on your track requirements)
    min_x, max_x = np.min([inner_border[:, 0].min(), outer_border[:, 0].min()]), np.max([inner_border[:, 0].max(), outer_border[:, 0].max()])
    min_y, max_y = np.min([inner_border[:, 1].min(), outer_border[:, 1].min()]), np.max([inner_border[:, 1].max(), outer_border[:, 1].max()])
    # print(min_x, max_x)
    x_smooth = np.clip(x_smooth, min_x, max_x)
    y_smooth = np.clip(y_smooth, min_y, max_y)

    return np.column_stack((x_smooth, y_smooth))

def find_optimal_racing_line(inner_coords, outer_coords):
    # Calculate the midline of the track
    # inner_coords = inner_coords[75:225]
    # outer_coords = outer_coords[75:225]
    midline = (inner_coords + outer_coords) / 2

    curvature = calculate_curvature(midline)
    optimal_line = np.copy(midline)

    for i in range(len(optimal_line)):
        # Dynamic factor based on curvature
        curvature_factor = np.clip(np.abs(curvature[i]), 0.1, 10)
        direction = np.array([-midline[i, 1], midline[i, 0]])
        if curvature[i] < 0:
            direction = np.array([-midline[i, 1], midline[i, 0]])
        elif curvature[i] > 0:
            direction = np.array([midline[i, 1], -midline[i, 0]])

        direction /= np.linalg.norm(direction)
        movement = direction * curvature_factor * 20 # Adjust 0.5 to a suitable scaling factor
        new_point = midline[i] + movement

        # Ensure the new point is within track limits
        if np.linalg.norm(new_point - inner_coords[i]) < np.linalg.norm(inner_coords[i] - outer_coords[i]) and \
            np.linalg.norm(new_point - outer_coords[i]) < np.linalg.norm(inner_coords[i] - outer_coords[i]):
            optimal_line[i] = new_point

    # Smoothing the line
    x_spline = UnivariateSpline(range(len(optimal_line[:, 0])), optimal_line[:, 0], s=1.5)
    y_spline = UnivariateSpline(range(len(optimal_line[:, 1])), optimal_line[:, 1], s=1.5)
    for i in range(len(optimal_line)):
        optimal_line[i, 0] = x_spline(i)
        optimal_line[i, 1] = y_spline(i)

    return optimal_line

def calculate_curvature(path):
    # Calculate the curvature of a path
    dx_dt = np.gradient(path[:, 0])
    dy_dt = np.gradient(path[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / ((dx_dt ** 2 + dy_dt ** 2) ** 1.5)
    return curvature

def path_len_traj(inner_border: np.ndarray, outer_border: np.ndarray, follow_distance, num_points):
    midpoints = (inner_border + outer_border) / 2
    midpoints = np.array(midpoints)
    x = midpoints[:, 0]
    y = midpoints[:, 1]

    # Calculate the length of the path
    path_length = np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

    # Calculate the distance step
    distance_step = path_length / num_points

    # Initialize the trajectory with the starting point
    trajectory = np.array([[x[0], y[0]]])

    # Initialize the current position and heading angle
    current_x, current_y = x[0], y[0]
    current_heading = 0

    # Generate points along the trajectory
    for _ in range(num_points - 1):
        # Find the point on the reference path that is the follow_distance ahead
        print(_)
        target_distance = follow_distance
        while True:
            target_x = current_x + target_distance * np.cos(current_heading)
            target_y = current_y + target_distance * np.sin(current_heading)
            closest_point_index = np.argmin((x - target_x) ** 2 + (y - target_y) ** 2)
            closest_point_distance = np.sqrt((x[closest_point_index] - target_x) ** 2 + (y[closest_point_index] - target_y) ** 2)
            if closest_point_distance < follow_distance * 0.1:  # Adjust the tolerance as needed
                break
            target_distance += follow_distance * 0.1  # Move closer to the reference path

        current_x = x[closest_point_index]
        current_y = y[closest_point_index]

        trajectory = np.vstack((trajectory, [current_x, current_y]))

    return trajectory





# Define your boundary points
boundary_points = []
file_path = 'boundary2.txt'
with open(file_path, 'r') as file:
# Read each line in the file one by one.
    for line in file:
        # Strip whitespace and then split the line into two parts at the comma.
        # This assumes that there is no space after the comma.
        left_part, right_part = line.strip().split('),(')
        
        # Remove any remaining parentheses.
        left_part = left_part.strip('(')
        right_part = right_part.strip(')')
        
        # Now split by comma to separate the values and convert them to integers or floats as needed.
        left_x, left_y = map(float, left_part.split(','))
        right_x, right_y = map(float, right_part.split(','))
        size = 0.5
        # Add the tuple to the list. It assumes size1 and size2 are the same, as per your structure.
        # If size1 and size2 are different, adjust accordingly.
        boundary_points.append(((left_x, left_y)))
        boundary_points.append((right_x, right_y))

dppath = []
file_path = 'curr_dp.txt'
with open(file_path, 'r') as file:
# Read each line in the file one by one.
    for line in file:
        left_x, left_y = map(float, line.split(','))
        dppath.append(((left_x, left_y)))

curr = []
file_path = 'curr.txt'
with open(file_path, 'r') as file:
# Read each line in the file one by one.
    for line in file:
        left_x, left_y = map(float, line.split(','))
        curr.append(((left_x, left_y)))


hyb = []
file_path = 'curr_hybrid.txt'
with open(file_path, 'r') as file:
# Read each line in the file one by one.
    for line in file:
        left_x, left_y = map(float, line.split(','))
        hyb.append(((left_x, left_y)))


# print(path)

astar = []  # Initialize an empty list to store the data
file_path = 'a_star.txt'



with open(file_path, 'r') as file:
    points_text = file.read()

# Extracting the points
astar_points = re.findall(r"\((\d+\.?\d*), (\d+\.?\d*)\)", points_text)
astar_points = [(float(x), float(y)) for x, y in astar_points]
astar_points = np.array(astar_points)
# Separating the x and y coordinates
# x, y = zip(*points)


# with open(file_path, 'r') as file:
#     # Read the entire file as a string
#     data = file.read()
    
#     # Split the string into individual data points using ','
#     data_points = data.split(',')
    
#     # Iterate through the data points
#     for data_point_str in data_points:
#         # Remove parentheses and convert the values to floats
#         data_point_str = data_point_str.strip('()')
#         values = data_point_str.split(',')
#         x = values[0]
#         y = values[1]
#         astar.append((x, y))





# Generate the smooth path
boundary_points = np.array(boundary_points)
dppath = np.array(dppath)
curr_arr = np.array(curr)

hyb = np.array(hyb)
inner_border = boundary_points[::2]
outer_border = boundary_points[1::2]
inner_border = inner_border[0:]
outer_border = outer_border[0:]
midpoints = (inner_border + outer_border) / 2
# smooth_path = find_optimal_racing_line(inner_border, outer_border)

# Initialize a racing line (for simplicity, here it's just the midpoints)
racing_line = np.copy(midpoints)

# Find the optimal path

# optimal_path = waypoints[optimal_path_indices]
# divisions = 5
# border_threshold = 1  # This should be set according to the minimum allowed distance to the border

# # Divide the track and generate the waypoints grid
# divided_track = divide_track(inner_border, outer_border, divisions)
# waypoints_grid = generate_waypoints(divided_track)

# # Find the optimal path
# optimal_path = dynamic_programming(waypoints_grid, inner_border, outer_border, border_threshold)
# # optimal_path = midpoints

# track_width = waypoints_grid.shape[1]  # Number of lanes

# Plotting the track boundaries
plt.scatter(inner_border[:, 0], inner_border[:, 1], c='grey', marker='o', label='Inner Border')
plt.scatter(outer_border[:, 0], outer_border[:, 1], c='grey', marker='o', label='Outer Border')

# Plotting the midpoints as a midline
# plt.plot(midpoints[:, 0], midpoints[:, 1], 'k--', label='Midline')

# Plotting the waypoints
# for i in range(waypoints_grid.shape[0]):
#     for j in range(track_width):
#         plt.scatter(waypoints_grid[i, j, 0], waypoints_grid[i, j, 1], c='blue', s=10,)  # s is the size of points

# Plotting the optimal path
plt.plot(dppath[:, 0], dppath[:, 1], 'g-', label='Dynamic Programming Path Score 95.5')
plt.plot(curr_arr[:, 0], curr_arr[:,1], 'b-',label='Spline Interpolation Star Path Score 94.4')
plt.plot(hyb[:, 0], hyb[:, 1], 'r-',label='Hybrid A Star Path Score 94.2')

# plt.plot(curr_arr[:, 0], curr_arr[:, 1], 'g-')
# Setting up the legend and titles
plt.legend()
plt.title('Optimized Racing Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')  # Ensure that the scale is the same on both axes

# Show the plot
plt.show()






# spline = generate_smooth_path(inner_border, outer_border)
# follow_distance = 2.0 
# num_points = len(inner_border)

# print(num_points)
# path = path_len_traj(inner_border, outer_border, follow_distance,num_points)
# The variable 'smooth_path' now contains the points of the optimal path
# print(smooth_path)

# x_coords, y_coords = zip(*path)
# print(x_coords)
# # print(midpoints.shape,path.shape)

# astar_arr  = np.array(astar)
# print(astar)
# # Now use plt.plot with these lists
# plt.plot(astar_arr[0], astar_arr[1], 'b-')
# print(spline2)
# # plt.plot(path_array[:, 0], path_array[:,1], 'm-')
# plt.plot(spline2[:, 0], spline2[:, 1], 'b-')
# plt.plot(midpoints[:, 0], midpoints[:, 1],color='black', linestyle='-')
# # plt.plot(curr_arr[:, 0], curr_arr[:, 1], 'r-')
# plt.scatter([p[0] for p in boundary_points], [p[1] for p in boundary_points], color='black',linestyle='-')

# plt.plot(midpoints[:, 0], midpoints[:, 1], 'k--', label='Midline')
# plt.plot(optimal_path[:, 0], optimal_path[:, 1], 'r', label='Optimal Path')
# plt.scatter(waypoints[:, 0], waypoints[:, 1], c='b', label='Waypoints')
# plt.legend()
# plt.title('Optimized Racing Line')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.axis('equal')  # This will ensure that the scale is the same on both axes
# plt.show()
