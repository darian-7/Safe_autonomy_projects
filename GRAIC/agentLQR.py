import numpy as np
import scipy.linalg
import carla
import math

# Define the vehicle model and LQR parameters
def get_speed(velocity):
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

class Agent():
    def __init__(self, vehicle=None, L= 2.875):
        self.vehicle = vehicle
        self.L = L  # Wheelbase of the vehicle
        self.current_index = 0  

    def calculate_desired_state(self, waypoints, current_index, look_ahead=1):
        """
        Calculate the desired velocity and yaw based on the waypoints.
        The look_ahead parameter determines how many waypoints ahead we look to calculate the desired state.
        """
        # Ensure we have enough waypoints
        if current_index + look_ahead >= len(waypoints):
            look_ahead = len(waypoints) - current_index - 1

        # Get the current waypoint and the waypoint 'look_ahead' steps ahead
        current_waypoint = waypoints[current_index]
        # print(current_index)
        future_waypoint = waypoints[current_index + look_ahead]

        # Calculate desired yaw (orientation) as the angle between the two waypoints
        desired_yaw = math.atan2(
            future_waypoint[1] - current_waypoint[1],
            future_waypoint[0] - current_waypoint[0]
        )

        # Define the desired velocity
        # This could be a fixed value or could change based on road conditions, traffic, etc.
        desired_velocity = 10  # Let's say we want to drive at 10 m/s

        return desired_velocity, desired_yaw

    # Usage in the run_step method

    def lqr_controller(self, A, B, Q, R, x, x_ref):
        """
        Compute the LQR controller gain and control input.
        """
        # Solve Riccati equation
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)

        # Compute the LQR gain
        K = np.linalg.inv(R) @ B.T @ P

        # Compute the control input
        u = -K @ (x - x_ref)

        return u

    def update_current_index(self, waypoints, current_position):
        """
        Update the current index based on the vehicle's current position.
        If the vehicle has passed a waypoint, the index is incremented.
        """
        # Loop through the waypoints starting from the current index
        for i in range(self.current_index, len(waypoints)):
            # Calculate the distance from the current position to the waypoint
            waypoint = waypoints[i]
            distance = math.sqrt((waypoint[0] - current_position[0])**2 + (waypoint[1] - current_position[1])**2)
            
            # If the distance to the next waypoint is greater than the distance to the current,
            # it means we have passed the current waypoint and can move to the next
            if i + 1 < len(waypoints):
                next_waypoint = waypoints[i + 1]
                next_distance = math.sqrt((next_waypoint[0] - current_position[0])**2 + (next_waypoint[1] - current_position[1])**2)
                if next_distance > distance:
                    continue
            # Update the current index to the waypoint we have reached
            self.current_index = i
            break
        
        return self.current_index


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
        # Get the current velocity in m/s
        current_velocity = get_speed(vel)

        # Get the current position and orientation
        curr_x = transform.location.x
        curr_y = transform.location.y
        curr_yaw = math.radians(transform.rotation.yaw)
        # dt = 0.0167
        # Define your system dynamics matrices (A and B) and cost matrices (Q and R)
        dt = 0.05  # Time step

        # Linearized state-space matrices around the operating point (straight driving at constant velocity)
        A = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

        B = np.array([[0, 0],
                    [0, 0],
                    [1, 0],
                    [0, dt]])


        # State cost matrix (Q)
        Q = np.diag([0.1, 0.1, 0.1, 0.1])  # Adjust these values based on how much you want to penalize each state deviation

        # Control input cost matrix (R)
        R = np.diag([0.01, 0.01])  # This penalizes large control inputs (throttle/steering)
        print(Q,R)
        # Tune weights
        # Q[0, 0] = 5  # Increase weight for the first state variable
        # R[0, 0] = 0.1  # Decrease weight for the control input
        # Define your current state and reference state
        # Usage
        current_position = (curr_x, curr_y)  # Get the current position of the vehicle
        self.current_index = self.update_current_index(waypoints, current_position)
        desired_velocity, desired_yaw = self.calculate_desired_state(waypoints, self.current_index)

        x = np.array([curr_x, curr_y, current_velocity, curr_yaw])
        x_ref = np.array([waypoints[-1][0], waypoints[-1][1], desired_velocity, desired_yaw])

        # Compute the control input using LQR
        u = self.lqr_controller(A, B, Q, R, x, x_ref)

        # Extract the control inputs (e.g., steering and throttle)
        steer = u[0]
        throttle = u[1]
        # print(u)
        # Create the control command
        throttle = np.clip(u[0], 0, 1)  # Assuming throttle ranges from 0 to 1
        steer = np.clip(u[1], -1, 1)    
        print(u,throttle,steer)
        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steer
        control.brake = 0.0  # Set the brake to 0 for now

        # Return the control command
        return control

