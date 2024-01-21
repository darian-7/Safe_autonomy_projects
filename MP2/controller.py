import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateResponse
from gazebo_msgs.msg import ModelState
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from std_msgs.msg import Float32MultiArray
import math
from util import euler_to_quaternion, quaternion_to_euler
import time
from waypoint_list import WayPoints

class vehicleController():

    def __init__(self):
        # Publisher to publish the control input to the vehicle model
        self.controlPub = rospy.Publisher("/ackermann_cmd", AckermannDrive, queue_size = 1)
        self.prev_vel = 0
        self.L = 1.75 # Wheelbase, can be get from gem_control.py
        self.log_acceleration = False

    def getModelState(self):
        # Get the current state of the vehicle
        # Input: None
        # Output: ModelState, the state of the vehicle, contain the
        #   position, orientation, linear velocity, angular velocity
        #   of the vehicle
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp = serviceResponse(model_name='gem')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
            resp = GetModelStateResponse()
            resp.success = False
        return resp


    # Tasks 1: Read the documentation https://docs.ros.org/en/fuerte/api/gazebo/html/msg/ModelState.html
    #       and extract yaw, velocity, vehicle_position_x, vehicle_position_y
    # Hint: you may use the the helper function(quaternion_to_euler()) we provide to convert from quaternion to euler
    def extract_vehicle_info(self, currentPose):

        ####################### TODO: Your TASK 1 code starts Here #######################
        pos_x, pos_y, vel, yaw = 0, 0, 0, 0

        msg = self.getModelState()
        #print("pos is ", msg )
        pos_x = msg.pose.position.x
        pos_y = msg.pose.position.y
        vel = msg.twist
        quaternion = msg.pose.orientation
        euler = quaternion_to_euler(quaternion.x,quaternion.y, quaternion.z, quaternion.w)
        yaw = euler[2]
        #print("yaw is", (yaw*180)/np.pi)
        

        

        ####################### TODO: Your Task 1 code ends Here #######################

        return pos_x, pos_y, vel, yaw # note that yaw is in radian

    # Task 2: Longtitudal Controller
    # Based on all unreached waypoints, and your current vehicle state, decide your velocity
    def longititudal_controller(self, curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints):

        ####################### TODO: Your TASK 2 code starts Here #######################
        
        if len(future_unreached_waypoints)>1:
            t1_x, t1_y = future_unreached_waypoints[0]
            t2_x, t2_y = future_unreached_waypoints[1]
        else:
            t1_x, t1_y = future_unreached_waypoints[0]
            t2_x, t2_y = future_unreached_waypoints[0]   
           
        print("differnce x1 is ",t1_x-curr_x , "diff y is", t1_y-curr_y)
        print("differnce x1 is ",t2_x-curr_x , "diff y is", t2_y-curr_y)
        if ((((abs(t1_x-curr_x)<0.25) and abs(t2_x-curr_x)<0.25)) or ((abs(t1_y-curr_y)<0.25) and abs(t2_y-curr_y)<0.25)):
            target_velocity = 16  #11
            print("straight")
        else:
            target_velocity = 9  #8
            print("curve")   
     

            ####################### TODO: Your TASK 2 code ends Here #######################
            return target_velocity
      

    # Task 3: Lateral Controller (Pure Pursuit)
    def pure_pursuit_lateral_controller(self, curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints):

        ####################### TODO: Your TASK 3 code starts Here #######################
        if len(future_unreached_waypoints)>2:
            p1 = future_unreached_waypoints[0]
            p2 = future_unreached_waypoints[1]
            if p1[0]-p2[0]>18 or p1[1]-p2[1]>18:
                print ("way point to farrrrr")
                target_point = p1 
            else:
                target_point = p2
        else:
          target_point = future_unreached_waypoints[0]
        curr_point = np.array([curr_x, curr_y])         
        ld = math.sqrt((target_point[0]-curr_x)**2 + (target_point[1]-curr_y)**2)
        alpha = np.arctan2((target_point[1]-curr_point[1]),(target_point[0]- curr_point[0])) - curr_yaw
        delta = (np.arctan2((2*1.75*np.sin(alpha)),(ld)))
        if delta>1:
            delta =1
        if delta <-1:
            delta =-1

        print("alpha is ", (alpha*180)/np.pi)
        print("delta is ", (delta*180)/np.pi)
        print("curr_yaw is ", curr_yaw*180/np.pi)
        print("curernt xy", curr_point)
        print("target xy", target_point)
        target_steering = delta
            
        return target_steering


    def execute(self, currentPose, target_point, future_unreached_waypoints):
        # Compute the control input to the vehicle according to the
        # current and reference pose of the vehicle
        # Input:
        #   currentPose: ModelState, the current state of the vehicle
        #   target_point: [target_x, target_y]
        #   future_unreached_waypoints: a list of future waypoints[[target_x, target_y]]
        # Output: None

        curr_x, curr_y, curr_vel, curr_yaw = self.extract_vehicle_info(currentPose)

        # Acceleration Profile
        if self.log_acceleration:
            acceleration = (curr_vel- self.prev_vel) * 100 # Since we are running in 100Hz



        target_velocity = self.longititudal_controller(curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints)
        target_steering = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints)


        #Pack computed velocity and steering angle into Ackermann command
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = target_velocity
        newAckermannCmd.steering_angle = target_steering

        # Publish the computed control input to vehicle model
        self.controlPub.publish(newAckermannCmd)

    def stop(self):
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = 0
        self.controlPub.publish(newAckermannCmd)