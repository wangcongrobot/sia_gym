# This class is used to communicate with sia robot.

from __future__ import print_function

import copy
import os
import sys
import threading
import time
from time import sleep
import subprocess
import numpy as np

# ROS 
import rospy
import roslaunch
import rosparam
import rostopic

import rosmsg, rosservice
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import JointState

from sia_7f_arm_train.srv import EePose, EePoseRequest, EeRpy, EeRpyRequest, EeTraj, EeTrajRequest, JointTraj, JointTrajRequest, EeDelta, EeDeltaRequest


class SIA7FARMROS(object):

    def __init__(self,
                 robot_name='sia_7f_arm',
                 use_base=True,
                 use_camera=True,
                 use_gripper=True,
                 debug_print=True,
                 ):

        # Environment variable
        self.env = os.environ.copy()

        self.debug_print = debug_print

        # run ROS core if not already running
        self.core = None # roscore
        self.gui = None # RQT
        master_uri = 11311
        self.init_core(port=master_uri)
        try:
            rospy.init_node('sia_7f_arm_ros_interface', anonymous=True)
            rospy.logwarn("ROS node sia_7f_arm_ros_interface has already been initialized")
        except rospy.exceptions.ROSException:
            rospy.logwarn('ROS node sia_7f_arm_ros_interface has not been initialized')

        ####### Arm

        # Params
        self.arm_joint_names = ["sia_7f_arm_joint1", 
                                "sia_7f_arm_joint2", 
                                "sia_7f_arm_joint3", 
                                "sia_7f_arm_joint4", 
                                "sia_7f_arm_joint5", 
                                "sia_7f_arm_gripper"]

        self.arm_joint_home = [0., 0., 0., 0., 0., 0.]

        self.arm_joint_prepare = [0., 0., 0., 0., 0., 0.]
        self.arm_dof = len(self.arm_joint_names)
        self.arm_base_frame = ''
        self.arm_ee_frame = ''

        self.arm_joint_angles = dict()
        self.arm_joint_velocities = dict()
        self.arm_joint_efforts = dict()

        self.arm_joint_state_lock = threading.RLock()

        # Topics
        self.rostopic_arm_joint_states = '/joint_states'
        self.rostopic_arm_set_joint = ''

        # Subscribers
        rospy.Subscriber(self.rostopic_arm_joint_states, JointState, self.arm_callback_joint_states)

        # Publishers
        # self.arm_joint_pub = rospy.Publisher(self.rostopic_arm_set_joint, JointState, queue_size=1)

        # Service
        self.arm_ee_traj_client = rospy.ServiceProxy('/ee_traj_srv', EeTraj)
        self.arm_joint_traj_client = rospy.ServiceProxy('/joint_traj_srv', JointTraj)
        self.arm_ee_pose_client = rospy.ServiceProxy('ee_pose_srv', EePose)
        self.arm_ee_rpy_client = rospy.ServiceProxy('/ee_rpy_srv', EeRpy)
        self.arm_ee_delta_pose_client = rospy.ServiceProxy('/ee_delta_srv', EeDelta)

        ####### Gripper
        # Topics
        # self.rostopic_gripper_cmd = ''

        # Publisher
        # self.gripper_pub_cmd


        ####### Camera RGBD

        ####### Target
        self.target_position = [0,0,0]
        rospy.Subscriber('/object_target', Pose, self.object_callback)


        ####### Initialization !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!######
        
        self.check_all_systems_ready()

        rospy.sleep(2)
        print("SIA_7F_ARM ROS Interface Initialize Successfully!")

    # Initialization
    def check_all_systems_ready(self):
        joints = None
        while joints is None and not rospy.is_shutdown():
            try:
                joints = rospy.wait_for_message(self.rostopic_arm_joint_states, JointState, timeout=1.0)
                rospy.logwarn("Current " + str(self.rostopic_arm_joint_states) + "READY=>" + str(joints))
            except:
                rospy.logerr("Current " + str(self.rostopic_arm_joint_states) + " not ready yet, retrying...")
        
        self.gripper_reset()
        self.gripper_activate()

    @staticmethod
    def is_core_running():
        """
        Return True is the ROS core is running.
        """
        try:
            rostopic.get_topic_class('/roscore')
        except rostopic.ROSTopicIOException as e:
            return False
        return True

    def init_core(self, uri='localhost', port=11311):
        """Initialize the core if it is not running.
        Form [1], "the ROS core will start up:
        - a ROS master
        - a ROS parameter server
        - a rosout logging node"
        
        Args:
            uri (str): ROS master URI. The ROS_MASTER_URI will be set to `http://<uri>:<port>/`.
            port (int): Port to run the master on.
        References:
            - [1] roscore: http://wiki.ros.org/roscore 
        """
        # if the core is not already running, run it
        if not self.is_core_running():
            self.env["ROS_MASTER_URI"] = "http://" + uri + ":" + str(port)

            # this is for the rospy methods such as: wait_for_service(), init_node(), ...
            os.environ['ROS_MASTER_URI'] = self.env['ROS_MASTER_URI']

            # run ROS core if not already running
            # if "roscore" not in [p.name() for p in ptutil.process_iter()]:
            # subprocess.Popen("roscore", env=self.env)
            self.core = subprocess.Popen(["roscore", "-p", str(port)], env=self.env, preexec_fn=os.setsid) # , shell=True)

    ### Arm 
    def arm_get_joint_name(self):
        return self.arm_joint_names

    def arm_get_ee_pose(self):
        ee_pose_req = EePoseRequest()
        ee_pose = self.arm_ee_pose_client(ee_pose_req)

        return ee_pose

    def arm_get_ee_rpy(self):
        ee_rpy_req = EeRpyRequest()
        ee_rpy = self.arm_ee_rpy_client(ee_rpy_req)

        return ee_rpy

    def arm_get_joint_angles(self):

        return self.arm_joint_angles

    def arm_get_joint_velocity(self):

        return self.arm_joint_velocities

    def arm_get_joint_torque(self, arm):

        return NotImplementedError

    def arm_set_joint_positions(self, positions):
    
        joint_positions = JointTrajRequest()
        for i in range(len(positions)):
            joint_positions.point.positions.append(positions[i])

        result = self.arm_joint_traj_client(joint_positions)

        return result.success

    def arm_set_joint_velocities(self, velocityies):
    
        return NotImplementedError

    def arm_set_joint_torques(self, torques):

        return NotImplementedError

    def arm_set_ee_pose(self, action):
        ee_target = EeTrajRequest()
   
        ee_target.pose.position.x = action[0]
        ee_target.pose.position.y = action[1]
        ee_target.pose.position.z = action[2]
        ee_target.pose.orientation.w = action[3]
        ee_target.pose.orientation.x = action[4]
        ee_target.pose.orientation.y = action[5]
        ee_target.pose.orientation.z = action[6]

        result = self.arm_ee_traj_client(ee_target)

        return True

    def arm_set_ee_pose_relative(self, action):
        ee_delta_target = EeDeltaRequest()
        ee_delta_target.pose.position.x = action[0]
        ee_delta_target.pose.position.y = action[1]
        ee_delta_target.pose.position.z = action[2]
        return self.arm_ee_delta_pose_client(ee_delta_target)

    def arm_move_ee_xyz(self, displacement, eef_step=0.005):
        """
        Keep the current orientation fixed, move the end
        effector in {xyz} directions
        """

    def arm_go_home(self, arm):
        """
        Arm to the home position
        """
        self.arm_set_joint_positions(self.arm_joint_home)
    
    def arm_move_to_neutral(self, arm):

        return NotImplementedError

    def arm_go_prepare(self, arm):
        self.arm_set_joint_positions(self.arm_joint_prepare)

    def arm_callback_joint_states(self, msg):
        """
        ROS subscriber callback for arm joint state (position, velocity)
        :param msg: Contains message published in topic
        :type msg: sensor_msgs/JointState
        """
        self.arm_joint_state_lock.acquire()
        for idx, name in enumerate(msg.name):
            if name in self.arm_joint_names:
                if idx < len(msg.position):
                    self.arm_joint_angles[name] = msg.position[idx]
                if idx < len(msg.velocity):
                    self.arm_joint_velocities[name] = msg.velocity[idx]
                if idx < len(msg.effort):
                    self.arm_joint_efforts[name] = msg.effort[idx]
        self.arm_joint_state_lock.release()
        # print("callback")

    ### Camera BB8 Stereo
    def camera_get_rgb(self):

        return NotImplementedError

    def get_object_position(self):
        if self.target_position[2] > 0:
            return self.target_position
        else:
            rospy.logerr("No Object Position, Please check the Object")
            return NotImplementedError

    def object_callback(self, msg):
        self.target_position[0] = msg.position.x
        self.target_position[1] = msg.position.y
        self.target_position[2] = msg.position.z
