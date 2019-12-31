
import rospy

from sia_7f_arm_ros import SIA7FARMROS

sia_7f_arm_ros = SIA7FARMROS()

def test_arm():
    
    joint_names = sia_7f_arm_ros.arm_get_joint_name()

    print(joint_names)

    joint_values = sia_7f_arm_ros.arm_get_joint_angles()
    print(joint_values)

    joint_target = [0., 
                    0., 
                    0., 
                    0., 
                    0., 
                    0.]
    # husky_ur_ros.arm_set_joint_positions(joint_target)

    ee_pose = sia_7f_arm_ros.arm_get_ee_pose()
    print(ee_pose)
    action = [0.02,0.02,0.02]
    sia_7f_arm_ros.arm_set_ee_pose_relative(action)
    return 0
    '''
    pose: 
  position: 
    x: 0.1858453113077738
    y: 0.19050837742130322
    z: 0.6712833076182397
  orientation: 
    x: -0.6269294640671255
    y: 0.21130153236837768
    z: -0.6742962500890209
    w: 0.32807876587668194
    '''
    # ee_pose.pose.position.z += 0.1

    action = [ee_pose.pose.position.x, 
              ee_pose.pose.position.y, 
              ee_pose.pose.position.z + 0.05, 
              ee_pose.pose.orientation.w,
              ee_pose.pose.orientation.x,
              ee_pose.pose.orientation.y,
              ee_pose.pose.orientation.z]
    action = [0.3169, -0.41, 0.93, 0.924, 0, 0, 0.383]
    sia_7f_arm_ros.arm_set_ee_pose(action)

def test_gripper():
    sia_7f_arm_ros.gripper_open()
    rospy.sleep(1.0)
    sia_7f_arm_ros.gripper_open()


def test_base():
    pass


def test_camera():
    pass



if __name__ == "__main__":
    test_arm()
    # test_gripper()
    # test_base()
