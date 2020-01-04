import numpy as np

from gym.envs.robotics import robot_gym_env, utils_sia
from gym.envs.robotics import rotations, utils



def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class MobileSIA7FARMGymEnv(robot_gym_env.RobotGymEnv):
    """Superclass for environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold, initial_qpos, reward_type, n_actions,
        use_real_robot, debug_print
    ):
        """Initializes a new environment.
        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            n_actions : the number of actuator
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.n_actions = n_actions

        self.arm_dof = 3
        self.gripper_dof = 1
        # self.n_actions = self.arm_dof + self.gripper_dof
        
        self.gripper_actual_dof = 6
        self.gripper_close = False

        #self.sia_init_pos = [0,0]

        self.arm_joint_names = ["sia_7f_arm_joint1", 
                          "sia_7f_arm_joint2", 
                          "sia_7f_arm_joint3", 
                          "sia_7f_arm_joint4", 
                          "sia_7f_arm_joint5", 
                          "sia_7f_arm_gripper"]

        self.init_pos = {
            'sia_7f_arm_joint1': 0.0,
            'sia_7f_arm_joint2': 0.0,
            'sia_7f_arm_joint3': 0.0,
            'sia_7f_arm_joint4': 0.0,
            'sia_7f_arm_joint5': 0.0,
            'sia_7f_arm_gripper': 0.0,
        }

        self.debug_print = debug_print

        self._is_success = 0

        self._use_real_robot = use_real_robot
        if self._use_real_robot:
            # import rospy
            from gym.envs.robotics.ros_interface import sia_7f_arm_ros
            self.sia_7f_arm_robot = sia_7f_arm_ros.SIA7FARMROS(debug_print=debug_print)

        super(MobileSIA7FARMGymEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=self.n_actions,
            initial_qpos=initial_qpos)

    # GoalEnv methods
    # ----------------------------
    

    def compute_reward(self, action, goal):
        return self.reward_pick(action, goal)

    def reward_pick(self, action, goal):
        """
        Simple reward function: reach and pick
        """
        object_pos = self.sim.data.get_site_xpos('object0')
        if self.debug_print:
            print("self.sim.data.get_site_xpos('object0'): ", object_pos)
        grip_pos = self.sim.data.get_site_xpos('r_grip_site')
        if self.debug_print:
            print("self.sim.data.get_site_xpos('grip_pos'): ", grip_pos)

        grip_obj_pos = object_pos - grip_pos
        obj_target_pos = goal - object_pos

        reward_ctrl = 0
        reward_dist_object = 0
        reward_grasping = 0
        reward_dist_target = 0
        reward_target = 0
        reward = 0
        self._is_success = 0

        reward_ctrl = -np.square(action).sum()

        reward_dist_object = -np.linalg.norm(grip_obj_pos)
        if self.debug_print:
            print("distance between gripper and object: ", reward_dist_object)
        reward_dist_target = -np.linalg.norm(obj_target_pos)

        self.gripper_close = False
        if np.linalg.norm(grip_obj_pos) < 0.1:
            # reward_grasping += 0.5
            if np.linalg.norm(grip_obj_pos) < 0.05:
                # reward_grasping += 1.0
                self.gripper_close = True
                # stage 1: approaching and grasping/lifting
                if object_pos[2] > 0.75: # table hight + object hight + lift distance
                    # grasping success
                    reward_grasping += 10.0
                    if object_pos[2] > 0.8:
                        reward_grasping += 100.0
                        self._is_success = 1
                        # if object_pos[2] > 0.5:
                            # reward_grasping += 10.0
        reward = 0.01 * reward_ctrl + reward_dist_object + reward_grasping
            
        # reward = 0.05 * reward_ctrl + reward_dist_object + reward_grasping + 10 * reward_dist_target + reward_target
        if self.debug_print:
            print("object_pose: ", object_pos)
            print("reward_dist_object: ", reward_dist_object)
            print("reward_ctrl: ", 0.05 * reward_ctrl)
            print("reward_grasping: ", reward_grasping)
            # print("reward_dist_target: ", reward_dist_target)
            # print("reward_target: ", reward_target)
            print("total reward: ", reward)
        done = False
        if object_pos[2] < 0.1:
            # done = True
            reward -= 10
        info = {
            'is_success': self._is_success,
        }
        return reward, done, info

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            # self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            # self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        # For real robot
        if self._use_real_robot:
            assert action.shape == (self.n_actions,) # 6 mobile base
            action = action.copy()  # ensure that we don't change the action outside of this scope
            if self.debug_print:
                print("_set_action:", action)

            pos_ctrl, gripper_ctrl = action[:3], action[3:]

            pos_ctrl *= 0.03  # limit maximum change in position
            rot_ctrl = [0.5, 0.5, -0.5, -0.5] # fixed rotation of the end effector, expressed as a quaternion

            if self.gripper_close:
                gripper_ctrl = 1.0
            else:
                gripper_ctrl = -1.0
            gripper_ctrl = self.gripper_format_action(gripper_ctrl)
            assert gripper_ctrl.shape == (self.gripper_actual_dof,)
            if self.block_gripper:
                gripper_ctrl = np.zeros_like(gripper_ctrl)
            # action = np.concatenate([pos_ctrl, rot_ctrl, base_ctrl, gripper_ctrl])
            action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

            ee_pose = self.sia_7f_arm_robot.arm_get_ee_pose()

            arm_action = pos_ctrl
            print("arm_action: ", arm_action)

            # Applay action to real robot
            self.sia_7f_arm_robot.arm_set_ee_pose_relative(arm_action)
            if self.gripper_close:
                self.sia_7f_arm_robot.gripper_close()
            else:
                self.sia_7f_arm_robot.gripper_open()
        # For simulation
        else:   
            assert action.shape == (self.n_actions,) # 6 mobile base
            action = action.copy()  # ensure that we don't change the action outside of this scope
            if self.debug_print:
                print("_set_action:", action)
            pos_ctrl, gripper_ctrl = action[:3], action[3:]

            pos_ctrl *= 0.03  # limit maximum change in position
            rot_ctrl = [0.5, 0.5, -0.5, -0.5] # fixed rotation of the end effector, expressed as a quaternion

            if self.gripper_close:
                gripper_ctrl = 1.0
            else:
                gripper_ctrl = -1.0
            gripper_ctrl = self.gripper_format_action(gripper_ctrl)
            # assert gripper_ctrl.shape == (2,)
            assert gripper_ctrl.shape == (self.gripper_actual_dof,)
            if self.block_gripper:
                gripper_ctrl = np.zeros_like(gripper_ctrl)
            action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

            # Apply action to simulation.
            utils.ctrl_set_action(self.sim, action) # base control + gripper control
            utils.mocap_set_action(self.sim, action) # arm control in cartesion (x, y, z)

    def _get_obs(self):
        # For real robot
        if self._use_real_robot:
            joint_angles = []
            joint_velocity = []
            ee_position = []
            ee_orientation = []

            joint_names_dict = self.sia_7f_arm_robot.arm_get_joint_angles()
            joint_velocity_dict = self.sia_7f_arm_robot.arm_get_joint_velocity()
            for i in self.arm_joint_names:
                joint_angles.append(joint_names_dict[i])
                joint_velocity.append(joint_velocity_dict[i])
            ee_pose = self.sia_7f_arm_robot.arm_get_ee_pose()
            ee_position = [ee_pose.pose.position.x, 
                           ee_pose.pose.position.y,
                           ee_pose.pose.position.z]
            ee_orientation = [ee_pose.pose.orientation.w,                
                              ee_pose.pose.orientation.x,
                              ee_pose.pose.orientation.y,
                              ee_pose.pose.orientation.z,]

            grip_pos = np.array(ee_position)
            # TODO use the real object pose
            object_pos = np.array(self.sia_7f_arm_robot.get_object_position())
            object_rel_pos = object_pos - grip_pos
            sia_qpos = np.array(joint_angles)
            sia_qvel = np.array(joint_velocity)
            if self.debug_print:
                print("grip_pos: ", grip_pos)
                print("object_pos: ", object_pos)
                print("object_rel_pos: ", object_rel_pos)
                print("sia_qpos: ", sia_qpos)
                print("sia_qvel: ", sia_qvel)
            obs = np.concatenate([
                grip_pos,
                object_pos,
                object_rel_pos,
                sia_qpos,
                sia_qvel,
                [self._is_success],
            ])
            if self.debug_print:
                print("observation: ", obs)
            return obs
        # For simulation
        else:
            # positions
            grip_pos = self.sim.data.get_site_xpos('r_grip_site')
            dt = self.sim.nsubsteps * self.sim.model.opt.timestep
            grip_velp = self.sim.data.get_site_xvelp('r_grip_site') * dt
            robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
            sia_qpos, sia_qvel = utils_sia.robot_get_sia_joint_state_obs(self.sim)
            if self.has_object:
                object_pos = self.sim.data.get_site_xpos('object0')
                # rotations
                object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
                # velocities
                object_velp = self.sim.data.get_site_xvelp('object0') * dt
                object_velr = self.sim.data.get_site_xvelr('object0') * dt
                # gripper state
                object_rel_pos = object_pos - grip_pos
                object_velp -= grip_velp
            else:
                object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
            # gripper_state = robot_qpos[-2:]
            # gripper_state = robot_qpos[-13:-1]
            # gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

            if not self.has_object:
                achieved_goal = grip_pos.copy()
            else:
                achieved_goal = np.squeeze(object_pos.copy())
            if self.debug_print:
                print("grip_pos: ", grip_pos)
                print("object_pos: ", object_pos)
                print("object_pos.ravel: ", object_pos.ravel())
                print("object_rel_pos.ravel: ", object_rel_pos.ravel())
                print("object_rel_pos: ", object_rel_pos)
                print("sia_qpos: ", sia_qpos)
                print("sia_qvel: ", sia_qvel)
            obs = np.concatenate([
                grip_pos, 
                object_pos.ravel(), 
                object_rel_pos.ravel(), 
                sia_qpos,
                sia_qvel,
                #gripper_state, 
                #object_rot.ravel(),
                #object_velp.ravel(), 
                #object_velr.ravel(), 
                #grip_velp, 
                #gripper_vel,
                [self._is_success],
            ])

            return obs


    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('r_gripper_palm_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)

            object_xpos = np.array([1.0, -0.5])  + self.np_random.uniform(-0.02, 0.07, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            # object_qpos1 = self.sim.data.get_joint_qpos('object1:joint')
            assert object_qpos.shape == (7,)
            if self.debug_print:
                print("object_xpos0: ", object_xpos)
            # print("object1 pos: ", object_qpos1)
            object_qpos[:2] = object_xpos
            object_qpos[2] = 0.9
            # object_qpos[0] += 0.3
            # object_qpos[1] -= 0.1
            if self.debug_print:
                print("set_joint_qpos object_qpos: ", object_qpos)
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
            # print("get_body_xquat: ", self.sim.data.get_body_xquat('r_gripper_palm_link'))

        # set random gripper position
        # for i in range(3):
        #     gripper_target[i] += self.np_random.uniform(-0.2, 0.2)
        # print("gripper target random: ", gripper_target)
 

        #gripper_control = self.np_random.uniform(-1.0, 1.0)
        #gripper_control = self.gripper_format_action(gripper_control)

        gripper_target = np.array([0.80326763, 0.01372008, 0.7910795])
        gripper_rotation = np.array([0.5, 0.5, -0.5, -0.5]) #(0, 0, -90)
        # for i in range(3):
        gripper_target[0] += self.np_random.uniform(-0.0, 0.1) # x
        gripper_target[1] += self.np_random.uniform(-0.1, 0.1) # y
        gripper_target[2] += self.np_random.uniform(-0.1, 0.1) # z
        self.sim.data.set_mocap_pos('gripper_r:mocap', gripper_target)
        self.sim.data.set_mocap_quat('gripper_r:mocap', gripper_rotation)

        #action = np.concatenate([gripper_target, gripper_rotation, gripper_control])
        # Apply action to simulation.
        #utils.ctrl_set_action(self.sim, action) # base control + gripper control
        # utils.mocap_set_action(self.sim, action) # arm control in cartesion (x, y, z)

        for _ in range(10):
            self.sim.step()

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0.2, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        # gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('r_grip_site')
        gripper_target = np.array([0.498, 0.005, 0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('r_grip_site')
        if self.debug_print:
            print("gripper quat: ", self.sim.data.get_site_xmat('r_grip_site'))
            print("get_mocap_quat: ", self.sim.data.get_mocap_quat('gripper_r:mocap'))
        # gripper_rotation = np.array([1., 0., 1., 0.])
        # gripper_rotation = np.array([-0.82031777, -0.33347336, -0.32553968,  0.33150896])

        # gripper_target = np.array([0.9, -0.3, 0.6])
        gripper_target = np.array([0.80326763, 0.01372008, 0.7910795]) 
        # gripper_rotation = np.array([0.5, -0.5, -0.5, -0.5]) #(-90, 0, -90)
        # gripper_rotation = np.array([0.5, 0.5, 0.5, -0.5]) #(90, 0, 90) gripper down
        # gripper_rotation = np.array([0.707, 0.0, 0.0, -0.707]) #(0, 0, -90)
        gripper_rotation = np.array([0.5, 0.5, -0.5, -0.5])
        # gripper_rotation = np.array([1.0, 0, 0, 0])
        # set random gripper position
        # for i in range(3):
        #     gripper_target[i] += self.np_random.uniform(-0.2, 0.2)
        # print("gripper target random: ", gripper_target)
        self.sim.data.set_mocap_pos('gripper_r:mocap', gripper_target)
        self.sim.data.set_mocap_quat('gripper_r:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # # set random end-effector position
        # end_effector_pos = self.initial_gripper_xpos
        # rot_ctrl = [0, 0.707, 0.707, 0] #(0 0 0)
        # end_effector_pos = np.concatenate([end_effector_pos, rot_ctrl])
        # print("end_effector_pos: ", end_effector_pos)
        # for i in range(3):
        #     end_effector_pos[i] = self.initial_gripper_xpos[i] + self.np_random.uniform(-0.1, 0.1)
        # utils.mocap_set_action(self.sim, end_effector_pos) # arm control in cartesion (x, y, z)
        # # for _ in range(100):
        #     # self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('r_grip_site').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(MobileSIA7FARMGymEnv, self).render(mode, width, height)

    def gripper_format_action(self, action):
        """ Given (-1,1) abstract control as np-array return the (-1,1) control signals
        for underlying actuators as 1-d np array
        Args:
            action: 1 => open, -1 => closed
        """
        movement = np.array([-1, -1, -1, 1, 1, 1])
        return -1 * movement * action
