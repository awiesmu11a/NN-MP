import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client
import math
import numpy as np
import os
import re
import time
import inspect
import json

"""Pybullet simulation of a Ackermann car robot."""

currentdir = os.path.dirname(os.path.abspath(
  inspect.getfile(inspect.currentframe())))

parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

# Setting keys for controlling and terminating the program
endKey = 32  # Spacebar
leftKey = p.B3G_LEFT_ARROW  
rightKey = p.B3G_RIGHT_ARROW  
upKey = p.B3G_UP_ARROW  
downKey = p.B3G_DOWN_ARROW  

# Recording Wheel and Hinge indices to give throttle and steering commands


NUM_MOTORS = 2
MOTOR_NAMES = ["steering_motor", "throttle_motor"]

INIT_POSITION = [-1, 0, 0.1]
INIT_ORIENTATION = p.getQuaternionFromEuler([0,0,0])
MOTOR_DIRECTIONS = np.ones(2) # Was called JOINT_DIRECTIONS, not sure what this does 

# Not sure what these are
PURE_RL_UPPER_BOUND = 0.2
PURE_RL_LOWER_BOUND = -0.2

DOFS_PER_MOTOR = 1

PI = math.pi

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.5


INIT_MOTOR_ANGLES = np.array([0, 0.9, -1.8] * NUM_MOTORS)

IMU_NAME_PATTERN = re.compile(r"imu\d*")

URDF_FILENAME = "urdf/ackermann_car.urdf"

class Car(): 
  """A simulation for the Ackermann car robot."""

  # At high replanning frequency, inaccurate values of BODY_MASS/INERTIA
  # doesn't seem to matter much. However, these values should be better tuned
  # when the replan frequency is low (e.g. using a less beefy CPU).

  def __init__(
      self,
      client,
      urdf_filename=URDF_FILENAME,
      enable_clip_motor_commands=False,
      time_step=0.001,
      action_repeat=10,
      sensors=None,
      control_latency=0.002,
      on_rack=False,
      enable_action_interpolation=True,
      enable_action_filter=False,
      motor_control_mode=None,
      reset_time=1,
      is_render=False,
      reset_position_random_range=0,
      init_pos=None
  ):
    self._urdf_filename = urdf_filename
    self.init_pos = init_pos
    self._enable_clip_motor_commands = enable_clip_motor_commands

    self._p = client

    self.wheelIndices = [2, 4, 5, 6]
    self.hingeIndices = [1, 3]

    self.carThrottle = 3  # constant acceleration
    self.carFriction = 2
    self.carVelocity = 0
    self.carAngle = 0
    self.maxSpeed = 40
    self.reverseMaxSpeed = -40

    self.action_space = [(t, s) for t in [-1,0,1] for s in [-1,0,1]]


  def _LoadRobotURDF(self):
    ackermann_urdf_path = currentdir + '/' + URDF_FILENAME
    self.car = self._p.loadURDF(ackermann_urdf_path, INIT_POSITION, INIT_ORIENTATION)
    return self.car
    
  def GetContacts(self):
    """
    for more information: http://dirkmittler.homeip.net/blend4web_ce/uranium/bullet/docs/pybullet_quickstartguide.pdf
    on page 25.
    """
    all_contacts = self._p.getContactPoints(bodyA=self.car, bodyB=-1, linkIndexA=-1)
    return all_contacts


  def ResetPose(self, add_constraint=None):
    if add_constraint is not None:
      del add_constraint
      
    self._p.resetBasePositionAndOrientation(bodyUniqueId=self.car, posObj=INIT_POSITION, ornObj=INIT_ORIENTATION)

  def GetURDFFile(self):
    return self._urdf_filename

  def _GetMotorNames(self):
    return MOTOR_NAMES

  def GetDefaultInitPosition(self):
    return INIT_POSITION if self.init_pos is None else self.init_pos

  def GetDefaultInitOrientation(self):
    return INIT_ORIENTATION

  def GetDefaultInitJointPose(self):
    print('inside Get default initial joint pose method.')
    

  def action_cont(self, control):

    with open("config/config.json", "r") as json_file:
      config = json.load(json_file)

    if self.carVelocity > 0:
      self.carVelocity -=0.5
    elif self.carVelocity < 0:
       self.carVelocity +=0.5
    
    if self.carAngle < 0:
      self.carAngle += 0.05
    elif self.carAngle >0:
      self.carAngle -= 0.05
    
    
    self.carVelocity += 1.0 * control[1]
    self.carAngle = control[0]

    self.carAngle = np.clip(self.carAngle, config["env"]["steer_range"][0], config["env"]["steer_range"][1])
    self.carVelocity = np.clip(self.carVelocity, config["env"]["acc_range"][0], config["env"]["acc_range"][1])

    for i in self.wheelIndices:
        self._p.setJointMotorControl2(self.car,
                                i,
                                self._p.VELOCITY_CONTROL,
                                targetVelocity=self.carVelocity)
    for i in self.hingeIndices:
        self._p.setJointMotorControl2(self.car,
                                i,
                                self._p.POSITION_CONTROL,
                                targetPosition=self.carAngle)
    

     

  def simpleAction(self, event: int):
    if event==0:
        if self.carVelocity < self.maxSpeed:
            self.carVelocity += self.carThrottle
    else:
        if self.carVelocity > 0:
            self.carVelocity -= self.carFriction

    if event==1:
        if self.carVelocity > self.reverseMaxSpeed:
            self.carVelocity -= self.carThrottle
    else:
        if self.carVelocity < 0:
            self.carVelocity += self.carFriction

    for i in self.wheelIndices:
        self._p.setJointMotorControl2(self.car,
                                i,
                                self._p.VELOCITY_CONTROL,
                                targetVelocity=self.carVelocity)

  def ApplyAction(self, event: int, flag=False) -> None:
    """
      This function will take in the output the network and apply throttle and steering to the car model.
      [
            (-1, -1),  ==> Reverse, Left
            (-1,  0),  ==> Reverse, None
            (-1,  1),  ==> Reverse, Right

            (0,  -1),  ==> None,    Left
            (0,   0),  ==> None,    None
            (0,   1),  ==> None,    Right

            (1,  -1),  ==> Forward, Left
            (1,   0),  ==> Forward, None
            (1,   1)   ==> Forward, Right
      ]  

      :param event: this is the output of the network. Should be 0-8
      :type event: int
    """
    # Mapping network output to possible actions
    
    throttle, steering = self.action_space[event]

    # ################# Control logic begin ################## #
    if abs(self.carVelocity) < 1:
        self.carVelocity = 0

    if abs(self.carAngle) < .1:
        self.carAngle = 0

    # if upKey in event and event[upKey] & p.KEY_IS_DOWN:
    if throttle==1:
        if self.carVelocity < self.maxSpeed:
            self.carVelocity += self.carThrottle
    else:
        if self.carVelocity > 0:
            self.carVelocity -= self.carFriction

    # if downKey in event and event[downKey] & p.KEY_IS_DOWN:
    if throttle==-1:
        if self.carVelocity > self.reverseMaxSpeed:
            self.carVelocity -= self.carThrottle
    else:
        if self.carVelocity < 0:
            self.carVelocity += self.carFriction

    # if rightKey in event and event[rightKey] & p.KEY_IS_DOWN:
    if steering==1:
        if self.carAngle >= -.9:
            self.carAngle -= .2
    else:
        if self.carAngle <= 0:
            self.carAngle += .1

    # if leftKey in event and event[leftKey] & p.KEY_IS_DOWN:
    if steering==-1:
        if self.carAngle <= .9:
            self.carAngle += .2
    else:
        if self.carAngle >= 0:
            self.carAngle -= .1

    # ################# Control logic end ################## #

    # Applying throttle and steering to all corresponding wheels and hinges
    for i in self.wheelIndices:
        self._p.setJointMotorControl2(self.car,
                                i,
                                self._p.VELOCITY_CONTROL,
                                targetVelocity=self.carVelocity)
    for i in self.hingeIndices:
        self._p.setJointMotorControl2(self.car,
                                i,
                                self._p.POSITION_CONTROL,
                                targetPosition=self.carAngle)
  
  def resetVelocity(self):
     self.carVelocity = 0
     for i in self.wheelIndices:
        self._p.setJointMotorControl2(self.car,
                                i,
                                self._p.VELOCITY_CONTROL,
                                targetVelocity=self.carVelocity)
  
  def resetAngle(self):
     self.carAngle = 0
     for i in self.hingeIndices:
        self._p.setJointMotorControl2(self.car,
                                i,
                                self._p.POSITION_CONTROL,
                                targetPosition=self.carAngle)

  def FOVProjectionMatrix(self, fov=120, aspect=1, near=0.02, far=15):
    return self._p.computeProjectionMatrixFOV(fov, aspect, near, far)

  def GetImage(self):
    print("trying to get image")

  def GetIMU(self):
    position, orientation = self._p.getBasePositionAndOrientation(self.car)
    linear_velocity, angular_velocity = self._p.getBaseVelocity(self.car)
    return linear_velocity, angular_velocity

  def PrintJoinInfo(self):
    numJoints = p.getNumJoints(self.car)
    for joint in range(numJoints):
      print(p.getJointInfo(self.car, joint))