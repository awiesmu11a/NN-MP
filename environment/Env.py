import time
import pybullet_data
from pybullet_utils import bullet_client
import math
import json

from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 

from obstacle import Obstacle

from stable_baselines3.common.env_checker import check_env

from Car import *
from point import Point
import cv2
import matplotlib.pyplot as plt
import sys

INIT_POSITION = [-1, 0, 0.1]
INIT_ORIENTATION = p.getQuaternionFromEuler([0,0,0])

class CarEnv(Env):
    def __init__(self, render = False, eval = False):

        self.rewards = []

        with open("config/config.json", "r") as json_file:
            config = json.load(json_file)

        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        if render:
            self.client = bullet_client.BulletClient(connection_mode=p.GUI)
        else:
            self.client = bullet_client.BulletClient(connection_mode=p.DIRECT)
        self.eval = eval
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Setting initial environment parameters
        self.client.setGravity(0, 0, -100)
        self.client.setRealTimeSimulation(0)
        self.client.loadURDF("plane.urdf")
        self.client.setTimeStep(0.05)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        self.car = Car(self.client)

        self.projection_matrix = self.car.FOVProjectionMatrix()

        self.boxPos = [10, -10]
        self.box = self.client.loadURDF(currentdir + "/urdf/cube.urdf",
                        [self.boxPos[0], self.boxPos[1], 1],
                        [0, 0, 1.73, 0.7],
                        useFixedBase=False,
                        globalScaling=0.5)
        self.carObj = self.car._LoadRobotURDF()

        self.num_obstacles = config["env"]["obs"]
        self.obstacles = []
        self.obstacle_pos = self.generateRandomObstacles(self.num_obstacles)
        for i in range(self.num_obstacles):
            temp = Obstacle(self.client, [self.obstacle_pos[i][0], self.obstacle_pos[i][1], 1])
            temp.loadURDF()
            self.obstacles.append(temp)
        
        self.timeElapsed = 0
        self.steps = 0
        self.endTime = config["PPO"]["n_steps"]
        self.rewardAccumulated = 0

        self.action_space = Box(low=np.array([config["env"]["steer_range"][0], config["env"]["acc_range"][0]]), 
                                high=np.array([config["env"]["steer_range"][1], config["env"]["acc_range"][1]]), 
                                shape=(2,), dtype=np.float32)

        self.img_size = (64, 64, 1)
        self.history_len = config["env"]["num_states"]
        self.state = np.array([20, 0, self.car.carVelocity, self.car.carAngle])
        for i in range(self.history_len - 1):
            self.state = np.concatenate((self.state, np.array([20, 0, self.car.carVelocity, self.car.carAngle])))

        obs_space_low = np.array([-20, -20, config["env"]["acc_range"][0], config["env"]["steer_range"][0]])
        obs_space_high = np.array([20, 20, config["env"]["acc_range"][1], config["env"]["steer_range"][1]])

        for i in range(self.history_len - 1):
            obs_space_low = np.concatenate((obs_space_low, np.array([-20, -20, config["env"]["acc_range"][0], config["env"]["steer_range"][0]])))
            obs_space_high = np.concatenate((obs_space_high, np.array([20, 20, config["env"]["acc_range"][1], config["env"]["steer_range"][1]])))

        self.num_images = config["env"]["num_images"]
        self.images = self.getDepthMap()
        for i in range(self.num_images - 1):
            self.images = np.concatenate((self.images, self.getDepthMap()), axis=2)
        

        self.observation_space = Dict(
            spaces={
                "vec": Box(low = obs_space_low, high = obs_space_high, shape = (self.history_len * 4,), dtype=np.float32),
                "img": Box(0, 255, self.images.shape, dtype=np.uint8),
            }
        )

        self.done = False
        self.init_box = self.boxPos
        self.init_car = [0,0,0]
        self.prev_action = 0
        self.start = time.time()
        print("Generating env.. Box - ", self.boxPos)

    def generateRandomBox(self):
        return np.array([10,0])
        with open("config/config.json", "r") as json_file:
            config = json.load(json_file)
        goal_range = config["env"]["goal_range"]
        quadrant = np.random.randint(0,2,2)
        if quadrant[0] == 0:
            quadrant[0] = -1
        if quadrant[1] == 0:
            quadrant[1] = -1
        return np.random.randint(goal_range[0], goal_range[1], 2) * quadrant


    def generateRandomObstacles(self, n):
        with open("config/config.json", "r") as json_file:
            config = json.load(json_file)
        space_range = config["env"]["space_range"]
        obs = []
        obs.append([5, 0])
        return obs
        for _ in range(n):
            quadrant = np.random.randint(0,2,2)
            if quadrant[0] == 0:
                quadrant[0] = -1
            if quadrant[1] == 0:
                quadrant[1] = -1
            obs.append(np.random.randint(space_range[0], space_range[1], 2) * quadrant)
        
        return obs
    
    def generateRandomPoint(self):
        with open("config/config.json", "r") as json_file:
            config = json.load(json_file)
        car_init = config["env"]["car_init"]
        return np.array([car_init[0], car_init[1], 0.5])

    def getBoxPos(self):
        currentPos, _ =  self.client.getBasePositionAndOrientation(self.box)
        currentPos = list(self.currPos)
        return currentPos

    def getCarPose(self):
        pos = list(self.client.getBasePositionAndOrientation(self.carObj)[0])
        orn = list(self.client.getEulerFromQuaternion(self.client.getBasePositionAndOrientation(self.carObj)[1]))

        return pos, orn
    
    def getDepthMap(self):
        pos, orn = self.getCarPose()

        front_cam = [0.39 * (math.cos(orn[2])), 0.345 * (math.sin(orn[2])), 0.4]  # needs to match our dimensions
        camera_pos = [pos[i] + front_cam[i] for i in range(3)]

        x = math.cos(orn[2]) + pos[0]
        y = math.sin(orn[2]) + pos[1]

        camera_target = [x, y, 0.4]
        view_matrix = self.client.computeViewMatrix(camera_pos, camera_target, [0, 0, 1])

        # Get depth values using the OpenGL renderer
        images = self.client.getCameraImage(self.img_size[0],
                                self.img_size[1],
                                view_matrix,
                                self.projection_matrix,
                                shadow=True)
        depthmap = np.uint8(255.0*images[3])
        depthmap = depthmap.reshape(self.img_size)
        return depthmap
        
    
    def step(self, action):

        with open("config/config.json", "r") as json_file:
            config = json.load(json_file)
        self.steps+=1
        # print(time.time() - self.start)
        # print("------------------------")
        if(self.eval):
            print(action)
        self.car.action_cont(action)
        self.start = time.time()
        time.sleep(0.003)
        self.client.stepSimulation()
        check = self.checkComplete()
        reward = self.getReward(self.state, action)
        self.timeElapsed+=1

        if check == 1:
            self.done = True
        elif check == 2:
            reward += config["env"]["goal_reward"]
            self.done = True
        elif check==3:
            reward -= config["env"]["obs_penalty"]
            self.done = True
            reward

        state_obs = self.getCurrObservations()

        info = {}
        self.rewardAccumulated += reward

        #self.boxPos[0] += 0.04
        #self.boxPos[1] += 0.04
        #self.client.resetBasePositionAndOrientation(self.box, [self.boxPos[0], self.boxPos[1], 1], [0, 0, 1.73, 0.7])


        if self.done:
            print(self.steps, "Steps")
            print(check, "Check")
            print(self.rewardAccumulated)
            self.rewards.append(self.rewardAccumulated)
        return state_obs, reward, self.done, info
    
    def norm_steer(self, x):
        return ((x + 0.7) / 1.4)
    
    def getReward(self, obs, action):

        # reward = -1.0
        
        # # Force norm component
        # # cf = 0.1
        # # reward += -cf*np.sqrt(action[0]**2 + action[1]**2)

        # # Velocity norm component
        # # vel = self.point.getVel()
        # # cv = 0.1
        # # reward += -cv*np.sqrt(vel[0]**2 + vel[1]**2)

        # reward = -1.0

        # # Distance

        # alpha = 1.0
        # beta = 0.005
        # # reward -= 0.1 * (self.car.carVelocity / self.goalDist())
        
        # reward += alpha*np.exp(-beta*self.goalDist() + 0.5 * (self.car.carVelocity / self.goalDist()))
        # mean_change = ((abs(action[0] - obs[3]) + abs(obs[3] - obs[7]) + abs(obs[7] - obs[11])) / 3)
        # # reward -= 0.5 * np.exp(abs(action[0] - obs[3]))

        # return reward

        with open("config/config.json", "r") as json_file:
            config = json.load(json_file)
        # Penalising for time
        reward = -1 # * np.exp(self.timeElapsed / self.endTime)
        step_distance = np.sqrt((obs[0] - obs[4])**2 + (obs[1] - obs[5]) ** 2)
        # reward -= 5 * step_distance
        # print(step_distance)
        # action_delta = abs(action[0] - self.prev_action)
        # reward -= 0.5*np.exp(action_delta / 1.4)
        '''
        reward -= 0.5 * np.exp(((abs(self.norm_steer(obs[3]) - self.norm_steer(obs[7])) 
                        + abs(self.norm_steer(obs[7]) - self.norm_steer(obs[11]))) / 2) * 0.5)
        '''
        # reward -= 0.5 * np.exp(((abs(action - obs[3]) + abs(obs[3] - obs[7]) + abs(obs[7] - obs[11])) / 3) * 0.5)
        # reward -= 0.7 * np.exp((abs(obs[3] - obs[7])))
        # print(1 * ((abs(action - obs[3]) + abs(obs[3] - obs[7]) + abs(obs[7] - obs[11])) / 3))

        # Distance
        x = self.goalDist()

        # Reward depending on the yaw angle of the car
        pos, orn = self.getCarPose()
        yaw = orn[2]

        car_pos = [pos[0], pos[1]]
        box_pos = self.boxPos

        delta = box_pos - car_pos
        yaw_diff = math.atan2(delta[1], delta[0])

        y = abs(yaw_diff - yaw)

        # Constants
        alpha = config["env"]["dist_rew_cons"][0]
        beta = config["env"]["dist_rew_cons"][1]

        gamma = config["env"]["dir_rew_cons"][0]
        eta = config["env"]["dir_rew_cons"][1]
        
        reward += alpha * np.exp(-beta * x + 0.0 * (self.car.carVelocity / self.goalDist()))
        reward += gamma * np.exp(-eta * y / x)
        return reward

    def carOutOfBounds(self):
        pos, orn = self.getCarPose()
        if max(abs(np.array(pos)))>=30:
            return True
        return False

    def pointOutOfBounds(self):
        if max(abs(np.array(self.pointPos)))>=30:
            return True
        return False
    
    def moveMagnitude(self):
        linear_vel, angular_val = self.car.GetIMU()
        movement = linear_vel[0]**2 + linear_vel[1]**2 + angular_val[0]**2 + angular_val[1]**2
        return movement
    
    def moving(self):
        linear_vel, angular_val = self.car.GetIMU()
        movement = linear_vel[0]**2 + linear_vel[1]**2 + angular_val[0]**2 + angular_val[1]**2

        return movement>1

    def checkComplete(self):
        if self.timeElapsed >= self.endTime:
            return 1
        elif self.goalReached():
            return 2
        elif self.obstructed():
            return 3
        
        return 0

    def obstructed(self):
        pos, _ = self.getCarPose()
        for i in range(self.num_obstacles):
            dist = np.sqrt((pos[0]-self.obstacle_pos[i][0])**2 + (pos[1]-self.obstacle_pos[i][1])**2)
            if dist<1.0:
                return True
        return False

    def goalReached(self):
        with open("config/config.json", "r") as json_file:
            config = json.load(json_file)
        dist = self.goalDist()
        threshold = config["env"]["goal_threshold"]
        return dist <= threshold

    def goalDist(self):
        pos, _ = self.getCarPose()
        dist = (pos[0]-self.boxPos[0])**2 + (pos[1]-self.boxPos[1])**2
        return np.sqrt(dist)

    def getCurrObservations(self):
        pos, orn = self.getCarPose()
        R = self.getRotation(orn[2])
        car_pos = np.array([[pos[0]], [pos[1]]])
        box_pos = np.array(self.boxPos).reshape(-1,1)
        relative_pos = np.squeeze(R@(box_pos-car_pos))

        # History of states
        self.state = np.concatenate((np.array([relative_pos[0], relative_pos[1], self.car.carVelocity, self.car.carAngle]), self.state[:-4]))
        obs = self.state

        # History of images
        img = self.getDepthMap()
        self.images = np.concatenate((self.getDepthMap(), self.images[:,:,:self.num_images-1]), axis=2)

        return {"img": self.images, "vec": obs}
    
    def getResetObservations(self):

        pos, orn = self.getCarPose()
        R = self.getRotation(orn[2])
        car_pos = np.array([[pos[0]], [pos[1]]])
        box_pos = np.array(self.boxPos).reshape(-1,1)
        relative_pos = np.squeeze(R@(box_pos-car_pos))

        self.state = np.array([relative_pos[0], relative_pos[1], self.car.carVelocity, self.car.carAngle])

        for i in range(self.history_len - 1):
            self.state = np.concatenate((self.state, np.array([relative_pos[0], relative_pos[1], self.car.carVelocity, self.car.carAngle])))
        obs = self.state

        self.images = self.getDepthMap()
        for i in range(self.num_images - 1):
            self.images = np.concatenate((self.images, self.getDepthMap()), axis=2)
        
        return {"img": self.images, "vec": obs}


    def render(self):
        self.client = self.client.connect(p.GUI)
        
    def getRotation(self, theta):
        return np.array([
            [np.cos(theta), np.sin(theta)],
            [np.sin(theta), -np.cos(theta)]
        ])

    def close(self):
        print("Disconnecting....")
        self.client.disconnect()

    def reset(self):

        print("Initial Box - ", self.init_box)
        print("Initial Car - ", self.init_car)

        print("Final distance - ", self.goalDist())
        pos, _ = self.getCarPose()
        print("Final Coordinates - ", pos)

        self.car.resetAngle()
        self.car.resetVelocity()

        self.car_pos = self.generateRandomPoint()
        self.client.resetBasePositionAndOrientation(self.carObj, self.car_pos, INIT_ORIENTATION)
        
        self.boxPos = self.generateRandomBox()
        self.client.resetBasePositionAndOrientation(self.box, [self.boxPos[0], self.boxPos[1], 1], [0, 0, 1.73, 0.7])
        
        self.obstacle_pos = self.generateRandomObstacles(self.num_obstacles)
        for i in range(self.num_obstacles):
            self.obstacles[i].resetPosition([self.obstacle_pos[i][0], self.obstacle_pos[i][1], 1])

        self.done = False
        self.timeElapsed = 0
        self.steps = 0
        self.client.stepSimulation()
        self.rewardAccumulated = 0
        observations = self.getResetObservations()
        
        self.init_box = self.boxPos
        self.init_car = self.car_pos
        print("[RESET] Generating env...")
        return observations
    

if __name__ == "__main__":

    # To display the depth sensor data    
    env = CarEnv()
    obs = env.getCurrObservations()
    imgs = obs["img"][:,:,0]
    plt.imshow(imgs)
    plt.show()
    env.close()
