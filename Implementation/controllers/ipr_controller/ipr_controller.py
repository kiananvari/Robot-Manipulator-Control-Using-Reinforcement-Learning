# Add Webots controlling libraries
import select
from controller import Robot
from controller import Supervisor


# Some general libraries
import os
import time
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

# Open CV
import cv2 as cv

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# Stable_baselines3
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

# Create an instance of robot
robot = Robot()

# Seed Everything
seed = 42
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.loss_values = []  # Add this line to store loss values


    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              
              mean_reward = np.mean(y[-100:])
            #   print("mean_reward:-------------------------------> ", mean_reward)
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print("----------------------------------------------------------------------------------------------")
                    print(f"Saving new best model to {self.save_path}")
                    print("----------------------------------------------------------------------------------------------")
                  self.model.save(self.save_path)

        return True

class Environment(gym.Env, Supervisor):
    """The robot's environment in Webots."""
    
    def __init__(self):
        super().__init__()
                
        # General environment parameters
        self.floor_size = np.linalg.norm([5, 5])

        self.last_EF_pos = np.zeros(3)
        self.previous_EF_pos = np.zeros(3)
        self.poses = []
        self.next_index = 0      
        self.flag = 0
        
        # Activate Devices
        sampling_period = 1 # in ms
        # Get Devices
        self.base = robot.getDevice("base")
        self.forearm = robot.getDevice("forearm")
        self.left_gripper = robot.getDevice("gripper::left")
        self.right_gripper = robot.getDevice("gripper::right")
        self.rotational_wrist = robot.getDevice("rotational_wrist")
        self.upperarm = robot.getDevice("upperarm")
        self.wrist = robot.getDevice("wrist")

        # Get Encoders
        self.base_sensor = robot.getDevice("base_sensor")
        self.base_sensor.enable(sampling_period)
        self.forearm_sensor = robot.getDevice("forearm_sensor")
        self.forearm_sensor.enable(sampling_period)
        self.left_gripper_sensor = robot.getDevice("gripper::left_sensor")
        self.left_gripper_sensor.enable(sampling_period)
        self.right_gripper_sensor = robot.getDevice("gripper::right_sensor")
        self.right_gripper_sensor.enable(sampling_period)
        self.rotational_wrist_sensor = robot.getDevice("rotational_wrist_sensor")
        self.rotational_wrist_sensor.enable(sampling_period)
        self.upperarm_sensor = robot.getDevice("upperarm_sensor")
        self.upperarm_sensor.enable(sampling_period)
        self.wrist_sensor = robot.getDevice("wrist_sensor")
        self.wrist_sensor.enable(sampling_period)
        
        # Enable Goal Senors
        # Camera
        # self.cameraG1 = robot.getDevice("cameraG1")
        # self.cameraG1.enable(sampling_period)
        # self.cameraG2 = robot.getDevice("cameraG2")
        # self.cameraG2.enable(sampling_period)
        # self.cameraG3 = robot.getDevice("cameraG3")
        # self.cameraG3.enable(sampling_period)

        # DistanceSensor
        self.distance_sensors = []
        self.dist_sensorG1 = robot.getDevice("DS1")
        self.dist_sensorG1.enable(sampling_period)
        self.dist_sensorG2 = robot.getDevice("DS2")
        self.dist_sensorG2.enable(sampling_period)
        self.dist_sensorG3 = robot.getDevice("DS3")
        self.dist_sensorG3.enable(sampling_period)    
        self.dist_sensorG4 = robot.getDevice("DS4")
        self.dist_sensorG4.enable(sampling_period)
        self.dist_sensorG5 = robot.getDevice("DS5")
        self.dist_sensorG5.enable(sampling_period)
        self.dist_sensorG6 = robot.getDevice("DS6")
        self.dist_sensorG6.enable(sampling_period)
        self.dist_sensorG7 = robot.getDevice("DS7")
        self.dist_sensorG7.enable(sampling_period)
        self.dist_sensorG8 = robot.getDevice("DS8")
        self.dist_sensorG8.enable(sampling_period)
        self.dist_sensorG9 = robot.getDevice("DS9")
        self.dist_sensorG9.enable(sampling_period)
        self.dist_sensorG10 = robot.getDevice("DS10")
        self.dist_sensorG10.enable(sampling_period)
        self.distance_sensors.append(self.dist_sensorG1)
        self.distance_sensors.append(self.dist_sensorG2)
        self.distance_sensors.append(self.dist_sensorG3)
        self.distance_sensors.append(self.dist_sensorG4)
        self.distance_sensors.append(self.dist_sensorG5)
        self.distance_sensors.append(self.dist_sensorG6)
        self.distance_sensors.append(self.dist_sensorG7)
        self.distance_sensors.append(self.dist_sensorG8)
        self.distance_sensors.append(self.dist_sensorG9)
        self.distance_sensors.append(self.dist_sensorG10)


        self.distance_sensorsOBS = []
        self.dist_sensorO1 = robot.getDevice("DSO1")
        self.dist_sensorO1.enable(sampling_period)
        self.dist_sensorO2 = robot.getDevice("DSO2")
        self.dist_sensorO2.enable(sampling_period)
        self.dist_sensorO3 = robot.getDevice("DSO3")
        self.dist_sensorO3.enable(sampling_period)    
        self.dist_sensorO4 = robot.getDevice("DSO4")
        self.dist_sensorO4.enable(sampling_period)


        self.max_sensor = 0
        self.min_sensor = 0
        self.max_sensor = max(self.dist_sensorG1.max_value, self.max_sensor)    
        self.min_sensor = min(self.dist_sensorG1.min_value, self.min_sensor)

        # TouchSensor
        self.touchG1 = robot.getDevice("touch_sensorG1")
        self.touchG1.enable(sampling_period)
        self.touchG2 = robot.getDevice("touch_sensorG2")
        self.touchG2.enable(sampling_period)

        self.touchO1 = robot.getDevice("touch_sensorO1")
        self.touchO1.enable(sampling_period)

        # End Effector GPS
        self.gps = robot.getDevice("gps")
        self.gps.enable(sampling_period)


        n_actions = 5
        #Space
        self.action_space = spaces.Box(low = 0, high = 1, shape = (n_actions,), dtype = np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        self.max_steps = 20

        # Reset
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        robot.step(200) # take some dummy steps in environment for initialization
        
    def add_pose(self, pose_array, current_pose, threshold=0.2):
        closest_index = None
        min_distance = float('inf')

        if len(pose_array) != 0:
            for i, pose in enumerate(pose_array):
                distance = np.linalg.norm(current_pose - pose, ord=2)
                if distance < min_distance:
                    min_distance = distance
                    closest_index = i

        if min_distance >= threshold or len(pose_array) == 0:
            pose_array.append(current_pose)
            return len(pose_array) - 1
        else:
            return closest_index
    
    def initial_pos(self):
                
        self.forearm.setPosition(2.15)
        self.base.setPosition(3.01)
        self.left_gripper.setPosition(0)
        self.right_gripper.setPosition(0)
        self.rotational_wrist.setPosition(-2.9)   
        self.upperarm.setPosition(-0.55)
        self.wrist.setPosition(-2)
        robot.step(500)
       
    def normalizer(self, value, min_value, max_value):
        """
        Performs min-max normalization on the given value.

        Returns:
        - float: Normalized value.
        """
        normalized_value = (value - min_value) / (max_value - min_value)        
        return normalized_value

    def get_distance_sensors_data(self):
        """
        Retrieves and normalizes data from distance sensors.
        
        Returns:
        - numpy.ndarray: Normalized distance sensor data.
        """

        # Gather values of distance sensors.
        # print(self.distance_sensors[0].getValue())
        sensors_data = []
        for z in range(0,len(self.distance_sensors)):
            sensors_data.append(self.distance_sensors[z].getValue())  
            
        sensors_data = np.array(sensors_data)
        normalized_sensors_data = self.normalizer(sensors_data, self.min_sensor, self.max_sensor)

        sensorsOBS_data = []
        for z in range(0,len(self.distance_sensorsOBS)):
            sensorsOBS_data.append(self.distance_sensorsOBS[z].getValue())  
            
        sensorsOBS_data = np.array(sensorsOBS_data)
        normalized_sensorsOBS_data = self.normalizer(sensorsOBS_data, self.min_sensor, self.max_sensor)

        return normalized_sensors_data, normalized_sensorsOBS_data
    
    def get_current_position(self):
        """
        Retrieves and normalizes data from distance sensors.
        
        Returns:
        - numpy.ndarray: Normalized distance sensor data.
        """
        
        # Gather values of distance sensors.

        base_pose = self.base_sensor.getValue()
        base_pose = np.array(base_pose)
        forearm_pos = self.forearm_sensor.getValue()
        forearm_pos = np.array(forearm_pos)
        left_gripper_pos = self.left_gripper_sensor.getValue()
        left_gripper_pos = np.array(left_gripper_pos)
        right_gripper_pos = self.right_gripper_sensor.getValue()
        right_gripper_pos = np.array(right_gripper_pos)
        rotational_wrist_pos = self.rotational_wrist_sensor.getValue()
        rotational_wrist_pos = np.array(rotational_wrist_pos)
        upperarm_pos = self.upperarm_sensor.getValue()
        upperarm_pos = np.array(upperarm_pos)
        wrist_pos = self.wrist_sensor.getValue()
        wrist_pos = np.array(wrist_pos)

        EF_position = self.gps.getValues()[0:3]
        EF_position = np.array(EF_position)


        return base_pose, forearm_pos, left_gripper_pos, right_gripper_pos, rotational_wrist_pos, upperarm_pos, wrist_pos, EF_position
    

    def get_observations(self):
        # """
        # Obtains and returns the normalized sensor data, current distance to the goal, and current position of the robot.
    
        # Returns:
        # - numpy.ndarray: State vector representing distance to goal, distance sensor values, and current position.
        # """
    
        base_pose, forearm_pos, left_gripper_pos, right_gripper_pos, rotational_wrist_pos, upperarm_pos, wrist_pos, EF_position = self.get_current_position()

        state_vector = np.array([base_pose, forearm_pos, rotational_wrist_pos, upperarm_pos, wrist_pos, EF_position[0], EF_position[1], EF_position[2]])

        return state_vector
     
    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state and returns the initial observations.
        
        Returns:
        - numpy.ndarray: Initial state vector.
        """

        self.simulationReset()
        self.simulationResetPhysics()
        # self.initial_pos()

        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        return self.get_observations(), {}

    def step(self, action):    
        """
        Takes a step in the environment based on the given action.
        
        Returns:
        - state       = float numpy.ndarray with shape of (3,)
        - step_reward = float
        - done        = bool
        """
        self.apply_action(action)
        step_reward, done = self.get_reward()
        state = self.get_observations()
        # Time-based termination condition
        # print((int(self.getTime()) + 1))
        if (int(self.getTime()) + 1) % self.max_steps == 0:
            done = True
        none = 0
        return state, step_reward, done, none, {}
        

    def get_reward(self):
        """
        Calculates and returns the reward based on the current state.
        
        Returns:
        - The reward and done flag.
        """
        done = False
        reward = 0
        

        normalized_distance_sensors_data, normalized_distance_sensorsOBS_data = self.get_distance_sensors_data()

        # DS Rewards
        if np.any(normalized_distance_sensors_data[normalized_distance_sensors_data < 0.3]):
            reward += 0.01
        if np.any(normalized_distance_sensors_data[normalized_distance_sensors_data < 0.2]):
            reward += 0.01

        if np.any(normalized_distance_sensorsOBS_data[normalized_distance_sensorsOBS_data < 0.3]):
            reward -= 0.01
        if np.any(normalized_distance_sensorsOBS_data[normalized_distance_sensorsOBS_data < 0.2]):
            reward -= 0.01

        #Touch
        check_collisionG1 = self.touchG1.value
        check_collisionG2 = self.touchG2.value
        check_collisionO1 = self.touchO1.value


        if check_collisionG1 or check_collisionG2:
            current_EF_pos = self.gps.getValues()[0:3]
            current_EF_pos = np.array(current_EF_pos)

            # print(f"Current Pos = {current_EF_pos}")
            # print(f"Last Pos = {self.last_EF_pos}")

            difference = current_EF_pos - self.last_EF_pos
            difference_last = np.linalg.norm(difference, ord=2)

            difference_last = current_EF_pos - self.last_EF_pos
            difference_previous = current_EF_pos - self.previous_EF_pos

            euclidean_distance_last = np.linalg.norm(difference_last, ord=2)
            euclidean_distance_previous = np.linalg.norm(difference_previous, ord=2)

            print(f"Difference with Last = {euclidean_distance_last}")
            print(f"Difference with Previous = {euclidean_distance_previous}")


            # IN ORDER
            current_EF_pos = np.array(current_EF_pos)

            if self.flag != 2:

                index = self.add_pose(self.poses, current_EF_pos, 0)
                if index == self.next_index:
                    reward += 15
                else:
                    reward -= 2
                self.next_index = (index + 1) % 3
                self.flag += 3

            else:
                index = self.add_pose(self.poses, current_EF_pos, 0.2)
                if index == self.next_index:
                    reward += 15
                else:
                    reward -= 3
                self.next_index = (index + 1) % 3


            if euclidean_distance_last < 0.3:
                reward -= 70
            if euclidean_distance_previous < 0.3:
                reward -= 40

            if euclidean_distance_last > 0.4 and euclidean_distance_previous > 0.4:
                reward += 130

            if euclidean_distance_last > 0.8:
                    reward -= 7
            if euclidean_distance_last > 0.7 and euclidean_distance_last < 0.8:
                    reward += 10
            if euclidean_distance_last > 0.5 and euclidean_distance_last < 0.7:
                    reward += 30
            if euclidean_distance_last > 0.3 and euclidean_distance_last < 0.5:
                    reward -= 7

            if euclidean_distance_last > 0.3:
                self.previous_EF_pos = self.last_EF_pos
                self.last_EF_pos = current_EF_pos


            print("TOUCHED!")
            done = True


        if check_collisionO1:
            print("OBS-TOUCHED!")
            reward -= 5

        return reward, done


    def apply_action(self, action):
        """
        Applies the specified action to the robot's motors.
        
        Returns:
        - None
        """

        def convert_range(value, old_min, old_max, new_min, new_max):
            old_range = old_max - old_min
            new_range = new_max - new_min
            new_value = ((value - old_min) / old_range) * new_range + new_min
            return new_value

        base_min = 0
        base_max = 6.03
        forearm_min = 0
        forearm_max = 4.21
        # rotational_wrist_min = -5.8
        rotational_wrist_min = -5.7
        rotational_wrist_max = 0
        upperarm_min = -2.44
        upperarm_max = 0
        # wrist_min = -4.05
        wrist_min = -4.0
        wrist_max = 0

        self.left_gripper.setPosition(0)
        self.right_gripper.setPosition(0)
        base_pos = convert_range(action[0], 0, 1, base_min, base_max)
        forearm_pos = convert_range(action[1], 0, 1, forearm_min, forearm_max)
        rotational_wrist_pos = convert_range(action[2], 0, 1, rotational_wrist_min, rotational_wrist_max)
        upperarm_pos = convert_range(action[3], 0, 1, upperarm_min, upperarm_max)
        wrist_pos = convert_range(action[4], 0, 1, wrist_min, wrist_max)

        self.base.setPosition(base_pos)
        self.forearm.setPosition(forearm_pos)
        self.rotational_wrist.setPosition(rotational_wrist_pos)
        self.upperarm.setPosition(upperarm_pos)
        self.wrist.setPosition(wrist_pos)

        robot.step(200)
        

class Agent_FUNCTION(): #Train and Test
    def __init__(self, save_path, num_episodes):
        self.save_path = save_path
        self.num_episodes = num_episodes

        self.env = Environment()
        self.env = Monitor(self.env, "tmp/")

        self.policy_network = PPO("MlpPolicy", self.env,verbose=1, tensorboard_log=self.save_path)
    

    
    def save(self):
        print(self.save_path ,"PPO-Best")
        self.policy_network.save(self.save_path + "PPO-Best")


    def load(self):
        self.policy_network = PPO.load("./tmp/best_model.zip")


    def train(self):

        log_dir = "tmp/"
        os.makedirs(log_dir, exist_ok=True)
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
        # Train the agent
        self.policy_network.learn(total_timesteps=int(self.num_episodes), callback=callback)

        self.env.reset()


    def test(self):
        
        state = self.env.reset()
        for episode in range(1, self.num_episodes+1):
            rewards = []
            done = False
            ep_reward = 0
            state=np.array(state[0])
            while not done:
                # Ensure the state is in the correct format (convert to numpy array if needed)
                state_np = np.array(state)

                # Get the action from the policy network
                action, _ = self.policy_network.predict(state_np)

                # Take the action in the environment
                state, reward, done, _,_ = self.env.step(action)
                ep_reward += reward

            rewards.append(ep_reward)
            print(f"Episode {episode}: Score = {ep_reward:.3f}")
            state = self.env.reset()

class Agent_FUNCTION1(): #CTrain
    def __init__(self, save_path, num_episodes, env):
        self.save_path = save_path
        self.num_episodes = num_episodes
        self.env = env

    
    def save(self):
        print(self.save_path ,"PPO-Best")
        self.policy_network.save(self.save_path + "PPO-Best")

    def load(self):

        self.policy_network = PPO.load("./tmp/PPO-Best")



    def train(self):

        log_dir = "tmp/"
        os.makedirs(log_dir, exist_ok=True)
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
        # Train the agent
        self.policy_network.learn(total_timesteps=int(self.num_episodes), callback=callback)

        self.env.reset()


 
    def test(self):
        
        state = self.env.reset()
        for episode in range(1, self.num_episodes+1):
            rewards = []
            done = False
            ep_reward = 0
            state=np.array(state[0])
            while not done:
                # Ensure the state is in the correct format (convert to numpy array if needed)
                state_np = np.array(state)

                # Get the action from the policy network
                action, _ = self.policy_network.predict(state_np)

                # Take the action in the environment
                state, reward, done, _,_ = self.env.step(action)
                ep_reward += reward

            rewards.append(ep_reward)
            print(f"Episode {episode}: Score = {ep_reward:.3f}")
            state = self.env.reset()


if __name__ == '__main__':
    
    # Configs
    save_path = './results/'   
    run_mode = "Test" 
    num_episodes = 600000 if run_mode == "Train" or "CTrain" else 25
    
    print("num_episodes: ", num_episodes)
    if run_mode == "Train":
        # Initialize Training
        agent = Agent_FUNCTION(save_path, num_episodes)

        agent.train()

    elif run_mode == "Test":
        agent = Agent_FUNCTION(save_path, 20)

        # Load PPO
        agent.load()
        # Test
        agent.test()
        
    elif run_mode == "CTrain":
        env = Environment()
        agent = Agent_FUNCTION1(save_path, num_episodes, env)
        
        env = Monitor(env, "tmp/")
        # env = Environment()
        agent.load()
        agent.policy_network.set_env(env)
        agent.train()




