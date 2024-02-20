# Robot-Manipulator-Control-Using-Reinforcement-Learning-
This repository contains the implementation of a robot manipulator control system using reinforcement learning techniques. The goal of this project is to train a robot to navigate an environment, detect different targets, and avoid obstacles using reinforcement learning algorithms such as PPO and curriculum learning.

# Project Overview
The project focuses on training a robot manipulator to perform specific tasks in a simulated environment. The robot, called "Neuronics' IPR," is trained to move towards different colored target objects (red, green, and blue) while avoiding obstacles. The robot starts with no prior knowledge of the positions of the targets and obstacles and relies on Distance and collision detection sensors (e.g., 2TouchSensor) to determine if it has reached them.

# Implementation Details
The project utilizes reinforcement learning techniques, specifically the PPO algorithm, to train the robot manipulator. The PPO algorithm is a policy optimization method that aims to find the optimal policy for the robot's actions based on the observed rewards and states.
To facilitate the learning process, curriculum learning is employed. The training is divided into two stages. In the first stage, the obstacle objects are removed, and the robot learns to reach the target objects efficiently. The trained model from this stage is saved for later use. In the second stage, the obstacle objects are reintroduced, and the robot learns to avoid colliding with them while still reaching the target objects.

# Documentation
You can see the description of the implementation method in the following file:
[Click Me](https://github.com/kiananvari/HMMEvaluationKit/raw/main/Documentation.pdf)

# Results
![App Screenshot](https://raw.githubusercontent.com/kiananvari/HMMEvaluationKit/main/Results/1.png)
![App Screenshot](https://raw.githubusercontent.com/kiananvari/HMMEvaluationKit/main/Results/2.png)
