# Simulator
the RM-PRT simulator is a new high-fidelity digital twin scene based on Unreal Engine 5, which includes 782 categories, 2023 objects.

## Getting Started
### Installation
```bash
pip install gym==0.21.0 protobuf==3.20.0 grpcio==1.53.0
```
### Quickstart
Here is a basic example of how to create an environment under our simulator.
```bash
import argparse
import gym
from gym import spaces
import numpy as np


import time
import random
from google.protobuf import message
import grpc
import GrabSim_pb2_grpc
import GrabSim_pb2

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str,default="localhost:30001")
parser.add_argument("--action_nums", type=int,default=8)
args, opts = parser.parse_known_args()

from SimEnv4 import SimEnv
client=args.host

deterministic=True
action_nums=args.action_nums
bins = 32
abs_distance=12
level=1

use_image = True
max_steps = 160

env=SimEnv(client,0,deterministic=deterministic,action_nums=action_nums,bins=bins,abs_distance=abs_distance,use_image=use_image,max_steps = max_steps,level=level)
```
## Tasks
- Level 1: The scene contains only one object, and the robot receives explicit machine language commands consisting of a *verbs + nouns*, *e.g.*, to grasp the glass. 
- Level 2: Compared with Level 1, the scene contains multiple objects at this time, and the robot needs to grasp the target object from multiple objects according to the instructions. 
- Level 3: On the basis of Level 2, the robot receives concise human natural language, and the instruction contains the target object that needs to be grasped. 
- Level 4: Compared with Level 3, Level 4's natural language instructions are more ambiguous. It does not clearly point out the target object to be grasped but requires the robot to fully understand human intentions and grasp the corresponding target object according to the scene and instructions.

## Observation and Action Space
```bash
env.observation_space

>>> {
    "head_rgbd": Box(0, 255, shape=(h, w, 4), dtype=np.float64),
    
    "instruction": Box([0,0], [1e5,1e5],  dtype=np.int32),
    
    "state": Box(-inf, inf,  shape=(28), dtype=np.float64),
}
```
```bash
env.action_space

>>> {
    "back": Box(-45, 45, dtype=np.float64),
    
    "right_hand": Box([-180,-90,-90,-120,-90,-23,-23], [36,23,90,12,90,23,36],  dtype=np.float64),
    
    "grasp": Box(0, 1,   dtype=np.int32),
}
```
