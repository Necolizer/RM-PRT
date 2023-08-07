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
## Concepts

