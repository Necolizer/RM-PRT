# Simulator
The RM-PRT simulator is a new high-fidelity digital twin scene based on Unreal Engine 5, which includes 782 categories, 2023 object entities.

# Scene
<div align=center>
<img src="../imgs/HighresScreenshot00007.png" width="300"> <img src="../imgs/HighresScreenshot00008.png" width="300"> <img src="https://github.com/Necolizer/RM-PRT/blob/gh-pages/docs/static/images/render/3.jpg" width="300">
</div>

<div align=center>
<img src="../imgs/HighresScreenshot00009.png" width="300"> <img src="https://github.com/Necolizer/RM-PRT/blob/gh-pages/docs/static/images/render/2.jpg" width="300"> <img src="https://github.com/Necolizer/RM-PRT/blob/gh-pages/docs/static/images/render/4.jpg" width="300">
</div>

High-resolution rendering of the digital twin scene.

https://github.com/Necolizer/RM-PRT/assets/58028682/02fc70c6-a636-4fdc-90e5-4d6010bff0f2

Examples of robotic manipulation in our RM-PRT simulator.

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
## Joints range
```bash
joints_arrange = [
        [-36,30], 
        [-90,90], 
        [-45,45], 
        [-45,45], 
        [-180,180],
        [-45,36], 
        [-23,23], 

        # left hand
        [-180,36], 
        [-23,90], 
        [-90,90], 
        [-120,12], 
        [-90,90], 
        [-23,23], 
        [-36,23],

        #right hand
        [-180,36],
        [-90,23], 
        [-90,90],
        [-120,12], 
        [-90,90],
        [-23,23],
        [-23,36], 
    ]
```
## API using
Connect to the simulator
```bash
channel = grpc.insecure_channel('127.0.0.1:30001')  # FIXME
channel = grpc.insecure_channel(client,options=[
            ('grpc.max_send_message_length', 1024*1024*1024),
            ('grpc.max_receive_message_length', 1024*1024*1024)
        ])
stub=GrabSim_pb2_grpc.GrabSimStub(channel)
```
Init the world
```bash
initworld = stub.Init(GrabSim_pb2.Count(value = value)) # value present the number of scene 
```
Reset the scene
```bash
stub.Reset(GrabSim_pb2.ResetParams(sceneID=sceneID)) #sceneID start from 0
```
Get observation of the scene and show env info
```bash
scene = stub.Observe(GrabSim_pb2.SceneID(value=sceneID))
print('------------------show_env_info----------------------')
print(f"sceneID:{scene.sceneID}, location:{[scene.location.X, scene.location.Y]}, rotation:{scene.rotation}\n",
      f"joints number:{len(scene.joints)}\n")
```
Get the images
```bash
caremras=[GrabSim_pb2.CameraName.Head_Color,GrabSim_pb2.CameraName.Head_Depth]
action = GrabSim_pb2.CameraList(sceneID=sceneID, cameras=caremras)
images = stub.Capture(action).images
```
Do action
```bash
stub.Do(GrabSim_pb2.Action(sceneID=sceneID, action = GrabSim_pb2.Action.ActionType.WalkTo,values = [x, y, Yaw, q, v])) # walk
stub.Do(GrabSim_pb2.Action(sceneID=sceneID, action = GrabSim_pb2.Action.ActionType.RotateJoints,values = joints)) # changeJoints
stub.Do(GrabSim_pb2.Action(sceneID=sceneID, action=GrabSim_pb2.Action.Grasp,values=[0])) # control robot hand to grasp, 0 is left hand, 1 is right hand
stub.Do(GrabSim_pb2.Action(sceneID=sceneID, action=GrabSim_pb2.Action.Release,values=[0])) # release robot hand, 0 is left hand, 1 is right hand
```
Make Objects
```bash
obj_list = [
        GrabSim_pb2.ObjectList.Object(x=x, y=y, yaw=yaw, z=desk_h, type=obj_type),
    ]
scene = stub.MakeObjects(GrabSim_pb2.ObjectList(objects=obj_list, sceneID=sceneID))
```
Clean Objects
```bash
stub.CleanObjects(GrabSim_pb2.SceneID(value=sceneID))
```
