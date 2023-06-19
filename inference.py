# -*- coding: utf-8 -*-

import argparse
import gym
from gym import spaces
import numpy as np


import time
import random
from google.protobuf import message
import grpc
try:
    import GrabSim_pb2_grpc
    import GrabSim_pb2
except:
    import os;
    os.chdir("./python/")
    import GrabSim_pb2_grpc
    import GrabSim_pb2

from Predictor import predictor_construct

# 参数

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str,default="localhost:30001")
parser.add_argument("--action_nums", type=int,default=8)
args, opts = parser.parse_known_args()

# train22 modify Extractor
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,"
from SimEnv4 import SimEnv
client=args.host
save_path='./Log/logs8/'
deterministic=True
action_nums=args.action_nums
bins = 128
abs_distance=12
level=1
max_steps = 160
use_mask=False
use_image = True
channel = grpc.insecure_channel(client,options=[
            ('grpc.max_send_message_length', 1024*1024*1024),
            ('grpc.max_receive_message_length', 1024*1024*1024)
        ])

print('start')
#定义环境、DQN agent
# env = gym.make('CartPole-v1')
# 创建EvalCallback对象
stub=GrabSim_pb2_grpc.GrabSimStub(channel)
initworld = stub.Init(GrabSim_pb2.Count(value = 2))
time.sleep(5)

scene = stub.Observe(GrabSim_pb2.SceneID(value=0))
print('------------------show_env_info----------------------')
print(
    f"sceneID:{scene.sceneID}, location:{[scene.location.X, scene.location.Y]}, rotation:{scene.rotation}\n",
    f"joints number:{len(scene.joints)}, fingers number:{len(scene.fingers)}\n", f"objects number: {len(scene.objects)}\n"
    f"velocity:{scene.velocity}, rotation:{scene.rotating}, timestep:{scene.timestep}\n"
    # f"timestamp:{scene.timestamp}, collision:{scene.collision}, info:{scene.info}"
    )


env=SimEnv(client,0,deterministic=deterministic,action_nums=action_nums,bins=bins,abs_distance=abs_distance,use_image=use_image,max_steps = max_steps,level=level)

predictor=predictor_construct()

num_test=100
from tqdm import tqdm  
is_success=False  

obs=env.reset()
for i in range(max_steps):
    obs['head_rgb']=obs['head_rgb'].astype(np.float32)
    obs['state']=obs['state'].astype(np.float32)
    action = predictor.predict(obs)[0]
    obs, rewards, dones, info = env.step(action)
    if info['is_success']:
        is_success=True
    if dones:
        break
if is_success:
    print('Successfully grasp')
else:
    print('Fail to grasp')