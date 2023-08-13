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
    from Env import GrabSim_pb2_grpc
    from Env import GrabSim_pb2
except:
    import os;
    os.chdir("./python/")
    import GrabSim_pb2_grpc
    import GrabSim_pb2
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn

# seed = 0
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# cudnn.benchmark = False

import gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import robotic_transformer_pytorch.Extractor as CustomExtractor
from Wrapper import ContinuousTaskWrapper

# 参数

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str,default="localhost:30001")
parser.add_argument("--action_nums", type=int,default=8)
args, opts = parser.parse_known_args()
print(args.host)

# train22 modify Extractor
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,"
from Env.SimEnv4 import SimEnv
client=args.host
save_path='./Log/logs/'
deterministic=True
action_nums=args.action_nums
bins = 50
abs_distance=12
level=1
mode='grasping'

use_image = True
rollout_steps = 3200
max_steps = 80
learning_rate=3e-4
gamma = 0.99
freq = rollout_steps*10
batch_size = 400
target_kl=None
net_arch=[ 256,128]
channel = grpc.insecure_channel(client,options=[
            ('grpc.max_send_message_length', 1024*1024*1024),
            ('grpc.max_receive_message_length', 1024*1024*1024)
        ])

print('start')

stub=GrabSim_pb2_grpc.GrabSimStub(channel)
initworld = stub.Init(GrabSim_pb2.Count(value = 2))
# initworld = stub.Init(GrabSim_pb2.NUL())
# print(stub.AcquireAvailableMaps(GrabSim_pb2.NUL()))
map_id=4
# initworld = stub.SetWorld(GrabSim_pb2.BatchMap(count=2, mapID=map_id))
time.sleep(5)

scene = stub.Observe(GrabSim_pb2.SceneID(value=0))
print('------------------show_env_info----------------------')
print(
    f"sceneID:{scene.sceneID}, location:{[scene.location.X, scene.location.Y]}, rotation:{scene.rotation}\n",
    f"joints number:{len(scene.joints)}, fingers number:{len(scene.fingers)}\n", f"objects number: {len(scene.objects)}\n"
    f"velocity:{scene.velocity}, rotation:{scene.rotating}, timestep:{scene.timestep}\n"
    # f"timestamp:{scene.timestamp}, collision:{scene.collision}, info:{scene.info}"
    )



env=SimEnv(client,0,deterministic=deterministic,action_nums=action_nums,bins=bins,abs_distance=abs_distance,use_image=use_image,max_steps = max_steps,level=level,mode=mode)

model = PPO.load("model.zip", env=env) 
#可视化

num_test=100
env.training=True
env.level=1


from tqdm import tqdm    

obs=env.reset()


for test_index in range(num_test):
    is_success=False
    for _ in range(150):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if info['is_success']:
            is_success=True
        if dones:
            break
    if is_success:
        print('Success')
    else:
        print('Failed')