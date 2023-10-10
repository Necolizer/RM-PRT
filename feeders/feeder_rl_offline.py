import os
import pickle

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import math
import random
import grpc
import pandas as pd
import numpy as np
import re
from PIL import Image
import time
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('./feeders')


try:
    import GrabSim_pb2_grpc
    import GrabSim_pb2
except:
    import os;
    os.chdir("./python/")
    import GrabSim_pb2_grpc
    import GrabSim_pb2

def initJointsArrange():
    joints_arrange = [
        [-36,30], #全身前后（摇摆，包括腿）
        [-90,90], #躯体左右 (旋转，不是摇摆)
        [-45,45], #上半身前后（摇摆，不包括腿）
        [-45,45], #上半身左右（摇摆，不包括腿）
        [-180,180], #头旋转（扭脖子）
        [-45,36], #头前后（点头）
        [-23,23], #头左右 (摇头) 

        #左手
        [-180,36], #肩关节，整条手臂前后
        [-23,90], #肩关节，整条手臂左右
        [-90,90], #肘关节，小臂旋转
        [-120,12], #肘关节，小臂前后
        [-90,90], #腕关节，手掌旋转 [-90,90],
        [-23,23], #腕关节，手掌前后
        [-36,23], #腕关节，手掌左右

        #右手
        [-180,36], #肩关节，整条手臂前后
        [-90,23], #肩关节，整条手臂左右
        [-90,90], #肘关节，小臂旋转
        [-120,12], #肘关节，小臂前后
        [-90,90], #腕关节，手掌旋转
        [-23,23], #腕关节，手掌前后
        [-23,36], #腕关节，手掌左右

        #grasp
        [0,1], # 是否抓取，此为离散值
    ]
    # joints_arrange=joints_arrange/np.pi*180
    return np.array(joints_arrange,dtype=np.float32)

class Feeder(Dataset):
    def __init__(self, data_path, instructions_path, instructions_level='3',  sample_frame=100, bin=256):
        self.data_path = data_path
        self.instructions_path = instructions_path
        
        levels = instructions_level.split(',')
        self.instructions_level = [int(level) for level in levels]
        self.sample_frame = sample_frame
        self.bin = bin

        # self.channel = grpc.insecure_channel(self.channel_name,options=[
        #     ('grpc.max_send_message_length', 1024*1024*1024),
        #     ('grpc.max_receive_message_length', 1024*1024*1024)
        # ])
        # self.sim_client = GrabSim_pb2_grpc.GrabSimStub(self.channel)

        # self.initworld = self.sim_client.Init(GrabSim_pb2.Count(value=1))

        self.id2name = {0: 'Mug', 1: 'Banana', 2: 'Toothpaste', 3: 'Bread', 4: 'Softdrink',5: 'Yogurt',6: 'ADMilk',7: 'VacuumCup',8: 'Bernachon',9: 'BottledDrink',10: 'PencilVase',11: 'Teacup',12: 'Caddy',13: 'Dictionary',14: 'Cake',15: 'Date',16: 'Stapler',17: 'LunchBox',18: 'Bracelet',19: 'MilkDrink',20: 'CocountWater',21: 'Walnut',22: 'HamSausage',23: 'GlueStick',24: 'AdhensiveTape',25: 'Calculator',26: 'Chess',27: 'Orange',28: 'Glass',29: 'Washbowl',30: 'Durian',31: 'Gum',32: 'Towl',33: 'OrangeJuice',34: 'Cardcase',35: 'RubikCube',36: 'StickyNotes',37: 'NFCJuice',38: 'SpringWater',39: 'Apple',40: 'Coffee',41: 'Gauze',42: 'Mangosteen',43: 'SesameSeedCake',44: 'Glove',45: 'Mouse',46: 'Kettle',47: 'Atomize',48: 'Chips',49: 'SpongeGourd',50: 'Garlic',51: 'Potato',52: 'Tray',53: 'Hemomanometer',54: 'TennisBall',55: 'ToyDog',56: 'ToyBear',57: 'TeaTray',58: 'Sock',59: 'Scarf',60: 'ToiletPaper',61: 'Milk',62: 'Soap',63: 'Novel',64: 'Watermelon',65: 'Tomato',66: 'CleansingFoam',67: 'CocountMilk',68: 'SugarlessGum',69: 'MedicalAdhensiveTape',70: 'SourMilkDrink',71: 'PaperCup',72: 'Tissue'}
        self.name2id = {v: k for k, v in self.id2name.items()}

        # self.data = pd.read_csv(self.csv_path, encoding='ISO-8859-1')
        self.data,self.instructions=self.read_data(self.data_path,self.instructions_path,self.instructions_level)

        self.joint_boundary = initJointsArrange()
        self.joint_range = self.joint_boundary[:,1] - self.joint_boundary[:,0]

    def read_data(self,paths,instructions_path,instructions_level):
        # 获取文件夹下的所有文件和子文件夹
        total_files=[]
        for path in paths.split(','):
            all_items = os.listdir(path)

            # 过滤出文件
            files = [os.path.join(path, item) for item in all_items if os.path.isfile(os.path.join(path, item)) and item.endswith('pkl')]
            
            total_files+=files
            
        df=pd.read_csv(instructions_path)
        # df=df[df['level']==instructions_level]
        
        return total_files, df

    def gen_scene_toSim(self, scene):
        message = self.sim_client.CleanObjects(GrabSim_pb2.SceneID(value=self.sceneID))

        for obj in scene:
            name = re.sub(r"\d", "", obj[0])
            id = self.name2id[name]
            x = -2150.0+obj[2]
            y = -1350.0+obj[1]
            h = 84

            obj_list = [GrabSim_pb2.ObjectList.Object(x=x, y=y, yaw=0, z=h, type=id)]
            scene = self.sim_client.MakeObjects(GrabSim_pb2.ObjectList(objects=obj_list, sceneID=self.sceneID))
    
    def getCamera(self, caremras=[GrabSim_pb2.CameraName.Head_Color]):
        #caremras=[GrabSim_pb2.CameraName.Head_Color,GrabSim_pb2.CameraName.Head_Depth]
        action = GrabSim_pb2.CameraList(sceneID=self.sceneID, cameras=caremras)
        images = self.sim_client.Capture(action).images
        rgbd=[]
        for im in images:
            mat = np.frombuffer(im.data,dtype=im.dtype).reshape((im.height, im.width, im.channels))
            if(im.channels == 3):
                mat = Image.fromarray(mat, mode='RGB')
                mat = mat.resize((224, 224))
                mat = np.array(mat)
                mat = 1.0 * mat
                mat = mat/255.0
                rgbd.append(mat)
            else:
                t=150
                mat = 1.0 * mat
                mat[mat>t]=t
                mat=(mat/t*255).reshape((im.height, im.width)).astype(np.uint8)
                mat = Image.fromarray(mat,mode='L')
                mat = mat.resize((224, 224))
                mat = np.array(mat).reshape((224,224,1))
                mat = 1.0 * mat/255
                rgbd.append(mat)
        rgbd = np.concatenate(rgbd, axis=-1)
        return rgbd
    
    def reset(self):
        return self.sim_client.Reset(GrabSim_pb2.ResetParams(sceneID=self.sceneID))
    
    def getScene(self):
        self.scene=self.sim_client.Observe(GrabSim_pb2.SceneID(value=self.sceneID))
        return self.scene
    
    def getObjIDLocation(self,objID):
        self.scene=self.getScene()
        targetObjLoc=self.scene.objects[objID].location
        targetObjLoc=np.array([targetObjLoc.X,targetObjLoc.Y,targetObjLoc.Z])
        return targetObjLoc
    
    def getfingerLocation(self):
        self.scene=self.getScene()
        # print(self.scene)
        # print(self.scene.location)
        fingers=self.scene.fingers
        fingersLoc = []
        for finger in fingers:
            Loc=finger.location[1]
            fingersLoc.append([Loc.X, Loc.Y, Loc.Z])
        return np.array(fingersLoc)

    def getState(self):
        self.scene=self.getScene()
        state=[self.scene.location.X,self.scene.location.Y,self.scene.rotation.Yaw]
        for i in range(len(self.scene.joints)):
            state.append(self.scene.joints[i].angle)
        # id,Loc=self.get_nearest_obj()
        Loc=self.getObjIDLocation(self.targetid)
        
        finger = self.getfingerLocation()[5+3-1]
        # diff = [Loc[i]-finger[i] for i in range(3)]
        diff = [finger[0]-state[0], finger[1]-state[1], finger[2]]
        state.extend(diff)

        state.append(int(self.last_grasp))

        return np.array(state)

    def getObjLocation(self,objName):
        self.scene=self.getScene()
        locations=[]
        for i,obj in enumerate(self.scene.objects):
            if obj.name==objName or (objName is None):
                location = obj.location
                locations.append([i, location.X, location.Y, location.Z])
        return np.array(locations)
    
    def get_nearest_obj(self,targetObj=None):
        ObjLocs=self.getObjLocation(targetObj)
        assert len(ObjLocs)>0

        fingers=self.getfingerLocation()

        fingerR3Loc = (fingers[5+1-1]+fingers[5+5-1])/2
        tcp_to_obj_pos = ObjLocs[:,1:]-fingerR3Loc
        nearest_obj_id = np.linalg.norm(tcp_to_obj_pos,axis=1).argmin()
        

        return int(ObjLocs[nearest_obj_id][0]), ObjLocs[nearest_obj_id][1:]
    
    def action_tokenization(self, action):
        # 依照RT-1原文进行tokenization
        # 21 joints + 1 grasp or not
        action_numpy = np.array(action)
        bin_target=action_numpy/(self.joint_boundary[:,1]-self.joint_boundary[:,0])*50
        bin_target=torch.Tensor(bin_target)
        
        # bin_target = (action_numpy-self.joint_boundary[:,0]) // (self.joint_range/self.bin)
        # bin_target = torch.clamp(torch.tensor(bin_target, dtype=torch.int64), min=0, max=self.bin-1)
        # if torch.max(bin_target) >= self.bin or torch.min(bin_target) < 0:
        #     print(bin_target)
        #     exit(1)
        return bin_target

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def genObjwithLists(self, sim_client,sceneID,objList):
        for x,y,z,yaw,type in objList:
            obj_list = [GrabSim_pb2.ObjectList.Object(x=x, y=y, yaw=yaw, z=z, type=type)]
            # obj_list = [GrabSim_pb2.ObjectList.Object(X=ginger_loc[0] + x_rand, Y=ginger_loc[1] + y_rand, Yaw=yaw_rand, Z=h, type=type_rand)]
            scene = sim_client.MakeObjects(GrabSim_pb2.ObjectList(objects=obj_list, sceneID=sceneID))
    
    def __getitem__(self, index):
        # Output:
        # imgs_tensor: Torch tensor [F(self.sample_frame), H(224), W(224), C(4)]
        # instr: str * 1
        # joints_tok: Torch tensor [F(self.sample_frame), ActionNum(22)]
        # pre_joints_tok: Torch tensor [F(self.sample_frame), ActionNum(22)]
        # index: int * 1

        # 仿真环境
        # scene=self.reset()
        # # self.gen_scene_toSim(scene)
        # time.sleep(1)

        # 取数据
        file = self.data[index]
        with open(file,'rb') as f:
            sample=pickle.load(f)
            
        x,y=sample['robot location']['x'],sample['robot location']['y']
        
        # scene = self.sim_client.CleanObjects(GrabSim_pb2.SceneID(value=self.sceneID))

        objList=sample['objList']
        self.targetObj=sample['targetObj']
        level=random.choice(self.instructions_level)
        if level>2:
            df=self.instructions[self.instructions['level']==level]
            df=self.instructions[self.instructions['object']==self.targetObj]
            instr = df.sample()['instruction'].values[0]
        else:
            instr = sample['instruction']

        imgs = []
        states = []
        joints=[]
        
        now_joints = [0]*14 + [36.0,-40.0,40.0,-90.0,5.0,0.0,0.0]
        # for frame in random.choice([sample['trajectory'][:10]+sample['trajectory'][10::2],sample['trajectory'][:11]+sample['trajectory'][11::2]]):
        for frame in sample['trajectory']: 
            # each frame
            
            imgs.append(frame['img']) # numpy array
            states.append(frame['state'])
            
            # now_joints=[i+j for i,j in zip(now_joints,frame['action'])]
            joints.append(frame['action'])
            

        # Sample N frames per trajectory
        frame_num = len(joints)
        
        if frame_num < self.sample_frame:
            while len(joints)<self.sample_frame:
                joints = joints.copy() + joints
                imgs = imgs.copy() + imgs
                states = states.copy() + states
                # new_states = new_states.copy() + new_states
            joints = joints[-self.sample_frame:]
            imgs = imgs[-self.sample_frame:]
            states = states[-self.sample_frame:] 
            # new_states = new_states[:self.sample_frame]
        elif frame_num > self.sample_frame:
            zip_list = random.sample(list(zip(joints,imgs,states)), self.sample_frame)
            joints, imgs, states = zip(*zip_list)
            joints = list(joints)
            imgs = list(imgs)
            states = list(states)
            # new_states = list(new_states)

        # pre_joints=[initJoints.tolist()]+joints[:-1]

        frame_num = self.sample_frame

        # instr = [instr] * self.sample_frame

        # 按照RT-1原文 tokenization
        joints_tok = []
        imgs_tensor = [torch.from_numpy(img) for img in imgs]
        states_tensor = [torch.from_numpy(state) for state in states]
        # new_states_tensor = [torch.from_numpy(state) for state in new_states]
        
        # action tokenization
        for i in range(self.sample_frame):
            if i < self.sample_frame-1:
                # + not grasp (0)
                j_t = joints[i] + [0.] # 22
                joints_tok.append(self.action_tokenization(np.array(j_t)))
            else:
                # + grasp (1)
                j_t = joints[i] + [1./self.bin] # 22
                joints_tok.append(self.action_tokenization(np.array(j_t)))


        imgs_tensor = torch.stack(imgs_tensor, dim=0)
        joints_tok = torch.stack(joints_tok, dim=0)
        states_tensor = torch.stack(states_tensor, dim=0)
        return imgs_tensor, instr, joints_tok[:,-8:-1], states_tensor, index
    