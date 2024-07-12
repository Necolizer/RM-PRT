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
from torchvision import transforms

sys.path.append('./')
sys.path.append('../')
sys.path.append('./feeders')

try:
    from . import GrabSim_pb2_grpc
    from . import GrabSim_pb2
    from .Env.simUtils import SimServer
except:
    from Env import GrabSim_pb2_grpc
    from Env import GrabSim_pb2
    from Env.simUtils import SimServer

def Resize(mat,img_size=256):
    if isinstance(img_size,int):
        img_size = (img_size, img_size)
    if mat.dtype !=np.uint8:
        mat = (mat*255).astype(np.uint8)
    mat = Image.fromarray(mat, mode='RGB')
    mat = mat.resize(img_size)
    mat = np.array(mat)
    mat = 1.0 * mat
    mat = mat/255.0
    return mat

def find_img(frame,img_size=256):
    if 'img'+str(img_size) in frame.keys():
        return frame['img'+str(img_size)]
    return Resize(frame['img'],img_size)

actuatorRanges=np.array([[-30.00006675720215, 31.65018653869629],
 [-110.00215911865234, 30.00006675720215],
 [-90.00020599365234, 90.00020599365234],
 [-5.729577541351318, 64.74422454833984],
 [-5.729577541351318, 64.74422454833984],
 [-5.729577541351318, 64.74422454833984],
 [-5.729577541351318, 64.74422454833984],
 [-159.9984588623047, 129.99838256835938],
 [-15.000033378601074, 150.00035095214844],
 [-5.729577541351318, 64.74422454833984],
 [-30.00006675720215, 30.00006675720215],
 [-30.00006675720215, 30.00006675720215],
 [-90.00020599365234, 90.00020599365234],
 [-45.00010299682617, 58.49898910522461],
 [-39.999900817871094, 39.999900817871094],
 [-90.00020599365234, 90.00020599365234],
 [-45.00010299682617, 45.00010299682617],
 [-30.00006675720215, 30.00006675720215],
 [-90.00020599365234, 90.00020599365234],
 [-110.00215911865234, 30.00006675720215],
 [-90.00020599365234, 90.00020599365234],
 [-5.729577541351318, 64.74422454833984],
 [-5.729577541351318, 64.74422454833984],
 [-5.729577541351318, 64.74422454833984],
 [-5.729577541351318, 64.74422454833984],
 [-129.99838256835938, 159.9984588623047],
 [-150.00035095214844, 15.000033378601074],
 [-5.729577541351318, 64.74422454833984],
 [-30.00006675720215, 30.00006675720215],
 [-30.00006675720215, 30.00006675720215],
 [-90.00020599365234, 90.00020599365234]])


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        np_img = np.array(img)
        noise = np.random.normal(self.mean, self.std, np_img.shape).astype(np.float32)
        noisy_img = np_img + noise
        noisy_img = np.clip(noisy_img, 0, 255)  # 限制值的范围在[0, 255]
        noisy_img = Image.fromarray(noisy_img.astype(np.uint8))  # 转换回PIL图像
        return noisy_img

class Feeder(Dataset):
    objs = SimServer.objs
    def __init__(self, data_path, instructions_path, control='joint', history_len=3, instructions_level=[3],  sample_frame=100, bin=256, img_size=256, data_size=None,dataAug=True):
        self.data_path = data_path
        self.instructions_path = instructions_path
        
        self.control = control
        self.instructions_level = instructions_level
        self.history_len = history_len
        self.sample_frame = sample_frame
        self.bin = bin
        self.img_size = img_size
        self.dataAug = dataAug
        print('dataAug',dataAug)
        self.data_transforms = transforms.Compose([
            # transforms.RandomRotation(degrees=2),
            # transforms.Resize((256, 256)),
            # transforms.RandomCrop(224),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            # AddGaussianNoise(0., 1.),
            # transforms.ToTensor()
        ])
        # self.channel = grpc.insecure_channel(self.channel_name,options=[
        #     ('grpc.max_send_message_length', 1024*1024*1024),
        #     ('grpc.max_receive_message_length', 1024*1024*1024)
        # ])
        # self.sim_client = GrabSim_pb2_grpc.GrabSimStub(self.channel)

        # self.initworld = self.sim_client.Init(GrabSim_pb2.Count(value=1))

        self.id2name = {0: 'Mug', 1: 'Banana', 2: 'Toothpaste', 3: 'Bread', 4: 'Softdrink',5: 'Yogurt',6: 'ADMilk',7: 'VacuumCup',8: 'Bernachon',9: 'BottledDrink',10: 'PencilVase',11: 'Teacup',12: 'Caddy',13: 'Dictionary',14: 'Cake',15: 'Date',16: 'Stapler',17: 'LunchBox',18: 'Bracelet',19: 'MilkDrink',20: 'CocountWater',21: 'Walnut',22: 'HamSausage',23: 'GlueStick',24: 'AdhensiveTape',25: 'Calculator',26: 'Chess',27: 'Orange',28: 'Glass',29: 'Washbowl',30: 'Durian',31: 'Gum',32: 'Towl',33: 'OrangeJuice',34: 'Cardcase',35: 'RubikCube',36: 'StickyNotes',37: 'NFCJuice',38: 'SpringWater',39: 'Apple',40: 'Coffee',41: 'Gauze',42: 'Mangosteen',43: 'SesameSeedCake',44: 'Glove',45: 'Mouse',46: 'Kettle',47: 'Atomize',48: 'Chips',49: 'SpongeGourd',50: 'Garlic',51: 'Potato',52: 'Tray',53: 'Hemomanometer',54: 'TennisBall',55: 'ToyDog',56: 'ToyBear',57: 'TeaTray',58: 'Sock',59: 'Scarf',60: 'ToiletPaper',61: 'Milk',62: 'Soap',63: 'Novel',64: 'Watermelon',65: 'Tomato',66: 'CleansingFoam',67: 'CocountMilk',68: 'SugarlessGum',69: 'MedicalAdhensiveTape',70: 'SourMilkDrink',71: 'PaperCup',72: 'Tissue'}
        self.name2id = {v: k for k, v in self.id2name.items()}

        # self.data = pd.read_csv(self.csv_path, encoding='ISO-8859-1')
        self.data,self.instructions=self.read_data(self.data_path,self.instructions_path,self.instructions_level,data_size)


    def read_data(self,data_paths,instructions_path,instructions_level,data_size=None):
        # 获取文件夹下的所有文件和子文件夹
        total_files=[]
        for path in data_paths:
            all_items = os.listdir(path)
            # 过滤出文件
            files = [os.path.join(path, item) for item in all_items if os.path.isfile(os.path.join(path, item)) and item.endswith('pkl')]
            total_files+=files
        if isinstance(data_size,int):
            total_files=total_files[:data_size]
        elif isinstance(data_size,float):
            data_size = int(len(total_files)*data_size)
            total_files=total_files[:data_size]
            
        with open(instructions_path,'rb') as f:
            instructions = pickle.load(f)
        # df=df[df['level']==instructions_level]
        
        return total_files, instructions

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
        # 取数据
        file = self.data[index]
        with open(file,'rb') as f:
            sample=pickle.load(f)
            
        x,y,z=sample['robot_location']
        if 'event' not in sample.keys():
            event = 'graspTargetObj'
        else:
            event = sample['event']
        self.targetObjID=sample['targetObjID']
        self.targetObj = self.objs[self.objs.ID==self.targetObjID].Name.values[0]
        level=random.choice(self.instructions_level)
        target = self.objs[self.objs.ID == sample['targetObjID']].iloc[0]

        other_id = []
        for obj in sample['objList'][:]:
            if obj[0]!=self.targetObjID:
                other_id.append(obj[0])
        other = self.objs[self.objs.ID.isin(other_id)]
        if target.Name not in self.instructions.keys():
            level=0
        if level >0:
            instr = self.instructions[target.Name]
            way = random.choice(list(instr.keys()))
            instr = instr[way]
            if way=='descriptions':
                can_att = ['name', 'color', 'shape', 'application', 'other']
                if target.Name in other.Name.values:
                    can_att.remove('name')
                if target.Color in other.Color.values:
                    can_att.remove('color')
                if target.Shape in other.Shape.values:
                    can_att.remove('shape')
                if target.Application in other.Application.values:
                    can_att.remove('application')
                if target.Other in other.Other.values:
                    can_att.remove('other')    
                if (target.Size > other.Size.values+1).all():
                    can_att.append('largest')
                if (target.Size < other.Size.values-1).all():
                    can_att.append('smallest')
            else:
                origin_att = ['left','right','close','distant','left front','front right','behind left','behind rght']
                assert sample['targetObjID'] == sample['objList'][0][0]
                loc1 = sample['objList'][0][1:3]
                for obj in sample['objList'][1:] :
                    loc2 = obj[1:3]
                    can_att = []
                    if loc1[1]-loc2[1]>5:
                        can_att.append('left')
                    if loc1[1]-loc2[1]<-5:
                        can_att.append('right')
                    if loc1[0]-loc2[0]>5:
                        can_att.append('close')
                    if loc1[0]-loc2[0]<-5:
                        can_att.append('distant')   
                    if loc1[1]-loc2[1]>5 and loc1[0]-loc2[0]<-5:
                        can_att.append('left front') 
                    if loc1[1]-loc2[1]<-5 and loc1[0]-loc2[0]<-5:
                        can_att.append('front right') 
                    if loc1[1]-loc2[1]>5 and loc1[0]-loc2[0]>5:
                        can_att.append('behind left')     
                    if loc1[1]-loc2[1]<-5 and loc1[0]-loc2[0]>5:
                        can_att.append('behind rght')  
                    origin_att = set(origin_att).intersection(set(can_att))
                    origin_att = list(origin_att)
                can_att = origin_att
            have_att = set(instr.keys())
            can_att = list(set(can_att).intersection(have_att))
        if level==0 or len(can_att)==0:
            if event == 'graspTargetObj':
                instr = 'pick a '+self.targetObj
            elif event == 'placeTargetObj':
                instr = 'place ' + self.targetObj
            elif event == 'moveNear':
                instr = 'moveNear ' + self.targetObj
            elif event == 'knockOver':
                instr = 'knock ' + self.targetObj +' over'
            elif event == 'pushFront':
                instr = 'push ' + self.targetObj + ' front'
            elif event == 'pushLeft':
                instr = 'push ' + self.targetObj + ' left'
            elif event == 'pushRight':
                instr = 'push ' + self.targetObj + ' right'
        else:
            att = random.choice(can_att)
            instr = instr[att]
            if level==1:
                instr = instr['origin']
            else:
                instr = random.choice(instr['human'])
        imgs = []
        states = []
        actions=[]
        next_imgs = []
        now_joints = [0]*14 + [36.0,-40.0,40.0,-90.0,5.0,0.0,0.0]
        last_action = np.array(sample['initLoc'])

        # # 改变instr
        # target_index = sample['target_obj_index']-1
        # other_index = 1 if target_index==0 else 0
        # if sample['objList'][target_index][2]>sample['objList'][other_index][2]:
        #     instr='0'
        # else:
        #     instr='1'

        ## 临时改变动作
        for _,frame in enumerate(sample['trajectory'][:-1]):
            # if  _>10:
            #     break
            # if frame['action'][-1]==1:
            #     break
            # each frame
            
            imgs.append(frame['img']) # numpy array
            sensors=frame['state']['sensors']
            state = np.array(sensors[3]['data'])
            state[:3]-=np.array([x,y,z])
            # for sensor in sensors[4:]:
            #     if 'right' in sensor['name']:
            #         state = np.concatenate([state,np.array(sensor['data'])-np.array([x,y,z])])
            state[:]/=np.array([50,30,40])
            states.append(state)

            if self.control == 'ee':
                if frame['action'][5]>=5:
                    frame['action'][5]=0
                if len(frame['action'])==6:
                    frame['action'] = [*frame['action'],0,0]
                if frame['action'][5]>=1:
                    frame['action'][5] = 1
                def discretize_value(value, num_bins=256):
                    # 确保值在[-1, 1]范围内
                    value_clipped = np.clip(value, -1, 1)
                    # 将[-1, 1]区间映射到[0, num_bins-1]
                    discretized = np.round((value_clipped + 1) / 2 * (num_bins - 1))
                    return discretized
                frame['action'][:6] = discretize_value(frame['action'][:6])
                action = np.array(frame['action'], dtype=np.float64)
            else:
                before_joints = frame['state']['joints']
                before_joints = [joint['angle'] for joint in before_joints]
                after_joints = sample['trajectory'][_+1]['state']['joints'] # frame['after_state']['joints']
                after_joints = [joint['angle'] for joint in after_joints]
                map_id=[0,1,2,3,6,9,12,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,
                    33,36,39,42,43,44,46,47,48]
                before_joints=[before_joints[id] for id in map_id]
                after_joints=[after_joints[id] for id in map_id]
                joints = (np.array(after_joints)-np.array(before_joints)) /(actuatorRanges[:,1]-actuatorRanges[:,0])*50
                action = np.array([joints[-12],joints[-11],joints[-6],joints[-5],frame['action'][-1]],dtype=np.float64)

            # # 只要求输出物体是左还是右
            # target_index = sample['target_obj_index']-1
            # other_index = 1 if target_index==0 else 0
            # if sample['objList'][target_index][2]>sample['objList'][other_index][2]:
            #     action = np.array([0]*8)
            # else:
            #     action = np.array([1]*8)
                
            actions.append(action)
            last_action = frame['action']
            
        # print('states',states)
        # print('actions',actions)
        # next_imgs = imgs[1:]+[sample['trajectory'][-1]['img']]
        next_imgs = [sample['trajectory'][-1]['img']] * len(imgs)
        tmp_imgs=[]
        tmp_states=[]
        for i in range(len(imgs)):
            if i+1>=self.history_len:
                tmp_imgs.append(imgs[i-self.history_len+1:i+1])
                tmp_states.append(states[i-self.history_len+1:i+1])
            else:
                prefix = self.history_len-(i+1)
                tmp_imgs.append(imgs[:1]*prefix+imgs[:i+1])
                tmp_states.append(states[:1]*prefix+states[:i+1])
            tmp_imgs[-1]=np.array(tmp_imgs[-1])
            tmp_states[-1]=np.array(tmp_states[-1])
        imgs=tmp_imgs
        states=tmp_states 
        # Sample N frames per trajectory
        frame_num = len(actions)
        if frame_num < self.sample_frame:
            while len(actions)<self.sample_frame:
                actions = actions.copy() + actions
                imgs = imgs.copy() + imgs
                states = states.copy() + states
                next_imgs = next_imgs.copy() + next_imgs
                # new_states = new_states.copy() + new_states
            actions = actions[-self.sample_frame:]
            imgs = imgs[-self.sample_frame:]
            states = states[-self.sample_frame:] 
            next_imgs = next_imgs[-self.sample_frame:] 
            # new_states = new_states[:self.sample_frame]
        else:
            zip_list = random.sample(list(zip(actions,imgs,states,next_imgs)), self.sample_frame)
            actions, imgs, states, next_imgs = zip(*zip_list)
            actions = list(actions)
            imgs = list(imgs)
            states = list(states)
            next_imgs = list(next_imgs)
            # new_states = list(new_states)

        if self.dataAug:
            for i in range(len(imgs)):
                for j in range(len(imgs[i])):
                    img = (imgs[i][j]*255).astype(np.uint8)  # 转换为整数类型
                    img = Image.fromarray(img)
                    img = self.data_transforms(img)
                    img = np.array(img)
                    imgs[i][j] = img/255

        # pre_joints=[initJoints.tolist()]+joints[:-1]

        frame_num = self.sample_frame

        # instr = [instr] * self.sample_frame
        actions_tok = [torch.from_numpy(action) for action in actions]
        imgs_tensor = [torch.from_numpy(img) for img in imgs]
        states_tensor = [torch.from_numpy(state) for state in states]
        next_imgs_tensor = [torch.from_numpy(img) for img in next_imgs]

        try:
            imgs_tensor = torch.stack(imgs_tensor, dim=0)
        except:
            imgs_tensor = torch.stack(imgs_tensor, dim=0)
        try:
            actions_tok = torch.stack(actions_tok, dim=0)
        except:
            print('actions_tok',actions_tok)
            actions_tok = torch.stack(actions_tok, dim=0)
        try:
            states_tensor = torch.stack(states_tensor, dim=0)
        except:
            print('states_tensor',states_tensor)
            states_tensor = torch.stack(states_tensor, dim=0)
        next_imgs_tensor = torch.stack(next_imgs_tensor, dim=0)
        # print('data[index], instr:',self.data[index],instr,sample['from_file'])
        # print('states_tensor',states_tensor)
        # print('imgs_tensor',imgs_tensor.shape)
        # print('imgs_tensor',self.data[index],sample['from_file'],instr,actions_tok[2])
        # instr = str(actions_tok[2][-1])
        return imgs_tensor, instr, actions_tok, states_tensor, next_imgs_tensor, index
    