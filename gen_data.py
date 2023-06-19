import time
import random
import math
import pickle

try:
    import GrabSim_pb2_grpc
    import GrabSim_pb2
except:
    import os;
    os.chdir("./python/")
    import GrabSim_pb2_grpc
    import GrabSim_pb2

import grpc
import numpy as np
import pandas as pd
# import open3d as o3d
# import matplotlib.pyplot as plt

from gen_data import *

dis_ = 20 # 两个obj之间的最小距离
name_type = {0: 'Mug',
 1: 'Banana',
 2: 'Toothpaste',
 3: 'Bread',
 4: 'Softdrink',
 5: 'Yogurt',
 6: 'ADMilk',
 7: 'VacuumCup',
 8: 'Bernachon',
 9: 'BottledDrink',
 10: 'PencilVase',
 11: 'Teacup',
 12: 'Caddy',
 13: 'Dictionary',
 14: 'Cake',
 15: 'Date',
 16: 'Stapler',
 17: 'LunchBox',
 18: 'Bracelet',
 19: 'MilkDrink',
 20: 'CocountWater',
 21: 'Walnut',
 22: 'HamSausage',
 23: 'GlueStick',
 24: 'AdhensiveTape',
 25: 'Calculator',
 26: 'Chess',
 27: 'Orange',
 28: 'Glass',
 29: 'Washbowl',
 30: 'Durian',
 31: 'Gum',
 32: 'Towl',
 33: 'OrangeJuice',
 34: 'Cardcase',
 35: 'RubikCube',
 36: 'StickyNotes',
 37: 'NFCJuice',
 38: 'SpringWater',
 39: 'Apple',
 40: 'Coffee',
 41: 'Gauze',
 42: 'Mangosteen',
 43: 'SesameSeedCake',
 44: 'Glove',
 45: 'Mouse',
 46: 'Kettle',
 47: 'Atomize',
 48: 'Chips',
 49: 'SpongeGourd',
 50: 'Garlic',
 51: 'Potato',
 52: 'Tray',
 53: 'Hemomanometer',
 54: 'TennisBall',
 55: 'ToyDog',
 56: 'ToyBear',
 57: 'TeaTray',
 58: 'Sock',
 59: 'Scarf',
 60: 'ToiletPaper',
 61: 'Milk',
 62: 'Soap',
 63: 'Novel',
 64: 'Watermelon',
 65: 'Tomato',
 66: 'CleansingFoam',
 67: 'CocountMilk',
 68: 'SugarlessGum',
 69: 'MedicalAdhensiveTape',
 70: 'SourMilkDrink',
 71: 'PaperCup',
 72: 'Tissue'}

class obj:
    def __init__(self, objname, objid, x, y, yaw):
        self.objname = objname
        self.objid = objid
        self.x = x
        self.y = y
        self.yaw = yaw

train_list=list(name_type.keys())
ungrasp_list=[64,18]
unseen_obj_list=[46, 17, 26, 34, 41, 52, 3, 48, 25, 9, 67, 4, 44]
unseen_class_list=[39, 1, 15, 30, 50, 42, 27, 51, 49, 65, 21] #蔬菜水果
train_list=[x for x in train_list if (x not in ungrasp_list) and (x not in unseen_obj_list) and (x not in unseen_class_list)]
# can_list = [0, 2,4,5,6,7,8,9,10,11,12,19,20,23,31,33,35,37,38,39,40,42,47,48,60,61,62,65,66,67,68,70,71,72]
# can_list = [6, 47, 8, 9, 12, 48, 66, 67, 20, 40, 23, 31, 61, 19, 37, 33, 71, 10, 4, 70, 38, 68, 60, 2, 7, 5]
# can_list = [6, 47, 8, 9, 48, 66, 67, 40, 23, 31, 61, 19, 37, 33, 71, 10, 4, 70, 38, 68,  2, 7, 5]
can_list = [6, 47,         66,     40, 23,         19,                 4, 70,     68,  2,    5]
# can_list = train_list.copy()
# can_list = [ 38, 68,  2, 7, 5]

def rand_data(exist_locate,deterministic):
    t0 = time.time()
    while True:
        type_ =  random.randint(0, len(can_list)-1)
        # x =  random.randint(30, 55)
        x =  random.randint(40, 70) #(40,75)
        # x =  random.randint(35, 50)
        y =  random.randint(-15, 20)
        yaw = 0 #random.randint(0, 360)

        distances = [math.sqrt((x - coord[0])**2 + (y - coord[1])**2) for coord in exist_locate]
        if all(distance > dis_ for distance in distances):
            if deterministic:
                # return 7, x, y, yaw
                return 5, x, y, yaw
            else:
                return can_list[type_], x, y, yaw
        
        t1 = time.time()
        if t1 - t0 > 0.1:
            print('Can not find a new valid location!')
            return -1, -1, -1, -1

def gen_objs(sim_client,n,sceneID=0,deterministic=False):
    # print('------Generating objects------')
    # table_loc = [-1727,-1700,0,-1,1000]
    table_loc = [-210,-450,180,-1,1000]
    h = 100
    
    # scene = sim_client.Do(GrabSim_pb2.Action(sceneID=sceneID, action=GrabSim_pb2.Action.ActionType.WalkTo, values=table_loc))
    # print('message',scene.info)
    # print('location',scene.location)
    # print(scene.location)
    # if scene.location.X!=table_loc[0] or scene.location.Y!=table_loc[1]:
    #     print('reset error')
    #     return []
    scene = sim_client.CleanObjects(GrabSim_pb2.SceneID(value=sceneID))
    scene = sim_client.Observe(GrabSim_pb2.SceneID(value=sceneID))
    ginger_loc = [scene.location.X, scene.location.Y, scene.location.Z]
    exist_locate = []
    maked_objs=[]
    cnt=0
    while cnt<n:
        type_rand, x_rand, y_rand, yaw_rand = rand_data(exist_locate,deterministic)
        if type_rand == -1:
            print(f'Already have {cnt}')
            break
        exist_locate.append([x_rand, y_rand])

        obj_list = [GrabSim_pb2.ObjectList.Object(x=ginger_loc[0] + x_rand, y=ginger_loc[1] + y_rand, yaw=yaw_rand, z=h, type=type_rand)]

        scene = sim_client.MakeObjects(GrabSim_pb2.ObjectList(objects=obj_list, sceneID=sceneID))
        maked_objs.append([type_rand, ginger_loc[0] + x_rand, ginger_loc[1] + y_rand, h,yaw_rand])

        # print('collision',scene.collision)
        # print('Generate object: ', name_type[type_rand],  x_rand, y_rand, yaw_rand)

        cnt+=1
    
    # print('------Generate objects done------')
    # print()
    return maked_objs

import random
def find_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    assert True
    return None

def gen_scene_for_level3(sim_client,sceneID=0,training=True,targetObj=None):
    # print('------Generating objects------')
    # table_loc = [-1727,-1700,0,-1,1000]
    table_loc = [-210,-450,180,-1,1000]
    h = 100
    scene = sim_client.CleanObjects(GrabSim_pb2.SceneID(value=sceneID))
    scene = sim_client.Observe(GrabSim_pb2.SceneID(value=sceneID))
    ginger_loc = [scene.location.X, scene.location.Y, scene.location.Z]
    exist_locate = []
    maked_objs=[]
    cnt=0
    if targetObj==None:
        f=open('instructions/level3.pkl','rb')
        datas=pickle.load(f)
    else:
        f=open('instructions/level3_dict.pkl','rb')
        datas=pickle.load(f)
        datas=datas[targetObj]
        
    if training:
        data=random.choice(datas[:len(datas)//10*8])
    else:
        data=random.choice(datas[len(datas)//10*8:])

    for obj in data['scene']:
        type, x, y, yaw = find_key_by_value(name_type,obj[0]), obj[2], obj[1], 0
        exist_locate.append([x, y])

        obj_list = [GrabSim_pb2.ObjectList.Object(x=ginger_loc[0] + x, y=ginger_loc[1] + y, yaw=yaw, z=h, type=type)]

        scene = sim_client.MakeObjects(GrabSim_pb2.ObjectList(objects=obj_list, sceneID=sceneID))
        maked_objs.append([type, ginger_loc[0] + x, ginger_loc[1] + y, h,yaw])

        # print('collision',scene.collision)
        # print('Generate object: ', name_type[type_rand],  x_rand, y_rand, yaw_rand)

        cnt+=1
    
    # print('------Generate objects done------')
    # print()
    return data['object'], data['instruction'], maked_objs

def gen_scene_for_level4(sim_client,sceneID=0,training=True,file='instructions/level4.pkl',targetObj=None):
    # print('------Generating objects------')
    # table_loc = [-1727,-1700,0,-1,1000]
    table_loc = [-210,-450,180,-1,1000]
    h = 100
    scene = sim_client.CleanObjects(GrabSim_pb2.SceneID(value=sceneID))
    scene = sim_client.Observe(GrabSim_pb2.SceneID(value=sceneID))
    ginger_loc = [scene.location.X, scene.location.Y, scene.location.Z]
    exist_locate = []
    maked_objs=[]
    cnt=0
    if targetObj==None:
        f=open(file,'rb')
        datas=pickle.load(f)
    else:
        f=open('instructions/level4_dict.pkl','rb')
        datas=pickle.load(f)
        datas=datas[targetObj]

    if training:
        data=random.choice(datas[:len(datas)//10*8])
    else:
        data=random.choice(datas[len(datas)//10*8:])

    for obj in data['scene']:
        type, x, y, yaw = find_key_by_value(name_type,obj[0]), obj[2], obj[1], 0
        exist_locate.append([x, y])

        obj_list = [GrabSim_pb2.ObjectList.Object(x=ginger_loc[0] + x, y=ginger_loc[1] + y, yaw=yaw, z=h, type=type)]

        scene = sim_client.MakeObjects(GrabSim_pb2.ObjectList(objects=obj_list, sceneID=sceneID))
        maked_objs.append([type, ginger_loc[0] + x, ginger_loc[1] + y, h,yaw])

        # print('collision',scene.collision)
        # print('Generate object: ', name_type[type_rand],  x_rand, y_rand, yaw_rand)

        cnt+=1
    
    # print('------Generate objects done------')
    # print()
    return data['object'], data['instruction'][0], maked_objs

def gen_scene_from_data(sim_client,sceneID,scene):
    # print('------Generating objects------')
    # table_loc = [-1727,-1700,0,-1,1000]
    table_loc = [-210,-450,180,-1,1000]
    h = 100
    scene = sim_client.CleanObjects(GrabSim_pb2.SceneID(value=sceneID))
    scene = sim_client.Observe(GrabSim_pb2.SceneID(value=sceneID))
    ginger_loc = [scene.location.X, scene.location.Y, scene.location.Z]
    exist_locate = []
    maked_objs=[]
    cnt=0

    for obj in scene:
        type, x, y, yaw = find_key_by_value(name_type,obj[0]), obj[2], obj[1], 0
        exist_locate.append([x, y])

        obj_list = [GrabSim_pb2.ObjectList.Object(x=ginger_loc[0] + x, y=ginger_loc[1] + y, yaw=yaw, z=h, type=type)]

        scene = sim_client.MakeObjects(GrabSim_pb2.ObjectList(objects=obj_list, sceneID=sceneID))
        maked_objs.append([type, ginger_loc[0] + x, ginger_loc[1] + y, h,yaw])

        # print('collision',scene.collision)
        # print('Generate object: ', name_type[type_rand],  x_rand, y_rand, yaw_rand)

        cnt+=1
    
    # print('------Generate objects done------')
    # print()
    return maked_objs

def str2scene(a):
    scene=[]
    b=a.split(',')[:-1]
    for i in range(0,len(b),3):
        scene.append([b[i],int(b[i+1]),int(b[i+2])])
    return scene

def gen_scene(sim_client,level,sceneID=0,training=True,deterministic=False):
    assert level in [1,2,3,4]
    df=pd.read_csv('instructions/training.csv')
    if level<=3:
        if level==1:
            max_nums=1
        else:
            max_nums=3
        objs=gen_objs(sim_client,random.randint(1,max_nums),sceneID,deterministic)
        while len(objs)==0:
            time.sleep(1)
            objs=gen_objs(sim_client,random.randint(1,max_nums),sceneID,deterministic)
        targetObj=name_type[objs[0][0]]
        df=df[(df['level']==level) & (df['object']==targetObj)]
        if level>2:
            df=df[df['times']==2]
        data=df.sample(n=1)
        instruction=data['instruction'].values[0]
        id=data['id'].values[0]
    else:
        df=df[(df['level']==level) & (df['times']==2)]
        data=df.sample(n=1)
        targetObj=data['object'].values[0]
        instruction=data['instruction'].values[0]
        id=data['id'].values[0]
        scene=str2scene(data['scene'].values[0])
        objs=gen_scene_from_data(sim_client,sceneID,scene)

    instructionIndex=(id,0)

    return targetObj,instructionIndex,objs
