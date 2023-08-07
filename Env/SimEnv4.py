import random
import time

import gym
from gym import spaces
import numpy as np
import pickle
from PIL import Image

from utils import *
from gen_data import *

try:
    import GrabSim_pb2_grpc
    import GrabSim_pb2
except:
    import os;
    os.chdir("./python/")
    import GrabSim_pb2_grpc
    import GrabSim_pb2


class SimEnv(gym.Env):
    
    
    
    def __init__(self,client,sceneID, deterministic=False,target=None,action_nums=11,bins=64,abs_distance=10,use_image=True,max_steps=100,level=1,training=True):
        assert action_nums in [14, 11, 7, 8, 9]
        
        self.client=client
        channel = grpc.insecure_channel(self.client,options=[
            ('grpc.max_send_message_length', 1024*1024*1024),
            ('grpc.max_receive_message_length', 1024*1024*1024)
        ])
        self.stub = GrabSim_pb2_grpc.GrabSimStub(channel)
        self.sceneID=sceneID

        self.deterministic=deterministic
        self.target=target
        self.action_nums=action_nums
        self.bins=bins
        self.abs_distance=abs_distance
        self.use_image=use_image
        self.max_steps=max_steps
        self.level=level
        self.training=training

        self.scene=self.stub.Reset(GrabSim_pb2.ResetParams(sceneID=self.sceneID))
        self.jointsArrange=initJointsArrange()
        self.action_space = spaces.Box(low=-1, high=1, shape=(action_nums+1,), dtype=np.float32)
        self.observation_space = self.initObsSpaces()

        if target is not None:
            assert target in self.targets.keys()
        

        self.cnt=0
        self.reset_counts=0
        self.total_success=0
        self.total_rule_success=0
        self.reset()
        
        print('successfully initialized')
    
    def initObsSpaces(self):
        head_rgb=spaces.Box(low=0, high=1.0, shape=(224, 224, 4), dtype=np.float64)
        state=spaces.Box(low=-np.inf,high=np.inf,shape=(7+7+7+3+3+1,),dtype=np.float64)
        instruction=spaces.Box(low=0,high=10000,shape=(2,),dtype=np.int64)
        return spaces.Dict({'head_rgb':head_rgb,'state':state,'instruction':instruction})   
     
    def initLocation(self,Location,joints, deterministic=True):
        scale=1   #max 5
        x,y,yaw=Location
        initLocation=[x,y,yaw]
        if not deterministic:
            while True:
                initLocation[0]=round(random.uniform(x-10*scale,x+10*scale),0)
                initLocation[1]=round(random.uniform(y-4*scale,y+4*scale),0)
                initLocation[2]=round(random.uniform(yaw-6*scale,yaw+6*scale),0)
                msg=self.changeLocation(initLocation[0],initLocation[1],initLocation[2])
                if msg:
                    break
                scale*=0.9
                if scale<0.3:
                    initLocation=[x,y,yaw]
                    msg=self.changeLocation(initLocation[0],initLocation[1],initLocation[2])
                    break
        else:
            msg=self.changeLocation(initLocation[0],initLocation[1],initLocation[2])

        joints=np.array(joints)
        if (joints<self.jointsArrange[:,0]).sum() + (joints>self.jointsArrange[:,1]).sum()>0:
            print(joints)
            print(self.jointsArrange[:,0])
            print(self.jointsArrange[:,1])
        assert (joints<self.jointsArrange[:,0]).sum() + (joints>self.jointsArrange[:,1]).sum()==0
        self.changeJoints(joints)
        return msg, initLocation
    
    def getScene(self):
        self.scene=self.stub.Observe(GrabSim_pb2.SceneID(value=self.sceneID))
        return self.scene
    
    def getObjLocation(self,objName):
        self.scene=self.getScene()
        locations=[]
        for i,obj in enumerate(self.scene.objects):
            if obj.name==objName or (objName is None):
                location = obj.location
                locations.append([i, location.X, location.Y, location.Z])
        return np.array(locations)
    
    def getObjIDLocation(self,objID):
        self.scene=self.getScene()
        targetObjLoc=self.scene.objects[objID].location
        targetObjLoc=np.array([targetObjLoc.X,targetObjLoc.Y,targetObjLoc.Z])
        return targetObjLoc
    
    def getjointLocation(self,jointID):
        self.scene=self.getScene()
        Loc=self.scene.joints[jointID].location
        Loc=np.array([Loc.X,Loc.Y,Loc.Z])
        return Loc
    
    def getfingerLocation(self):
        self.scene=self.getScene()
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
        Loc=self.getObjIDLocation(self.targetid)
        finger = self.getfingerLocation()[5+3-1]
        diff = [Loc[i]-finger[i] for i in range(3)]
        state.extend(diff)

        state.append(int(self.last_grasp))

        return np.array(state)
    
    def getCamera(self, caremras=[GrabSim_pb2.CameraName.Head_Color,GrabSim_pb2.CameraName.Head_Depth]):
        action = GrabSim_pb2.CameraList(sceneID=self.sceneID, cameras=caremras)
        images = self.stub.Capture(action).images
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
    
    def getObservation(self):
        Obs={}
        if self.use_image:
            Obs['head_rgb']=self.getCamera()
        else:
            Obs['head_rgb']=np.random.rand(224,224,4)
        Obs['state']=self.getState()
        Obs['instruction']=self.instructionIndex
        return Obs
    
    def changeLocation(self,x,y,Yaw):
        self.cnt+=1
        message = self.stub.Do(GrabSim_pb2.Action(sceneID=self.sceneID, action = GrabSim_pb2.Action.ActionType.WalkTo,values = [x, y, Yaw, 0, 1000]))

        if message.info=='Unreachable' or message.info=='Failed':
            print('message.info',message.info)
            print(message.location,x,y,Yaw)
            return False
        
        message = self.stub.Do(GrabSim_pb2.Action(sceneID=self.sceneID, action = GrabSim_pb2.Action.ActionType.WalkTo,values = [x, y, Yaw, -1, 1000]))
        if (message.location.X!=x or message.location.Y!=y):
            return False
        return True
    
    def changeJoints(self,joints):
        lower_than_min=joints<self.jointsArrange[:,0]
        joints[lower_than_min]=self.jointsArrange[lower_than_min,0]
        higher_than_max=joints>self.jointsArrange[:,1]
        joints[higher_than_max]=self.jointsArrange[higher_than_max,1]
        if (joints<self.jointsArrange[:,0]).sum() + (joints>self.jointsArrange[:,1]).sum()>0:
            self.not_move_for_limit+=1
            return False

        message = self.stub.Do(GrabSim_pb2.Action(sceneID=self.sceneID, action = GrabSim_pb2.Action.ActionType.RotateJoints,values = joints))
        if message.info=='Unreachable':
            print('Unreachable')
            return False
        time.sleep(0.03)
        return True, message.collision
    
    def get_nearest_obj(self,targetObj=None):
        ObjLocs=self.getObjLocation(targetObj)
        assert len(ObjLocs)>0

        fingers=self.getfingerLocation()

        fingerR3Loc = fingers[5+3-1]
        tcp_to_obj_pos = ObjLocs[:,1:]-fingerR3Loc
        nearest_obj_id = np.linalg.norm(tcp_to_obj_pos,axis=1).argmin()
        

        return int(ObjLocs[nearest_obj_id][0]), ObjLocs[nearest_obj_id][1:]
    
    def check_arrive(self):
        Loc = self.getObjIDLocation(self.targetid)
        dis = self.compute_distance(Loc)
        return dis<=self.abs_distance
    
    def compute_distance(self, objLoc):
        fingers = self.getfingerLocation()

        tcp_to_obj_pos = objLoc-fingers[5+3-1]
        tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
        return tcp_to_obj_dist

    def compute_dense_reward(self, info):

        reward = 0.0

        if info["is_success"]:
            reward += 5
            return reward
  
        # id, Loc = self.get_nearest_obj()
        # Loc = self.getObjIDLocation(self.targetid)
        # fingers = self.getfingerLocation()

        # tcp_to_obj_pos = Loc-fingers[5+3-1]
        # tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
        # reward += ( 1 - np.tanh(0.01 * tcp_to_obj_dist)) - self.last_red 
        # self.last_red = ( 1 - np.tanh(0.01 * tcp_to_obj_dist))

        Loc = self.getObjIDLocation(self.targetid)
        tcp_to_obj_dist = self.compute_distance(Loc)
        reward += (self.last_dis -tcp_to_obj_dist  )/self.distance
        self.last_dis = tcp_to_obj_dist

        # reward -= np.abs(self.state[3+3-1])/45*(1-np.tanh(0.01 * self.distance))/4
        
        reward -= np.abs((self.state[3+3-1]-self.last_back))/100
        self.last_back=self.state[3+3-1]

        # if info['rule_grasp']:
        #     if self.rule_success==0:
        #         reward+=1
        #         self.rule_success=1

        # if self.is_grasp:
        #     reward=0
        #     reward += 2
        #     tcp_to_obj_pos = Loc-self.dest
        #     tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
        #     reward += (1 - np.tanh(0.01 * tcp_to_obj_dist))
        #     reward -= np.abs(self.state[3+3-1])/45/4
        #     # reward -= np.abs(self.state[3+19-1]-70)/180/4

        if not info["move_success"]:
            reward -= 0.1
        
        # if self.counts>=self.max_steps:

        return reward
    
    def check_grasp(self):
        # id, Loc = self.get_nearest_obj()
        id = self.targetid
        Loc = self.getObjIDLocation(self.targetid)
        fingers = self.getfingerLocation()

        flag=True
        for i in [1,3,5]:
            finger=fingers[5+i-1]
            if np.linalg.norm(finger[:2]-Loc[:2])>6.5*1.9:
                flag=False
            
        for i in [1,3]:
            finger=fingers[5+i-1]
            if np.abs(finger[2]-Loc[2])>4*1.9:
                flag=False
        
        return flag
    
        # if flag==False:
        #     return False, None, flag
        
        # self.stub.Do(GrabSim_pb2.Action(sceneID=self.sceneID, action=GrabSim_pb2.Action.Grasp, values=[1, id]))
        # joints=self.state[3:3+21].copy()
        # joints[2]+=10
        # self.changeJoints(joints)

        # scene = self.stub.Observe(GrabSim_pb2.SceneID(value=self.sceneID))
        # newLoc = scene.objects[id].location
        # newLoc = np.array([newLoc.X, newLoc.Y, newLoc.Z])

        # self.changeJoints(self.state[3:3+21])
        # # self.stub.Do(GrabSim_pb2.Action(sceneID=self.sceneID, action=GrabSim_pb2.Action.Release,values=[1]))

        # if (newLoc==Loc).all():
        #     return False, None, flag
        # return True,id,flag
    
    def grasp(self, id):
        self.stub.Do(GrabSim_pb2.Action(sceneID=self.sceneID, action=GrabSim_pb2.Action.Grasp, values=[1, id]))

    def Do(self,action):
        joints=self.state[3:3+21].copy()
        
        for i,v in enumerate(action):
            if self.action_nums==11:
                if i<4:
                    loc=i
                else:
                    loc=i-4+14
            elif self.action_nums==14:
                if i<7:
                    loc=i
                else:
                    loc=i-7+14
            elif self.action_nums==7:
                loc=i-7+14
            elif self.action_nums==8:
                if i==0:
                    loc=2
                else:
                    loc=i-1+14
            elif self.action_nums==9:
                if i==0:
                    loc=0
                elif i==1:
                    loc=2
                else:
                    loc=i-2+14

            joints[loc]=joints[loc]+v*(self.jointsArrange[loc,1]-self.jointsArrange[loc,0])/self.bins
            
        msg, collision=self.changeJoints(joints)
        if len(collision)>0:
            msg=False
  
        return msg, collision
        
    def step(self, action): 
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        x,y,Yaw=self.state[:3]
        info={"is_success":False,'move_success':True,'rule_grasp':False}

        arrive=False
        if action[-1]<0 and self.is_grasp:
            info['move_success']=False
        else:
            if action[-1]>0 and self.is_grasp==False and self.last_grasp==0:
                self.last_grasp=1
                arrive=self.check_grasp()
            else:
                self.last_grasp=0
                msg,collision=self.Do(action[:-1])
                if msg==False:
                    info['move_success']=False
        
        self.obs = self.getObservation()
        self.state=self.obs['state']
        self.counts += 1

        if arrive:
            self.stay_target+=1
        else:
            self.stay_target=0

        if arrive:
            info['is_success']=True
            self.total_success+=1
            # done=True

        reward=self.compute_dense_reward(info)

        if self.counts>=self.max_steps or info['is_success'] or info['move_success']==False:

            done=True
        else:
            done = False
          
        return self.obs, reward, done, info
    
    def reset(self):
        # if self.reset_counts%20==0:
        #     initworld = self.stub.Init(GrabSim_pb2.Count(value = 1))
        #     time.sleep(5)

        self.reset_counts+=1
        if self.reset_counts%20==0:
            print('total_rule_success',self.total_rule_success)
            print('total_success',self.total_success)

        self.stub.Reset(GrabSim_pb2.ResetParams(sceneID=self.sceneID))
        time.sleep(1)
        self.targetObj,self.instructionIndex,self.objs=gen_scene(self.stub,self.level,sceneID=self.sceneID)
        initJoints = np.array([0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, -60, -70, -60, -50, 80, 0, 0])
        self.changeJoints(initJoints)
        self.cnt=0
        self.stay_target=0
        self.is_grasp=False
        self.rule_success=0
        self.last_grasp=0
        self.last_back = initJoints[2]

        id, Loc=self.get_nearest_obj(self.targetObj)
        self.targetid = id
        self.distance = self.compute_distance(Loc)
        self.last_dis = self.distance
        self.dest=Loc.copy()
        self.dest[2]+=30
        
        self.counts=0
        self.obs=self.getObservation()
        self.state=self.obs['state']
        self.not_move=0
        self.not_move_for_limit=0
        
        return self.obs
        
    def render(self):
        return None
        
    def close(self):
        return None

    def __reduce__(self):
        # 返回用于序列化和反序列化对象的元组
        return (self.__class__, (self.sceneID,))
    
    def __getstate__(self):
        # 返回一个不包含网络连接对象的状态字典
        state = self.__dict__.copy()
        del state['stub']
        return state

    def __setstate__(self, state):
        # 根据状态字典重新设置对象状态
        self.__dict__.update(state)
        channel = grpc.insecure_channel(self.client,options=[
            ('grpc.max_send_message_length', 1024*1024*1024),
            ('grpc.max_receive_message_length', 1024*1024*1024)
        ])
        self.stub = GrabSim_pb2_grpc.GrabSimStub(channel)  # 在反序列化后重新创建网络连接对象
