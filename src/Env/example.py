import sys
import time
from simUtils import *

import grpc

host = '127.0.0.1:30007'
scene_num = 1
map_id = 2
server = SimServer(host,scene_num = scene_num, map_id = map_id)
sim=Sim(host,scene_id=0)

# 可视化模型结果
import pickle
with open('/data2/liangxiwen/zkd/SeaWave/outputs/2023-12-30/15-21-03/media/episodes/trajectory.pkl','rb') as f:
    df=pickle.load(f)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
success_num=0
for index,data in enumerate(df[1:]):
    sim.reset()
    sim.addDesk(data['deskInfo']['desk_id'])
    sim.addObjects(data['objs'])
    target_oringin_loc=sim.getObjsInfo()[1]['location']
    for frame in data['track']:
        if sim.grasp_state['Right']==0:
            sim.moveHand(*frame[:3],keep_rpy=(0,0,0),gap=0.1)
        else:
            sim.moveHand(*frame[:3],gap=0.1)
        # time.sleep(0.5)
        # print(sigmoid(frame[-1]))
        # print(frame[:3])
        if sigmoid(frame[-1])>0.5 and sim.grasp_state['Right']==0:
            sim.grasp(angle=(65,68))
            import matplotlib.pyplot as plt 
            time.sleep(3)
            # mat=sim.getImage()
            # plt.imshow(mat)
            # plt.show()
        target_now_loc=sim.getObjsInfo()[1]['location']
        if target_now_loc[2]-target_oringin_loc[2]>10:
            success_num+=1
            break
        # else:
        #     sim.release()
    print(f'success_rate: {success_num}/{index+1}')
        