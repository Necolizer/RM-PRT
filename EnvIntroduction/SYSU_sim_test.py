#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@Time    :   2023/04/14 17:34:55
@Author  :   alice.xiao@cloudminds.com
@File    :   SYSU_sim_test.py
"""
import math
import time
import sys

sys.path.append('./')
sys.path.append('../')

import grpc
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import GrabSim_pb2
import GrabSim_pb2_grpc

fig = plt.figure()
plt.ion() 

channel = grpc.insecure_channel('127.0.0.1:30001',
                                options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                                         ('grpc.max_receive_message_length', 1024 * 1024 * 1024)])  # FIXME  IP:port
sim_client = GrabSim_pb2_grpc.GrabSimStub(channel)


def init():
    # TODO If use initial function even number of times in the same simulator, the navigation API doest not work.
    # When you meet this problem, just inital again or restart the simulator.
    # initial environment
    scene = sim_client.Init(GrabSim_pb2.Count(value=1))
    return scene


def reset():
    scene = sim_client.Reset(GrabSim_pb2.ResetParams())


def show_env_info():
    scene = sim_client.Observe(GrabSim_pb2.SceneID(value=0))
    print('------------------show_env_info----------------------')
    print(
        f"sceneID:{scene.sceneID}, location:{[scene.location.X, scene.location.Y]}, rotation:{scene.rotation}\n",
        f"joints number:{len(scene.joints)}, fingers number:{len(scene.fingers)}\n", f"objects number: {len(scene.objects)}\n"
        f"velocity:{scene.velocity}, rotation:{scene.rotating}, timestep:{scene.timestep}\n"
        f"timestamp:{scene.timestamp}, collision:{scene.collision}, info:{scene.info}")


def walk_test():
    """
    change location 
    GrabSim_pb2.Action(sceneID=0, action=GrabSim_pb2.Action.ActionType.WalkTo, values=[x, y, YAW, Q, dis])
    Q: 0: query, not move; 1: navigate to target position; -1: teleportate to target position
    """
    scene = sim_client.Observe(GrabSim_pb2.SceneID(value=0))

    walk_value = [scene.location.X, scene.location.Y, scene.rotation.Yaw]
    print('------------------walk_test----------------------')
    print("position:", walk_value)

    v_list = [[-2150.000000, -1350.000000], [-2539.000000, -1218.000000], [-2557.000000, -1209.000000], [-2568.000000, -1189.000000],
              [-2577.000000, -1157.000000], [-2577.000000, -1088.000000], [-2565.000000, -1067.000000], [-2547.000000, -1058.000000],
              [-1951.000000, -1058.000000], [-1930.000000, -1070.000000], [-1921.000000, -1088.000000], [-1721.000000, -1792.000000],
              [-1633.000000, -2097.000000], [-1624.000000, -2115.000000], [-1604.000000, -2126.000000], [-1498.000000, -2156.000000],
              [-1478.000000, -2167.000000], [-1469.000000, -2185.000000], [-1453.000000, -2278.000000]]

    for walk_v in v_list:
        walk_v = walk_v + [0, -1, 1000]
        print("walk_v", walk_v)
        action = GrabSim_pb2.Action(sceneID=0, action=GrabSim_pb2.Action.ActionType.WalkTo, values=walk_v)
        scene = sim_client.Do(action)
        print(scene.info)  # print navigation info


def joint_test():
    print('------------------joint_test----------------------')
    action_list = [[0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 36.0, -39.37394720315933, 37.2, -92.4, 4.136367428302765, -0.6212447762489319, 0.4],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 36.0, -39.62803038954735, 34.758509969711305, -94.80000152587891,
                       3.2295145452022553, -0.26782984733581544, 0.8000000059604645
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 32.88, -39.599720031023026, 31.95851058959961, -97.20000305175782,
                       4.905763161182404, -0.7028014183044433, 0.977626097202301
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 35.783679265975955, -37.86990386247635, 29.158511352539062, -99.60000457763672,
                       4.293414348363877, 0.19719859361648562, 0.8532893657684326
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 35.04450035095215, -36.334353387355804, 26.358512115478515, -102.00000610351563,
                       3.785160183906555, 0.2779792249202728, 0.7081573009490967
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 36.0, -36.267006546258926, 23.558512878417968, -104.40000762939454,
                       1.5072380006313324, 1.160965543985367, 0.30815730094909666
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 33.4882368850708, -34.37560546398163, 20.75851364135742, -106.80000915527344,
                       3.552527117729187, 0.9852039992809296, 0.7081572949886322
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 34.018811264038085, -34.4423651099205, 17.958514404296874, -109.20001068115235,
                       4.677242231369019, 1.4663134217262268, 0.30815730094909666
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 32.63617506980896, -32.80620041489601, 15.158515167236327, -110.70582208633422,
                       6.866850197315216, 2.366313362121582, 0.4005480229854584
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 31.17517325401306, -30.556201934814453, 12.358514976501464, -112.59623470306397,
                       7.203055512905121, 2.755515289306641, 0.6910970985889435
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 30.272712197303772, -29.811969935894012, 9.5585147857666, -114.09121279716491,
                       8.334378957748413, 3.6555153369903564, 1.09109708070755
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 28.180964212417603, -27.92907726764679, 6.758514595031738, -115.02938404083253,
                       9.462134897708893, 4.280387842655182, 1.3546329498291017
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 27.10963956832886, -25.95158475637436, 3.9585144042968747, -115.48578491210938,
                       9.841001415252686, 4.6832902193069454, 1.7546329736709594
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 24.22848172187805, -24.606801509857178, 1.1585144519805906, -116.015571641922,
                       11.12361261844635, 5.414817237854004, 2.1546329498291015
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 21.992600212097166, -22.699784100055695, -1.6414855003356936, -115.98941040039062,
                       11.341147816181183, 5.9691917419433596, 2.0297587394714354
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 18.872599487304687, -21.17943376302719, -4.290557765960694, -115.45827894210815,
                       11.961835908889771, 6.324305510520935, 2.4196860790252686
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 15.752598648071288, -19.162457704544067, -6.737515234947205, -114.35903997421265,
                       11.875249648094178, 6.964981412887573, 2.5117132663726807
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 12.689268517494202, -17.415902346372604, -8.89284381866455, -113.1150676727295,
                       12.115758717060089, 7.438566112518311, 2.526390218734741
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 9.569268112182616, -15.957421630620956, -10.483200693130494, -111.41921491622925,
                       12.20034761428833, 8.239937198162078, 2.872011184692383
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 6.91395980834961, -14.416840374469757, -11.41349368095398, -109.44302797317505,
                       13.480275297164917, 7.907512050867081, 3.065640926361084
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 4.09075822353363, -13.153028726577759, -11.979567313194275, -107.35125188827514,
                       13.087086379528046, 8.586905109882355, 3.3361712455749513
                   ]]

    for value in action_list:
        print("v len:", len(value))
        action = GrabSim_pb2.Action(sceneID=0, action=GrabSim_pb2.Action.ActionType.RotateJoints, values=value)
        scene = sim_client.Do(action)

        for i in range(8, 21):  # arm
            print(
                f"{scene.joints[i].name}:{scene.joints[i].angle} location:{scene.joints[i].location.X},{scene.joints[i].location.Y},{scene.joints[i].location.Z}"
            )
        print('')
        for i in range(5, 10):  # Right hand
            print(
                f"{scene.fingers[i].name} angle:{scene.fingers[i].angle} location:{scene.fingers[i].location[0].X},{scene.fingers[i].location[0].Y},{scene.fingers[i].location[0].Z}"
            )
        print('----------------------------------------')
        time.sleep(0.03)


def cal_obj_dist(message):
    object_loc_list = []
    obj_dist_list = []
    hand_loc = np.array([message.fingers[6].location[2].X, message.fingers[6].location[2].Y, message.fingers[6].location[2].Z])

    for obj in message.objects:
        obj_loc = np.array([obj.location.X, obj.location.Y, obj.location.Z])
        object_loc_list.append(obj_loc)
        dist = np.sqrt(np.sum(np.square(hand_loc - obj_loc)))
        obj_dist_list.append(dist)

    return obj_dist_list


def grasp_test():
    # walk to
    message = sim_client.Do(GrabSim_pb2.Action(action=GrabSim_pb2.Action.WalkTo, values=[-1525, -1515, 270, 1, 40]))

    print(message.info)
    while (True):
        message = sim_client.Do(GrabSim_pb2.Action(action=GrabSim_pb2.Action.WalkTo, values=[-1525, -1515, 270, 0, 40]))

        print(message.info)
        if (message.info == "AlreadyAtGoal"):
            break

    print('----------------------------grasp-------------------------------')
    time.sleep(1)
    # raise hand
    v = [
        0, 0.45222780741751195, 10.197763514518738, 1.902936053276062, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -98.761526827812194, -8.422849625349045,
        -13.497337341308594, -20.88106689453124, -2.2021982192993166, 4.8524204134941105, -2.467571830749512
    ]
    message = sim_client.Do(GrabSim_pb2.Action(action=GrabSim_pb2.Action.RotateJoints, values=v))
    print(message.collision)

    # grasp
    obj_dist_list = cal_obj_dist(message)
    min_index = np.argmin(obj_dist_list)
    sim_client.Do(GrabSim_pb2.Action(action=GrabSim_pb2.Action.Grasp, values=[1, min_index]))

    # walk away
    time.sleep(1)
    message = sim_client.Do(GrabSim_pb2.Action(action=GrabSim_pb2.Action.WalkTo, values=[-1300, -1442, 270, 1, 40]))

    # check reach goal
    while (True):
        message = sim_client.Do(GrabSim_pb2.Action(action=GrabSim_pb2.Action.WalkTo, values=[-1300, -1442, 270, 0, 40]))
        # print('check goal')

        print(message.info)
        if (message.info == "AlreadyAtGoal"):
            break

    # release
    # sim_client.Do(GrabSim_pb2.Action(action=GrabSim_pb2.Action.Release,values=[1]))

    time.sleep(0.5)


def get_camera(part):
    """
    1: foot
    2: head
    3: chest
    4: waist
    """
    action = GrabSim_pb2.CameraList(cameras=part, )
    return sim_client.Capture(action)


def camera_test():
    depth = get_camera([GrabSim_pb2.CameraName.Head_Depth])
    rgb = get_camera([GrabSim_pb2.CameraName.Head_Color])


def gen_obs_grasp_test():
    scene = sim_client.Observe(GrabSim_pb2.SceneID(value=0))
    ginger_loc = [scene.location.X, scene.location.Y, scene.location.Z]

    obj_list = [GrabSim_pb2.ObjectList.Object(x=ginger_loc[0] + 35.7, y=ginger_loc[1], z=98, type=5)]
    scene = sim_client.MakeObjects(GrabSim_pb2.ObjectList(objects=obj_list, sceneID=0))

    action_list = [[
        0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 29.759999999999998, -37.16870200634003, 34.4, -94.8, 5.526752604544162, -1.5960516929626465, -1.3
    ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 23.520000228881834, -35.43504357337952, 28.800001525878905, -99.60000305175781,
                       6.143285357952118, -2.8406716704368593, -2.599999952316284
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 17.28000045776367, -31.86991360783577, 23.20000114440918, -104.40000610351562,
                       6.143508880212903, -3.1301510363817213, -3.899999904632568
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 11.040000686645508, -28.989239394664764, 17.60000076293945, -106.21411199569702,
                       6.67448120713234, -3.142177647911012, -5.176533442735672
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 5.792587394714356, -25.501918017864227, 12.000000381469725, -105.40141551494598,
                       6.531419031694531, -2.6693858325481417, -6.3448518991470335
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 2.0456994438171385, -21.610762268304825, 6.3999999999999995, -103.92878708839416,
                       5.9266946613788605, -2.0047139167785644, -7.451148962974548
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, -0.7889365482330324, -18.239059150218964, 0.8000000953674311, -101.29412269592285,
                       6.816295194625854, -1.0670621275901793, -8.484375327825546
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, -4.128442475795746, -13.739059448242188, -4.799999928474427, -97.5776556968689,
                       9.747106528282165, 0.2535855889320373, -9.077314746379852
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, -10.125079469680786, -10.519007980823517, -10.312075471878053, -92.77765197753907,
                       10.536645138263703, 1.4567770123481751, -9.519901067018509
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, -16.365079154968264, -7.617221236228943, -14.154940271377564, -88.3628602027893,
                       11.338555693626404, 2.174972289800644, -9.756899026036262
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, -19.757361974716186, -6.860800191760063, -15.782474684715272, -83.56286163330078,
                       10.674141398072242, 3.1573299527168275, -10.028058829903603
                   ],
                   [
                       0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, -25.1708584690094, -5.854124575853348, -17.268349981307985, -78.76285858154297,
                       9.046833562850953, 3.731233984231949, -10.222823572158813
                   ]]

    time.sleep(0.5)
    for value in action_list:
        action = GrabSim_pb2.Action(sceneID=0, action=GrabSim_pb2.Action.ActionType.RotateJoints, values=value)
        scene = sim_client.Do(action)

    obj_dist_list = cal_obj_dist(scene)
    min_index = np.argmin(obj_dist_list)
    graspAction2Sim = GrabSim_pb2.Action(
        action=GrabSim_pb2.Action.ActionType.Grasp,
        values=[1, min_index]  # [hand, obj_id]
    )
    message = sim_client.Do(graspAction2Sim)

    value = [
        0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, -25.1708584690094, -50, -17.268349981307985, -78.76285858154297, 9.046833562850953,
        3.731233984231949, -10.222823572158813
    ]
    action = GrabSim_pb2.Action(sceneID=0, action=GrabSim_pb2.Action.ActionType.RotateJoints, values=value)
    scene = sim_client.Do(action)
    time.sleep(0.5)
    return scene


def gen_obj(h=98):
    scene = sim_client.Observe(GrabSim_pb2.SceneID(value=0))
    ginger_loc = [scene.location.X, scene.location.Y, scene.location.Z]

    obj_list = [
        GrabSim_pb2.ObjectList.Object(x=ginger_loc[0] + 30, y=ginger_loc[1] + 5, yaw=10, z=h, type=2),
        GrabSim_pb2.ObjectList.Object(x=ginger_loc[0] + 40, y=ginger_loc[1] - 5, z=h, type=5),
        GrabSim_pb2.ObjectList.Object(x=ginger_loc[0] + 45, y=ginger_loc[1] - 10, z=h, type=4),
        GrabSim_pb2.ObjectList.Object(x=ginger_loc[0] + 50, y=ginger_loc[1] - 25, z=h, type=5),
        GrabSim_pb2.ObjectList.Object(x=ginger_loc[0] + 160, y=ginger_loc[1], z=130, type=5)
    ]
    scene = sim_client.MakeObjects(GrabSim_pb2.ObjectList(objects=obj_list, sceneID=0))
    print(scene.collision)
    time.sleep(0.5)


def remove_obj(id_list=[1]):
    print('------------------remove objs----------------------')
    remove_obj_list = id_list
    scene = sim_client.RemoveObjects(GrabSim_pb2.RemoveList(objectIDs=remove_obj_list, sceneID=0))
    print(f"remove objects {id_list}. current obj:")
    time.sleep(0.5)
    # print_obj_info(scene.objects)


def clean_obj():
    print('------------------clean objs----------------------')
    scene = sim_client.CleanObjects(GrabSim_pb2.SceneID(value=0))
    # print_obj_info(scene.objects)


def print_obj_info(scene_objs):
    obj_dict = {}
    for i, obj in enumerate(scene_objs):
        obj_dict[i] = obj.name
    print(f"current obj:{obj_dict}")


if __name__ == '__main__':
    init()
    reset()
    show_env_info()
    walk_test()
    joint_test()
    grasp_test()

    # obj test
    reset()
    gen_obj(h=98)
    scene = gen_obs_grasp_test()
    print(scene.info)
    print(scene.collision)
    remove_obj(id_list=[0])
    clean_obj()