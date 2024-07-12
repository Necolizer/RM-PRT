import sys
import time
from Env.simUtils import *

sys.path.append('./')
sys.path.append('../')

import grpc

host = '127.0.0.1:30007'
scene_num = 1
map_id = 2
server = SimServer(host,scene_num = scene_num, map_id = map_id)
sim=Sim(host,scene_id=0)

sim.addDesk(1,h=90)
scene=sim.addObjects([[22,-60-0,10-10,sim.desk_height,0,0,-90,0.2,0.2,0.2]])
print(sim.getObservation().objects[-1])