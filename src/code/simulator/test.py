from environment import physic_env
import numpy as np
from config import *

new_env = physic_env(cond,mass_list,force_list,T,pd_mode)
#control_vec = {'obj': np.append(np.repeat(0, 60), np.repeat(1, 180)), 'x':np.repeat(3, 240), 'y':np.repeat(3, 240)}


# test 10 time frame:
#print("*********",self.bodies[0].position[0])
print("start")
state, reward, is_done = new_env.step(0)
# state = new_env.reset()
# print(state)
print(reward)


# test the whole process.
# for i in range(cond['timeout']/10):
# 	print("time",i)
# 	reward = new_env.Step10(control_vec)
# 	print(reward)
