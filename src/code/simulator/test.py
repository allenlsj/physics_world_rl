from environment import physic_env
import numpy as np
from config import *

new_env = physic_env(cond,mass_list,force_list)
#control_vec = {'obj': np.append(np.repeat(0, 60), np.repeat(1, 180)), 'x':np.repeat(3, 240), 'y':np.repeat(3, 240)}


# test 10 time frame:
#
# print("start")
# reward = new_env.Step10(control_vec)
# print(reward)


# test the whole process.
for i in range(cond['timeout']/10):
	print("time",i)
	reward = new_env.Step10(control_vec)
	print(reward)
