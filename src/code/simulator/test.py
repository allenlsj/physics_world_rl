from environment import physic_env
import numpy as np
from config import *

new_env = physic_env(cond,mass_list,force_list,init_mouse,T,ig_mode, prior)
#control_vec = {'obj': np.append(np.repeat(0, 60), np.repeat(1, 180)), 'x':np.repeat(3, 240), 'y':np.repeat(3, 240)}


# test 10 time frame:
#print("*********",self.bodies[0].position[0])
# print("start")
#state, reward, is_done = new_env.step(0)
# state = new_env.reset()
# print(len(state))
#print(reward)

import time
# test the whole process.
#start = time.time()
#print("start",time.strftime("%H:%M:%S", time.localtime()))
for t in range(5):
	print(t)
	states = new_env.reset()
	for i in range(cond['timeout']/T):
		idx = np.random.randint(0,645)
		states, reward, is_done = new_env.step(idx)
		print(is_done)
		print(reward)
	
	#print("step ",i+1,time.time() - start)
	#start = time.time()