# from environment_baseline import physic_env # without multiprocess
#from environment_pool import physic_env # multiprocess without fix chunk size
from environment_multiprocess import physic_env # multiprocess with fix chunk size
import numpy as np
from config import *
import math
import time

new_env = physic_env(train_cond,init_mouse,T,mass_list,force_list,ig_mode, prior,reward_stop)

st = time.time()
#print("start",time.strftime("%H:%M:%S", time.localtime()))

# # test game multiprocess
# for i in range(cond['timeout']/T):
# 	idx = np.random.randint(0,645)
# 	new_env.step(idx,False)

# test action multiprocess
for i in range(3):
	idx = np.random.randint(0,645)
	new_env.step(idx,True)

print("average time:",(time.time()-st)/3)