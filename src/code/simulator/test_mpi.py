from environment_mpi4py import physic_env
import numpy as np
from config import *
import math
import time
from math import acos, cos
from mpi4py import MPI
import json

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
dest = 0

# total size of configuration space
CONFIG_SIZE = 1024
# allocate buffer size in advance
TRACE_SIZE = 7000000
COND_SIZE = 580


# Initialize all communication variable
trace = []
all_trace = []
buf_all_trace = np.zeros(TRACE_SIZE,dtype = int)
idx = np.zeros(2, dtype=int)
current_cond = None

# allocate simulated properties for each process
config_size = CONFIG_SIZE/size
sub_key = np.arange(rank*config_size,(rank+1)*config_size)
print(rank,len(sub_key))

# Initialize environment for all process
if(rank == 0):
	new_env = physic_env(train_cond,init_mouse,T,sub_key,ig_mode, prior,reward_stop)
else:
	new_env = physic_env([train_cond[0]],init_mouse,T,sub_key)

st = time.time()
for i in range(3):
	if(rank == 0):
		# use last all_trace to calculate reward if it's game
		if(len(all_trace)!=0):
			all_trace = json.loads(all_trace)
			new_env.game_step(all_trace)
		# specify action and run true trajectory
		dict_cond = new_env.step(idx[0],current_cond,true_case = True)
		str_cond = json.dumps(dict_cond)
		idx = np.array([np.random.randint(0,645),len(str_cond)],dtype = int)
		current_cond =np.array(list(map(lambda x:ord(x),str_cond)))
	else:
		current_cond = np.zeros(COND_SIZE, dtype=int)

	# Broadcast true condition and action idx to all processes
	comm.Bcast(idx,root = 0)
	if(rank!=0):
		current_cond = np.zeros(idx[1], dtype=int)
	comm.Bcast(current_cond, root = 0)
	current_cond = json.loads(''.join(list(map(lambda x:chr(x),current_cond))))
	# Simulate in each process, include the main process: update condition-->simulate-->return trajectory
	trace = json.dumps(new_env.step(idx[0],current_cond,true_case = False))
	# store trace to buffer
	buf_trace=np.zeros(TRACE_SIZE/size,dtype = int) - 1
	buf_trace[:len(trace)] = list(map(lambda x:ord(x),trace))
	# gather all traces to main process
	comm.Gather(buf_trace,buf_all_trace,root = 0)
	if(rank==0):
		buf_all_trace = buf_all_trace.reshape((size,TRACE_SIZE/size))
		if(len(all_trace)!=1):
			all_trace = '{'+','.join([''.join(list(map(lambda x: chr(x),xitem[xitem>-1])))[1:-1] for xitem in buf_all_trace])+'}'
		else:
			all_trace = str(map(lambda x: chr(x),xitem[xitem>-1]))
		print(i,"***** Gather trace in Process ",rank)
	# make sure gather information from all partition

print("average time:",(time.time()-st)/3)
