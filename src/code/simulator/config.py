#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
config.py
"""
import numpy as np
import os
import json
# --- Set constants ---
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
WIDTH, HEIGHT = 6, 4
BALL_RADIUS = 0.25
BORDER = 0.20
SIGMA = np.array([[0.2758276,0],[0,0.6542066]])

# hyper-parameter
T = 40
TIMEOUT = 1440
ig_mode = 1
state_dim = T*16
n_actions = 645
nn_h1 = 150
nn_h2 = 250
nn_h3 = 450
epsilon = 0.5
epsilon_decay = 0.9
qlearning_gamma = 0.99 
init_mouse = (0,0)
reward_stop = 0.95

# Functions for generating candidate settings
def transfer(n,x):
    b=[]
    while True:
        s=n//x
        y=n%x
        b=b+[y]
        if s==0:
            break
        n=s
    b.reverse()
    return b
def generate_force(force_possible):
    force_list = []
    for num in force_possible:
        num = np.array(num)
        force = np.zeros([4,4])
        force[1,0] = (num[0]-1)*3
        force[2,:2] = (num[1:3]-1)*3
        force[3,:3] = (num[3:]-1)*3
        force = force+force.T
        force_list.append(force)
    return force_list
def generate_possible(level,length):
    possible = []
    for i in range(level**length):
        possible.append([0]*(length-len(transfer(i,level)))+transfer(i,level))
    return possible

# --- SET CANDIDATE SETTINGS --- 

# 3**7
# mass_list = [[1,1,1,1], [1,2,1,1], [2,1,1,1]]
# force_possible = generate_possible(3,6)
# force_list = generate_force(force_possible)


# 2**2*2**6 = 2**8
# mass_list = [[1,1,1,1], [1,2,1,1], [1,2,3,1], [3,1,1,2]]
# #mass_list = [[1,1,1,1], [1,2,1,1]]
# force_possible =generate_possible(2,6)
# force_list = generate_force(force_possible)




# 2**5*2**5 = 2**10
np.random.seed(0)
mass_all_possible = generate_possible(3,4)
mass_possible = [mass_all_possible[i] for i in np.random.choice(81,32,replace= False)]
mass_list = [np.array(mass)+1 for mass in mass_possible]

force_all_possible =generate_possible(3,6)
force_possible = [force_all_possible[i] for i in np.random.choice(729,32,replace= False)]
force_list = generate_force(force_possible)


# --- SET INITIAL PRIOR --- 
prior = dict()
for m in mass_list:
    for f in force_list:
        prior[(tuple(m),tuple(np.array(f).flatten()))] = 1.0/(len(mass_list)*len(force_list))

# --- SET STARTING CONDITIONS --- 
# sls = starting locations, svs = starting velocities, lf = local forces, mass = object densities
cond = {'sls':[{'x':0.26, 'y':2}, {'x':0.28, 'y':2}, {'x':1, 'y':0.26}, {'x':2, 'y':3.7}],
        'svs':[{'x':0.1, 'y':0}, {'x':-0.1, 'y':0}, {'x':0, 'y':-0.1}, {'x':0, 'y':0.1}],
        'lf':[[0, 3, 0, -3],
              [3, 0, 0, 0],
              [0, 0, 0, -3],
              [-3, 0, -3, 0]],
        'mass':[1,2,1,1],
        'timeout': 1440 
    }

# Functions for generating initial conditions
def generate_cond(size,timeout = TIMEOUT):
    cond_list = []
    # locations:
    for i in range(size):
        X = 5.1*np.random.rand(4) + BALL_RADIUS + BORDER
        Y = 3.1*np.random.rand(4) + BALL_RADIUS + BORDER
        VX = np.random.uniform(-2,2,4)
        VY = np.random.uniform(-2,2,4)
        lf_idx = np.random.randint(32)
        mass_idx = np.random.randint(32)
        cond_list.append({'sls':[{'x':X[0], 'y':Y[0]}, {'x':X[1], 'y':Y[1]}, {'x':X[2], 'y':Y[2]}, {'x':X[3], 'y':Y[3]}],
        'svs':[{'x':VX[0], 'y':VY[0]}, {'x':VX[1], 'y':VY[1]}, {'x':VX[2], 'y':VY[2]}, {'x':VX[3], 'y':VY[3]}],
        'lf':generate_force([force_possible[lf_idx]])[0].tolist(),
        'mass':(np.array(mass_possible[mass_idx])+1).tolist(),
        'timeout': timeout
        })
    return cond_list
def load_cond(file_name, size):
    if(os.path.exists(file_name)):
        with open(file_name,'r') as f:
            cond_list = json.load(f)
    else:
        cond_list = generate_cond(size)
        with open(file_name,'w') as f:
            json.dump(cond_list,f)
    return cond_list
train_cond = load_cond("train_cond"+str(TIMEOUT)+".json",60)
test_cond = load_cond("test_cond"+str(TIMEOUT)+".json",20)
print("timeout",train_cond[0]['timeout'])
