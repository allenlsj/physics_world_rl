#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
config.py
"""
import numpy as np

# --- Set constants ---
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
WIDTH, HEIGHT = 6, 4
BALL_RADIUS = 0.25
BORDER = 0.20
SIGMA = np.array([[0.2758276,0],[0,0.6542066]])

# hyper-parameter
T = 40
ig_mode = 1
state_dim = T*8
n_actions = 645
nn_h1 = 150
nn_h2 = 250
nn_h3 = 450
epsilon = 0.5
epsilon_decay = 0.95
qlearning_gamma = 0.99 
init_mouse = (0,0)
#---For demonstration purposes, some random control---
control_vec = {'obj': np.append(np.repeat(0, 60), np.repeat(1, 180)), 'x':np.repeat(3, 240), 'y':np.repeat(3, 240)}
# control_vec = {'obj': np.repeat(0, 240), 'x':np.repeat(3, 240), 'y':np.repeat(3, 240)}
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


simulate_state_dic_list = None
true_state_dic_list = None

# prior = dict()
# for m in mass_list:
#     for f in force_list:
#         prior[(tuple(m),tuple(np.array(f).flatten()))] = 1.0/(len(mass_list)*len(force_list))


# --- SET STARTING CONDITIONS --- 
# sls = starting locations, svs = starting velocities, lf = local forces, mass = object densities
cond = {'sls':[{'x':1, 'y':1}, {'x':2, 'y':1}, {'x':1, 'y':2}, {'x':2, 'y':2}],
        'svs':[{'x':0, 'y':0}, {'x':0, 'y':0}, {'x':0, 'y':0}, {'x':0, 'y':0}],
        'lf':[[0, 3, 0, -3],
              [3, 0, 0, 0],
              [0, 0, 0, -3],
              [-3, 0, -3, 0]],
        'mass':[1,2,1,1],
        'timeout': 480
    }

# 3**7
# mass_list = [[1,1,1,1], [1,2,1,1], [2,1,1,1]]

# force_possible =[]
# for i in range(3**6):
#     force_possible.append([0]*(6-len(transfer(i,3)))+transfer(i,3))
# force_list = generate_force(force_possible)


# 2**2*2**6 = 2**8
# mass_list = [[1,1,1,1], [1,2,1,1], [1,2,3,1], [3,1,1,2]]
# #mass_list = [[1,1,1,1], [1,2,1,1]]
# force_possible =[]
# for i in range(2**6):
#     force_possible.append([0]*(6-len(transfer(i,2)))+transfer(i,2))
# force_list = generate_force(force_possible)



# 2**5*2**5 = 2**10
mass_possible =[]
for i in range(2**4):
    mass_possible.append([0]*(4-len(transfer(i,2)))+transfer(i,2))
mass_list = [np.array(mass)+1 for mass in mass_possible] + [np.array(mass)*2+1 for mass in mass_possible]
force_possible =[]
for i in range(2**6):
    force_possible.append([0]*(6-len(transfer(i,2)))+transfer(i,2))
force_list = generate_force(force_possible[:32])


# initial prior
prior = dict()
for m in mass_list:
    for f in force_list:
        prior[(tuple(m),tuple(np.array(f).flatten()))] = 1.0/(len(mass_list)*len(force_list))
