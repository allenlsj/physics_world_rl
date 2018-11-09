#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_PD.py
check whether PD is valid
"""
from config import *
from environment import physic_env
import matplotlib.pyplot as plt
mass_list = [[1,1,1,1], [1,2,1,1]]
force_possible =[]
for i in range(2**6):
    force_possible.append([0]*(6-len(generate_force(i,2)))+generate_force(i,2))
force_list = []
for num in force_possible:
    num = np.array(num)
    force = np.zeros([4,4])
    force[1,0] = (num[0]-1)*3
    force[2,:2] = (num[1:3]-1)*3
    force[3,:3] = (num[3:]-1)*3
    force = force+force.T
    force_list.append(force)

cond = {'sls':[{'x':1, 'y':1}, {'x':3, 'y':1}, {'x':1, 'y':3}, {'x':2, 'y':3}],
        'svs':[{'x':0.5, 'y':0}, {'x':0, 'y':0}, {'x':0, 'y':0}, {'x':0, 'y':0}],
        'lf':[[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]],
        'mass':[1,0.3,1,1],
        #'timeout': 240
        'timeout': 20
    }

time_stamp = 1
control_vec = {'obj': np.repeat(0, time_stamp+1), 'x':np.repeat(3, time_stamp+1), 'y':np.repeat(3, time_stamp+1)}


# RUN
new_env = physic_env(cond,mass_list,force_list)
PD_mass = []
PD_force = []
distance = []
for i in range(cond['timeout']/time_stamp):
	print("time",i)
	pm,pf,state = new_env.step(control_vec,time_stamp)
	d = state[-1].values()[0]['o2']['x'][-1] - state[-1].values()[0]['o1']['x'][-1]
	PD_mass.append(pm)
	PD_force.append(pf)
	distance.append(d)
	print("*******",state)
	print("*******",distance)
plt.plot(distance,PD_mass,label = 'pd_mass')
plt.legend()
plt.show()
plt.plot(distance,PD_force,label = 'pd_force')
plt.legend()
plt.show()