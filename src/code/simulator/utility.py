#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utility.py
"""

#IMPORT LIBRARIES
import numpy as np

#--- Set functions ---
def gaussian(s, d, Sigma, eta=1.0/50):
    diff = np.subtract(s,d)
    return np.exp(-eta/2 * np.dot(np.dot(diff.T, np.linalg.inv(Sigma)), diff))

def inner_expectation(true_dict, est_dict_ls, Sigma):
    if not isinstance(est_dict_ls, list):
        raise TypeError("Input est_dict_ls must be a list of dictionary object. Your input object is a {}"
                        .format(type(est_dict_ls)))
    if len(true_dict.values()) != 1:
        raise ValueError("For each timestamp, the true trajectory dictionary is fixed. Check the dimension first!")
    
    s_obj = true_dict.values()[0]
    gaussian_ls = []
    #print("masssssssssssssssssssss")
    for est_dict in est_dict_ls:
    	#print("*****************")
        for obj in est_dict:
            r = est_dict[obj]['r']
            theta =  est_dict[obj]['rotation']
            d = (r, theta)
            r_s = s_obj[obj]['r']
            theta_s = s_obj[obj]['rotation']
            s = (r_s, theta_s)
            gaussian_ls.append(gaussian(s, d, Sigma))  
            #print(gaussian(s, d, Sigma))       
    return np.mean(gaussian_ls)

def outer_expectation(true_dict, est_dict, Sigma, out_type):
    if out_type == 'mass':
        inner_e = []
        dict_ls = [i[0] for i in est_dict.keys()]
        for i in set(dict_ls):
            inner_force = []
            for j in est_dict:
                if i == j[0]:
                    inner_force.append(est_dict[j])
            inner_e.append(inner_expectation(true_dict, inner_force, Sigma))
        return np.mean(inner_e)
    
    elif out_type == 'force':
        inner_e = []
        dict_ls = [i[1] for i in est_dict.keys()]
        for i in set(dict_ls):
            inner_mass = []
            for j in est_dict:
                if i == j[1]:
                    inner_mass.append(est_dict[j])
            inner_e.append(inner_expectation(true_dict, inner_mass, Sigma))        
        return np.mean(inner_e)
    
    else:
        raise ValueError("Cannot recognize input out_type value. Currently only support computing predictive \
                         divergence of 'mass' and 'force'")

def get_reward(true_ls_dict, est_ls_dict, Sigma, mod=1):
    if mod == 1:
        rewards = []
        for i in range(len(est_ls_dict)):
            pd_mass = outer_expectation(true_ls_dict[i], est_ls_dict[i], Sigma, "mass")
            pd_force = outer_expectation(true_ls_dict[i], est_ls_dict[i], Sigma, "force")
            rewards.append((pd_mass+pd_force)/2)
        return 1-np.mean(rewards)
    
    elif mod == 2:
        reward_mass = []
        for i in range(len(est_ls_dict)):
            pd_mass = outer_expectation(true_ls_dict[i], est_ls_dict[i], Sigma, "mass")
            reward_mass.append(pd_mass)
        return 1-np.mean(reward_mass)
    
    elif mod == 3:
        reward_force = []
        for i in range(len(est_ls_dict)):
            pd_force = outer_expectation(true_ls_dict[i], est_ls_dict[i], Sigma, "force")
            reward_force.append(pd_force)
        return 1-np.mean(reward_force)

def generate_force(n,x):
    a=[0,1,2,3,4,5,6,7,8,9,'A','b','C','D','E','F']
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

def store_state(bodies):
    local_data = {}
    for i in range(0,len(bodies)):
        objname = 'o' + str(i + 1)
        local_data[objname] = {}
        local_data[objname]['x'] = [bodies[i].position[0]]
        local_data[objname]['y'] = [bodies[i].position[1]]
        local_data[objname]['vx'] = [bodies[i].linearVelocity[0]]
        local_data[objname]['vy'] = [bodies[i].linearVelocity[1]]
        local_data[objname]['rotation'] = [bodies[i].angle]
    return local_data

def simulate(bodies, cond, control_vec, t):
    #Loop over the dynamic objects
    for i in range(0,len(bodies)):
        #Grab and print current object name and location
        objname = bodies[i].userData['name']
        #Apply local forces
        for j in range(0, len(bodies)):
            #NB: The force strengths should be symmetric i,j==j,i for normal physics
            #otherwise you'll get "chasing" behaviour
            strength = cond['lf'][i][j]
            #If there's a pairwise interaction between these two objects...
            if strength!=0 and i!=j:
                #Calculate its force based on the objects masses and distances
                m = bodies[i].mass * bodies[j].mass
                d = ((bodies[i].position[0] - bodies[j].position[0])**2 +
                    (bodies[i].position[1] - bodies[j].position[1])**2)**0.5

                angle = np.arctan2(bodies[i].position[1] - bodies[j].position[1],
                            bodies[i].position[0] - bodies[j].position[0])
                f_mag = (strength * m) / d**2
                f_vec = (f_mag * np.cos(angle), f_mag * np.sin(angle))
                #Apply the force to the object
                bodies[j].ApplyForce(force=f_vec, point=(0,0), wake=True)
        if control_vec['obj'][t]==(i+1):
            bodies[i].linearDamping = 10
            c_vec = ( (1/0.19634954631328583) * 0.2*(control_vec['x'][t] - bodies[i].position[0]),
                    (1/0.19634954631328583) * 0.2*(control_vec['y'][t] - bodies[i].position[1]))
            #Apply the force to the object
            bodies[i].ApplyLinearImpulse(impulse=c_vec, point=(0,0), wake=True)
            if t!=(len(control_vec['obj'])-1):
                if control_vec['obj'][t+1]==0:
                    bodies[i].linearDamping = 0.05
        # Turned off all rotation but could include if we want
        bodies[i].angularVelocity = 0
        bodies[i].angle = 0
    return bodies
def update_simulate_data(local_data,bodies):
    for i in range(0,len(bodies)):
        objname = 'o' + str(i + 1)
        local_data[objname]['x'].append(bodies[i].position[0])
        local_data[objname]['y'].append(bodies[i].position[1])
        local_data[objname]['vx'].append(bodies[i].linearVelocity[0])
        local_data[objname]['vy'].append(bodies[i].linearVelocity[1])
        local_data[objname]['rotation'].append(bodies[i].angle)
    return local_data
def generate_trajectory(data,flag):
    trajectory = {}
    states = []
    for key in data:
        dict_r_theta = {}
        for obj in ['o1', 'o2', 'o3', 'o4']:
            vx = np.array(data[key][obj]['vx'])
            vy = np.array(data[key][obj]['vy'])
            x = np.array(data[key][obj]['x'])
            y = np.array(data[key][obj]['y'])
            r = np.sqrt(vx**2+vy**2)
            theta = np.arctan2(vy,vx)
            theta[theta<0] += 2 * np.pi
            if(flag):
                states += list(np.vstack([r,theta,x,y]).transpose().flatten())
            dict_r_theta[obj] = {'r':list(r),'rotation':list(theta)}
        trajectory[key] = dict_r_theta
    return trajectory,list(states)