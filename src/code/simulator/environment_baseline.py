#!/usr/bin/env python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
agent.py
"""

# IMPORT LIBRARIES
import Box2D  # The main library
import numpy as np
import random as rd
import copy
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody, vec2)
from config import *
from utility import gaussian, store_state ,generate_trajectory
from action_generator import generate_action
from information_gain import *
import time

class physic_env():
    def __init__(self, cond, init_mouse, time_stamp, mass_list, force_list, ig_mode, prior,reward_stop):
        # --- pybox2d world setup ---

        # Create the world
        self.world = world(gravity=(0, 0), doSleep=True)
        self.bodies = []
        self.walls = []
        self.data = {}
        self.cond_list = cond
        self.cond = cond[np.random.randint(len(cond))]
        self.init_mouse = init_mouse
        self.add_pucks()
        self.add_static_walls()
        self.mass_list = mass_list
        self.force_list = force_list
        self.T = time_stamp
        self.ig_mode = ig_mode
        self.prior = copy.deepcopy(prior)
        self.PRIOR = copy.deepcopy(prior)
        self.reward_stop = reward_stop
        if self.ig_mode == 1:
            self.total_reward = entropy(marginalize_prior(prior, 0))
            print("Total reward per game:{}".format(self.total_reward))
        elif self.ig_mode == 2:
            self.total_reward = entropy(marginalize_prior(prior, 1))
            print("Total reward per game:{}".format(self.total_reward))
        self.step_reward = []
        #self.simulate_state_dic = {}
        #self.true_state_dic = {}

    # --- add pucks (bodies) ---
    def add_pucks(self):
        for i in range(0, len(self.cond['sls'])):
                # Give each a unique name
            objname = 'o' + str(i + 1)
            # Create the body
            b = self.world.CreateDynamicBody(position=(self.cond['sls'][i]['x'], self.cond['sls'][i]['y']),
                                             linearDamping=0.05, fixedRotation=True,
                                             userData={'name': objname, 'bodyType': 'dynamic'})
            b.linearVelocity = vec2(
                self.cond['svs'][i]['x'], self.cond['svs'][i]['y'])
            # Add the the shape 'fixture'
            circle = b.CreateCircleFixture(radius=BALL_RADIUS,
                                           density=self.cond['mass'][i],
                                           friction=0.05, restitution=0.98)
            b.mass = self.cond['mass'][i]
            # Add it to our list of dynamic objects
            self.bodies.append(b)
            # Add a named entry in the data for this object
        # init self.data by intial condition
        self.data = self.initial_data(self.bodies)

    # --- add static walls ---
    def add_static_walls(self):
        w = self.world.CreateStaticBody(position=(WIDTH / 2, 0), #shapes=polygonShape(box=(WIDTH / 2, BORDER)),
                                        userData={'name': 'top_wall', 'bodyType': 'static'})
        w.CreateFixture(shape=polygonShape(
            box=(WIDTH / 2, BORDER)), friction=0.05, restitution=0.98)
        self.walls.append(w)
        w = self.world.CreateStaticBody(position=(WIDTH / 2, HEIGHT), #shapes=polygonShape(box=(WIDTH/2, BORDER)),
                                        userData={'name': 'bottom_wall', 'bodyType': 'static'})
        w.CreateFixture(shape=polygonShape(
            box=(WIDTH / 2, BORDER)), friction=0.05, restitution=0.98)
        self.walls.append(w)
        w = self.world.CreateStaticBody(position=(0, HEIGHT / 2), #shapes=polygonShape(box=(BORDER, HEIGHT/2)),
                                        userData={'name': 'left_wall', 'bodyType': 'static'})
        w.CreateFixture(shape=polygonShape(
            box=(BORDER, HEIGHT / 2)), friction=0.05, restitution=0.98)
        self.walls.append(w)
        w = self.world.CreateStaticBody(position=(WIDTH, HEIGHT / 2), #shapes=polygonShape(box=(BORDER, HEIGHT/2)),
                                        userData={'name': 'right_wall', 'bodyType': 'static'})
        w.CreateFixture(shape=polygonShape(
            box=(BORDER, HEIGHT / 2)), friction=0.05, restitution=0.98)
        self.walls.append(w)

    def update_condition(self,m=None,f=None):
        cond = {}
        loc_list = []
        vel_list = []
        for obj in ['o1', 'o2', 'o3', 'o4']:
            loc_list.append(
                {'x': self.data[obj]['x'][-1], 'y': self.data[obj]['y'][-1]})
            vel_list.append(
                {'x': self.data[obj]['vx'][-1], 'y': self.data[obj]['vy'][-1]})
        cond['sls'] = loc_list
        cond['svs'] = vel_list
        if(type(m) and type(f)):
            cond['mass'] = m
            cond['lf'] = f
        return cond

    def reset(self):
        time_stamp = self.T
        self.cond = self.cond_list[np.random.randint(len(self.cond_list))]
        self.update_bodies(self.cond)
        control_vec = {'obj': np.repeat(0, time_stamp), 'x': np.repeat(
            self.init_mouse[0], time_stamp), 'y': np.repeat(self.init_mouse[1], time_stamp)}
        true_key = (tuple(self.cond['mass']), tuple(np.array(self.cond['lf']).flatten()))
        true_data = {true_key: self.simulate_in_all(self.cond, control_vec)}
        self.data = self.initial_data(self.bodies)
        _, states = generate_trajectory(true_data,True)
        self.prior = copy.deepcopy(self.PRIOR)
        self.step_reward = []
        return states

    def initial_data(self,bodies = None,init_mouse = None):
        local_data = {}
        if(bodies==None):
            for i in range(0,len(self.bodies)):
                objname = 'o' + str(i + 1)
                local_data[objname] = {'x': [], 'y': [], 'vx': [], 'vy': [], 'rotation': []}
            local_data['co'] = []
            local_data['mouse'] = []
        else:
            # initialize data by specified bodies and init_mouse
            # for self.data
            for i in range(0,len(bodies)):
                objname = 'o' + str(i + 1)
                local_data[objname] = {'x': [bodies[i].position[0]], 'y': [bodies[i].position[1]], 'vx': [
                        bodies[i].linearVelocity[0]], 'vy': [bodies[i].linearVelocity[1]], 'rotation': [bodies[i].angle]}
            local_data['co'] = [0]
            if(init_mouse):
                local_data['mouse'] = {'x': [init_mouse[0]], 'y': [init_mouse[1]]}
            else:
                local_data['mouse'] = {'x': [self.init_mouse[0]], 'y': [self.init_mouse[1]]}
        return local_data

    def update_data(self,true_data,control_vec):
        for obj in ['o1', 'o2', 'o3', 'o4']:
            self.data[obj]['x'] += true_data[obj]['x']
            self.data[obj]['y'] += true_data[obj]['y']
            self.data[obj]['vx'] += true_data[obj]['vx']
            self.data[obj]['vy'] += true_data[obj]['vy']
            self.data[obj]['rotation'] += true_data[obj]['rotation']
        self.data['co'] += control_vec['obj'].tolist()
        self.data['mouse']['x'] += control_vec['x'].tolist()
        self.data['mouse']['y'] += control_vec['y'].tolist()

    def update_simulate_data(self,local_data):
        for i in range(0,len(self.bodies)):
            objname = 'o' + str(i + 1)
            local_data[objname]['x'].append(self.bodies[i].position[0])
            local_data[objname]['y'].append(self.bodies[i].position[1])
            local_data[objname]['vx'].append(self.bodies[i].linearVelocity[0])
            local_data[objname]['vy'].append(self.bodies[i].linearVelocity[1])
            local_data[objname]['rotation'].append(self.bodies[i].angle)
        return local_data

    def simulate(self, cond, control_vec, t):
        #Loop over the dynamic objects
        for i in range(0,len(self.bodies)):
            #Grab and print current object name and location
            objname = self.bodies[i].userData['name']
            #Apply local forces
            for j in range(0, len(self.bodies)):
                #NB: The force strengths should be symmetric i,j==j,i for normal physics
                #otherwise you'll get "chasing" behaviour
                strength = cond['lf'][i][j]
                #If there's a pairwise interaction between these two objects...
                if strength!=0 and i!=j:
                    #Calculate its force based on the objects masses and distances
                    m = self.bodies[i].mass * self.bodies[j].mass
                    d = ((self.bodies[i].position[0] - self.bodies[j].position[0])**2 +
                        (self.bodies[i].position[1] - self.bodies[j].position[1])**2)**0.5

                    angle = np.arctan2(self.bodies[i].position[1] - self.bodies[j].position[1],
                                self.bodies[i].position[0] - self.bodies[j].position[0])
                    f_mag = (strength * m) / d**2
                    f_vec = (f_mag * np.cos(angle), f_mag * np.sin(angle))
                    #Apply the force to the object
                    self.bodies[j].ApplyForce(force=f_vec, point=(0,0), wake=True)
            if control_vec['obj'][t]==(i+1):
                self.bodies[i].linearDamping = 10
                c_vec = ( (1/0.19634954631328583) * 0.2*(control_vec['x'][t] - self.bodies[i].position[0]),
                        (1/0.19634954631328583) * 0.2*(control_vec['y'][t] - self.bodies[i].position[1]))
                #Apply the force to the object
                self.bodies[i].ApplyLinearImpulse(impulse=c_vec, point=(0,0), wake=True)
                if t!=(len(control_vec['obj'])-1):
                    if control_vec['obj'][t+1]==0:
                        self.bodies[i].linearDamping = 0.05
            # Turned off all rotation but could include if we want
            self.bodies[i].angularVelocity = 0
            self.bodies[i].angle = 0

    def update_simulate_bodies(self,cond,control_vec,t,local_data):
        self.simulate(cond, control_vec, t)
        self.world.Step(TIME_STEP, 3, 3)
        self.world.ClearForces()
        local_data = self.update_simulate_data(local_data)
        return local_data

    def simulate_in_all(self, cond,control_vec):
        local_data = self.initial_data()
        for t in range(0, self.T):
            local_data = self.update_simulate_bodies(cond, control_vec,t,local_data)
        return local_data

    def update_bodies(self,cond):
        for i in range(0, len(cond['sls'])):
            objname = 'o' + str(i + 1)
            self.bodies[i].position = (cond['sls'][i]['x'], cond['sls'][i]['y'])
            self.bodies[i].linearDamping = 0.05
            self.bodies[i].linearVelocity = vec2(cond['svs'][i]['x'], cond['svs'][i]['y'])
            self.bodies[i].mass = cond['mass'][i]
            self.bodies[i].fixtures[0].density = cond['mass'][i]

    def step(self, action_idx, test_flag=False):
        obj, mouse_x, mouse_y = generate_action(
            self.data['mouse']['x'][-1], self.data['mouse']['y'][-1], action_idx, T = self.T)
        control_vec = {'obj': np.repeat(
            obj, self.T), 'x': np.array(mouse_x), 'y': np.array(mouse_y)}
        current_cond = self.update_condition(self.cond['mass'],self.cond['lf'])
        # true case
        true_key = (tuple(self.cond['mass']), tuple(np.array(self.cond['lf']).flatten()))
        self.update_bodies(current_cond)
        true_data = {true_key: self.simulate_in_all(current_cond, control_vec)}
        # simulate case
        simulate_data = {}
        for m in self.mass_list:
            for f in self.force_list:
                current_cond['mass'] = m
                current_cond['lf'] = f
                self.update_bodies(current_cond)
                key = (tuple(m), tuple(np.array(f).flatten()))
                simulate_data[key] = self.simulate_in_all(current_cond,control_vec)

        # Synchronize self.data to keep track of all steps from beginning to end
        #print("before update",self.data['o1']['x'])
        #print("update",true_data[true_key]['o1']['x'])
        self.update_data(true_data[true_key],control_vec)
        #print("after update",self.data['o1']['x'])
        #print("******************")
        true_trace, states = generate_trajectory(true_data,True)
        simulate_trace, _ = generate_trajectory(simulate_data,False)
        if(not test_flag):
            other_mode = 3 - self.ig_mode
            reward_others, _ = get_reward_ig(true_trace, simulate_trace, SIGMA, self.prior, other_mode, update_prior=False)
            reward, self.prior = get_reward_ig(true_trace, simulate_trace, SIGMA, self.prior, self.ig_mode, update_prior=True)
            self.step_reward.append(reward)
            #print("step reward: ", len(self.step_reward), np.sum(self.step_reward))
            current_time = len(self.data['o1']['x']) - 1
            if(current_time >= self.cond['timeout'] or np.sum(self.step_reward)>self.total_reward * self.reward_stop):
                #print(current_time)
                stop_flag = True
            else:
                stop_flag = False
            return states, reward, stop_flag, reward_others

    def step_data(self):
        return self.data
