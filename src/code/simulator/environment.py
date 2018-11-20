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
from utility import gaussian, store_state, simulate,update_simulate_data,generate_trajectory
from action_generator import generate_action
from information_gain import *
import time

class physic_env():
    def __init__(self, cond, mass_list, force_list, init_mouse, time_stamp, ig_mode, prior):
        # --- pybox2d world setup ---

        # Create the world
        self.world = world(gravity=(0, 0), doSleep=True)
        self.bodies = []
        self.walls = []
        self.data = {}
        self.cond = cond
        self.init_mouse = init_mouse
        self.add_pucks()
        self.add_static_walls()
        self.mass_list = mass_list
        self.force_list = force_list
        self.T = time_stamp
        self.ig_mode = ig_mode
        self.prior = prior
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
        w = self.world.CreateStaticBody(position=(WIDTH / 2, 0), shapes=polygonShape(box=(WIDTH / 2, BORDER)),
                                        userData={'name': 'top_wall', 'bodyType': 'static'})
        w.CreateFixture(shape=polygonShape(
            box=(WIDTH / 2, BORDER)), friction=0.05, restitution=0.98)
        self.walls.append(w)
        w = self.world.CreateStaticBody(position=(WIDTH / 2, HEIGHT),  # shapes=polygonShape(box=(WIDTH/2, BORDER)),
                                        userData={'name': 'bottom_wall', 'bodyType': 'static'})
        w.CreateFixture(shape=polygonShape(
            box=(WIDTH / 2, BORDER)), friction=0.05, restitution=0.98)
        self.walls.append(w)
        w = self.world.CreateStaticBody(position=(0, HEIGHT / 2),  # shapes=polygonShape(box=(BORDER, HEIGHT/2)),
                                        userData={'name': 'left_wall', 'bodyType': 'static'})
        w.CreateFixture(shape=polygonShape(
            box=(BORDER, HEIGHT / 2)), friction=0.05, restitution=0.98)
        self.walls.append(w)
        w = self.world.CreateStaticBody(position=(WIDTH, HEIGHT / 2),  # shapes=polygonShape(box=(BORDER, HEIGHT/2)),
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
        self.bodies = []
        self.data = {}
        self.add_pucks()
        true_state_list = []
        control_vec = {'obj': np.repeat(0, time_stamp), 'x': np.repeat(
            3, time_stamp), 'y': np.repeat(3, time_stamp)}
        for t in range(0, time_stamp):
            self.world.Step(TIME_STEP, 3, 3)
            self.world.ClearForces()
            # print(self.bodies[0].position[0])
            bodies = simulate(self.bodies, self.cond, control_vec, t)
            for i in range(0, len(bodies)):
                objname = bodies[i].userData['name']
                r = np.sqrt((bodies[i].position[0] - self.data[objname]['x'][-1])
                            ** 2 + (bodies[i].position[1] - self.data[objname]['y'][-1])**2)
                # print("^^^",r)
                theta = bodies[i].angle - self.data[objname]['rotation'][-1]
                theta = theta + 2 * np.pi if theta < 0 else theta
                true_state_list.append(r)
                true_state_list.append(theta)
                self.data[objname]['x'].append(bodies[i].position[0])
                self.data[objname]['y'].append(bodies[i].position[1])
                self.data[objname]['vx'].append(bodies[i].linearVelocity[0])
                self.data[objname]['vy'].append(bodies[i].linearVelocity[1])
                self.data[objname]['rotation'].append(bodies[i].angle)

                # Turned off all rotation but could include if we want
                self.bodies[i].angularVelocity = 0
                self.bodies[i].angle = 0

        # Store the target of the controller (i.e. is one of the objects selected?)
        # And the current position of the controller (i.e. mouse)
        self.data['co'].append(control_vec['obj'][t])
        self.data['mouse']['x'].append(control_vec['x'][t])
        self.data['mouse']['y'].append(control_vec['y'][t])
        #print(len(true_state_list))
        return true_state_list
    def initial_simulate(self, cond, control_vec):
        '''
        generate simulated bodies and initialize them with intial condition(m,f) and control
        '''
        bodies = []
        for i in range(0, len(cond['sls'])):
                # Give each a unique name
            objname = 'o' + str(i + 1)
            # Create the body
            b = self.world.CreateDynamicBody(position=(cond['sls'][i]['x'], cond['sls'][i]['y']),
                                             linearDamping=0.05, fixedRotation=True,
                                             userData={'name': objname, 'bodyType': 'dynamic'})
            b.linearVelocity = vec2(cond['svs'][i]['x'], cond['svs'][i]['y'])
            # Add the the shape 'fixture'
            circle = b.CreateCircleFixture(radius=BALL_RADIUS,
                                       density=cond['mass'][i],
                                       friction=0.05, restitution=0.98)
            b.mass = cond['mass'][i]
            # Add it to our list of dynamic objects
            bodies.append(b)
        local_data = self.initial_data()
        return bodies,local_data

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
        self.data['co'] += control_vec['obj']
        self.data['mouse']['x'] += control_vec['x']
        self.data['mouse']['y'] += control_vec['y']

    
    def update_simulate_bodies(self,bodies,cond,control_vec,t,local_data):
        bodies = simulate(bodies, cond, control_vec, t)
        self.world.Step(TIME_STEP, 3, 3)
        self.world.ClearForces()
        local_data = update_simulate_data(local_data,bodies)
        return bodies,local_data


    def step(self, action_idx):
        obj, mouse_x, mouse_y = generate_action(
            self.data['mouse']['x'][-1], self.data['mouse']['y'][-1], action_idx, T = self.T)
        control_vec = {'obj': np.repeat(
            obj, self.T), 'x': mouse_x, 'y': mouse_y}
        # initial true case
        true_key = (tuple(self.cond['mass']), tuple(np.array(self.cond['lf']).flatten()))
        true_data = {true_key: self.initial_data()}

        simulate_bodies = {}
        simulate_data = {}
        # correct simulated trajectory to true trajectory
        current_cond = self.update_condition()
        for t in range(0, self.T):
            simulate_state_dic = {}
            true_state_dic = {}
            # Simulate Cases
            idx = 0
            #print("begin simulate bodies",t,time.strftime("%H:%M:%S", time.localtime()))
            for m in self.mass_list:
                for f in self.force_list:
                    idx += 1
                    current_cond['mass'] = m
                    current_cond['lf'] = f
                    key = (tuple(m), tuple(np.array(f).flatten()))
                    if(t == 0):
                        simulate_bodies[key], simulate_data[key] = self.initial_simulate(current_cond, control_vec)
                    #print("simulate bodies",t,idx,time.strftime("%H:%M:%S", time.localtime()))
                    simulate_bodies[key],simulate_data[key] = self.update_simulate_bodies(simulate_bodies[key],current_cond,control_vec,t,simulate_data[key])
            #print("true bodies",t,time.strftime("%H:%M:%S", time.localtime()))
            # True Case
            self.bodies,true_data[true_key] = self.update_simulate_bodies(self.bodies,self.cond, control_vec, t,true_data[true_key])
        # Synchronize self.data to keep track of all steps from beginning to end
        self.update_data(true_data[true_key],control_vec)
        true_trace, states = generate_trajectory(true_data,True)
        simulate_trace, _ = generate_trajectory(simulate_data,False)
        reward, self.prior = get_reward_ig(true_trace, simulate_trace, SIGMA, self.prior, self.ig_mode)
        current_time = len(self.data['o1']['x']) - 1
        if(current_time >= self.cond['timeout']):
            stop_flag = True
        else:
            stop_flag = False
        return states, reward, stop_flag

