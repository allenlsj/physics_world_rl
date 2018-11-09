#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
agent.py
"""

#IMPORT LIBRARIES
import Box2D  # The main library
import numpy as np
import random as rd
import copy
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody, vec2)
from config import *
from utility import *

class physic_env():
	def __init__(self,cond,mass_list,force_list):
		# --- pybox2d world setup ---

		# Create the world
		self.world = world(gravity=(0, 0), doSleep=True)
		self.bodies = []
		self.walls = []
		self.data = {}
		self.cond = cond
		self.add_pucks()
		self.add_static_walls()
		self.mass_list = mass_list
		self.force_list = force_list
		#self.simulate_state_dic = {}
		#self.true_state_dic = {}

	# --- add pucks (bodies) ---
	def add_pucks(self):
		for i in range(0, len(self.cond['sls'])):
			#Give each a unique name
		    objname = 'o' + str(i + 1)
		    #Create the body
		    b = self.world.CreateDynamicBody(position=(self.cond['sls'][i]['x'], self.cond['sls'][i]['y']),
		                                linearDamping = 0.05, fixedRotation=True,
		                                userData = {'name': objname, 'bodyType': 'dynamic'})
		    b.linearVelocity = vec2(self.cond['svs'][i]['x'], self.cond['svs'][i]['y'])
		    #Add the the shape 'fixture'
		    circle = b.CreateCircleFixture(radius=BALL_RADIUS,
		                                   density=self.cond['mass'][i],
		                                   friction=0.05, restitution=0.98)
		    b.mass = self.cond['mass'][i]
		    #Add it to our list of dynamic objects
		    self.bodies.append(b)
		    #Add a named entry in the data for this object
		    # init self.data by intial condition
		    self.data[objname] = {'x':[b.position[0]], 'y':[b.position[1]], 'vx':[b.linearVelocity[0]], 'vy':[b.linearVelocity[1]], 'rotation':[b.angle]} 

		self.data['co'] = [] #Add an entry for controlled object's ID (0 none, 1-4 are objects 'o1'--'o4')
		self.data['mouse'] = {'x':[], 'y':[]} #Add an entry for the mouse position


	# --- add static walls ---
	def add_static_walls(self):
		w = self.world.CreateStaticBody(position=(WIDTH/2, 0),  shapes=polygonShape(box=(WIDTH/2, BORDER)), 
		                           userData = {'name':'top_wall', 'bodyType':'static'})
		w.CreateFixture(shape=polygonShape(box=(WIDTH/2, BORDER)), friction = 0.05, restitution = 0.98)
		self.walls.append(w)
		w = self.world.CreateStaticBody(position=(WIDTH/2, HEIGHT), #shapes=polygonShape(box=(WIDTH/2, BORDER)), 
		                           userData = {'name':'bottom_wall', 'bodyType':'static'})
		w.CreateFixture(shape=polygonShape(box=(WIDTH/2, BORDER)), friction = 0.05, restitution = 0.98)
		self.walls.append(w)
		w = self.world.CreateStaticBody(position=(0, HEIGHT/2), #shapes=polygonShape(box=(BORDER, HEIGHT/2)), 
		                           userData = {'name':'left_wall', 'bodyType':'static'})
		w.CreateFixture(shape=polygonShape(box=(BORDER, HEIGHT/2)), friction = 0.05, restitution = 0.98)
		self.walls.append(w)
		w = self.world.CreateStaticBody(position=(WIDTH, HEIGHT/2), #shapes=polygonShape(box=(BORDER, HEIGHT/2)), 
		                           userData = {'name':'right_wall', 'bodyType':'static'})
		w.CreateFixture(shape=polygonShape(box=(BORDER, HEIGHT/2)), friction = 0.05, restitution = 0.98)
		self.walls.append(w)

	def generate_states(self, cond, control_vec,t):
		bodies = []
		for i in range(0, len(cond['sls'])):
			#Give each a unique name
			objname = 'o' + str(i + 1)
			#Create the body
			b = self.world.CreateDynamicBody(position=(cond['sls'][i]['x'], cond['sls'][i]['y']),
	                                linearDamping = 0.05, fixedRotation=True,
	                                userData = {'name': objname, 'bodyType': 'dynamic'})
			b.linearVelocity = vec2(cond['svs'][i]['x'], cond['svs'][i]['y'])
			#Add the the shape 'fixture'
			# circle = b.CreateCircleFixture(radius=BALL_RADIUS,
			#                            density=cond['mass'][i],
			#                            friction=0.05, restitution=0.98)
			b.mass = cond['mass'][i]
			#Add it to our list of dynamic objects
			bodies.append(b)
		bodies = simulate(bodies, cond, control_vec, t)
		#print(t,"===============",bodies[0].position)
		#Update the world
		self.world.Step(TIME_STEP, 3, 3)
		#Remove any forces applied at the previous timepoint (these will be recalculated and reapplied below)
		self.world.ClearForces()
		#print(t,"~~~~~~~~~~~~~~~~",bodies[0].position)
		#Store the position and velocity of object i
		local_data = store_state(bodies)
		bodies[i].angularVelocity = 0 #Turned off all rotation but could include if we want
		bodies[i].angle = 0
		return local_data

	def step(self,control_vec,time_stamp = 10):
	    simulate_state_dic_list = []
	    true_state_dic_list = []
	    # current_time
	    current_time = len(self.data['o1']['x'])
	    print("current time",current_time)
	    current_initial_state = store_state(self.bodies)
	    for t in range(0,time_stamp):
	    	# Update the world
	    	# self.world.Step(TIME_STEP, 3, 3)
	    	# self.world.ClearForces()
	    	# print("^^^",t,self.bodies[0].position[0])
	    	# Simulate
	    	print(t)
	        simulate_state_dic = {}
	        true_state_dic = {}
	        for m in self.mass_list:
	            for f in self.force_list:
	            	cond = self.update_condition(m,f)
	                simulate_state_dic[(tuple(m),tuple(np.array(f).flatten()))] = self.generate_states(cond,control_vec,t)
	                #data = copy.deepcopy(self.data)
	        simulate_state_dic_list.append(simulate_state_dic)
	        
	        #data = self.data
	        #Loop over the dynamic objects
	        bodies = simulate(self.bodies, self.cond, control_vec, t)
	        self.world.Step(TIME_STEP, 3, 3)
	    	self.world.ClearForces()

	        for i in range(0,len(bodies)):
	            objname = bodies[i].userData['name']
	            #Store the position and velocity of object i
	            self.data[objname]['x'].append(bodies[i].position[0])
	            self.data[objname]['y'].append(bodies[i].position[1])
	            self.data[objname]['vx'].append(bodies[i].linearVelocity[0])
	            self.data[objname]['vy'].append(bodies[i].linearVelocity[1])
	            self.data[objname]['rotation'].append(bodies[i].angle)

	            self.bodies[i].angularVelocity = 0 #Turned off all rotation but could include if we want
	            self.bodies[i].angle = 0
	        #Store the target of the controller (i.e. is one of the objects selected?)
	        #And the current position of the controller (i.e. mouse)
	        self.data['co'].append(control_vec['obj'][t])
	        self.data['mouse']['x'].append(control_vec['x'][t])
	        self.data['mouse']['y'].append(control_vec['y'][t])
	        true_state_dic[(tuple(cond['mass']),tuple(np.array(cond['lf']).flatten()))] = store_state(bodies)
	        #import ipdb;ipdb.set_trace()
	        true_state_dic_list.append(true_state_dic)
	        #print("&&&&&&&&&&&&&&",true_state_dic)
	    if(simulate_state_dic_list):
	        diff_state = []
	        true_diff_state = []
	        # change: from t to t+9
	        for time in range(0,time_stamp):
	            diff_state_dic = {}
	            true_diff_state_dic = {}
	            for key in true_state_dic_list[0]:
	                true_diff_obj_dic = {}
	                if(time == 0):
		            	old_state = current_initial_state
	                else:
	                	old_state = true_state_dic_list[time-1][key]
	                for obj in ['o1','o2','o3','o4']:
	                    true_diff_r_theta = {}
	                    current_state = true_state_dic_list[time][key][obj]
	                    # if(obj == 'o1'):
	                    # 	print(time,"#################true current state",old_state[obj]['x'][-1],current_state['x'][-1])
	                    true_diff_r_theta['r'] = np.sqrt((current_state['x'][-1] - old_state[obj]['x'][-1])**2 + (current_state['y'][-1] - old_state[obj]['y'][-1])**2)
	                    true_diff_r_theta['rotation'] = current_state['rotation'][-1] - old_state[obj]['rotation'][-1]
	                    true_diff_r_theta['rotation'] = true_diff_r_theta['rotation'] + 2 * np.pi if true_diff_r_theta['rotation'] < 0 else true_diff_r_theta['rotation']
	                    true_diff_obj_dic[obj] = true_diff_r_theta
	                true_diff_state_dic[key] = true_diff_obj_dic
	            true_diff_state.append(true_diff_state_dic)
	            
	            for key in simulate_state_dic_list[0]:
	                diff_obj_dic = {}
	                for obj in ['o1','o2','o3','o4']:
	                    diff_r_theta = {}
	                    current_state = simulate_state_dic_list[time][key][obj]
	                    # if(obj== 'o1'):
	                    # 	print(time,"*********simulate current state",old_state[obj]['x'][-1],current_state['x'][-1])
	                    diff_r_theta['r'] = np.sqrt((current_state['x'][-1] - old_state[obj]['x'][-1])**2 + (current_state['y'][-1] - old_state[obj]['y'][-1])**2)
	                    diff_r_theta['rotation'] = current_state['rotation'][-1] - old_state[obj]['rotation'][-1] 
	                    diff_r_theta['rotation'] = diff_r_theta['rotation'] + 2 * np.pi if diff_r_theta['rotation'] < 0 else diff_r_theta['rotation'] 
	                    diff_obj_dic[obj] = diff_r_theta
	                diff_state_dic[key] = diff_obj_dic
	                #print("simulate diff ",diff_obj_dic)
	                #print("*********simulate current state",current_state[time][key]['o1'])
	            diff_state.append(diff_state_dic)          
	        #print "****************Rewards************:{}".format(get_reward(true_diff_state,diff_state))
	        #wreward,freward = get_reward(true_diff_state,diff_state)
	    	reward = get_reward(true_diff_state,diff_state,SIGMA)
	    	PD_mass = get_reward(true_diff_state,diff_state,SIGMA,2)
	    	PD_force = get_reward(true_diff_state,diff_state,SIGMA,3)
	    # return reward
	    return PD_mass,PD_force,true_state_dic_list
	        #print "****************Rewards************:{}".format(Reward)
	def update_condition(self, m, f):
		cond = {}
		loc_list = []
		vel_list = []
		for obj in ['o1','o2','o3','o4']:
			loc_list.append({'x':self.data[obj]['x'][-1],'y':self.data[obj]['y'][-1]})
			vel_list.append({'x':self.data[obj]['vx'][-1],'y':self.data[obj]['vy'][-1]})
		cond['sls'] = loc_list
		cond['svs'] = vel_list
		cond['mass'] = m
		cond['lf'] = f
		return cond

	def reset(self,time_stamp = 15):
		self.bodies = []
		self.data = {}
		self.add_pucks()
		true_state_list = []
		control_vec = {'obj': np.repeat(0, time_stamp), 'x':np.repeat(3, time_stamp), 'y':np.repeat(3, time_stamp)}
		for t in range(0,time_stamp):
			self.world.Step(TIME_STEP, 3, 3)
			self.world.ClearForces()
			#print(self.bodies[0].position[0])
			bodies = simulate(self.bodies, self.cond, control_vec, t)
			for i in range(0,len(bodies)):
				objname = bodies[i].userData['name']
				r = np.sqrt((bodies[i].position[0] - self.data[objname]['x'][-1])**2 + (bodies[i].position[1]- self.data[objname]['y'][-1])**2)
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

				self.bodies[i].angularVelocity = 0 #Turned off all rotation but could include if we want
				self.bodies[i].angle = 0
				
	        #Store the target of the controller (i.e. is one of the objects selected?)
	        #And the current position of the controller (i.e. mouse)
	        self.data['co'].append(control_vec['obj'][t])
	        self.data['mouse']['x'].append(control_vec['x'][t])
	        self.data['mouse']['y'].append(control_vec['y'][t])
		print(len(true_state_list))
		return true_state_list



