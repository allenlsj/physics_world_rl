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
from prior import *

class physic_env():
	def __init__(self,cond,mass_list,force_list,prior):
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
		self.force_list = force_list;self.prior = prior
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
		    self.data[objname] = {'x':[], 'y':[], 'vx':[], 'vy':[], 'rotation':[]} 

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


	def generate_states(self, cond, local_data, control_vec,t):
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
			#circle = b.CreateCircleFixture(radius=BALL_RADIUS,
			#                            density=cond['mass'][i],
			#                            friction=0.05, restitution=0.98)
			b.mass = cond['mass'][i]
			#Add it to our list of dynamic objects
			bodies.append(b)

		#Loop over the dynamic objects
		for i in range(0,len(bodies)):
			#Grab and print current object name and location
			objname = bodies[i].userData['name']
			# print (objname, bodies[i].position)

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
					#Print the calculated values
					# if i==0:
					#     print (i,j, 'force', strength, m, d,
					#            'rounded distance y', round(bodies[i].position[1] - bodies[j].position[1], 3),
					#            'rounded distance x', round(bodies[i].position[0] - bodies[j].position[0], 3),
					#            'angle', angle, f_mag, f_vec)
					
					#Apply the force to the object
					bodies[j].ApplyForce(force=f_vec, point=(0,0), wake=True)

			if control_vec['obj'][t]==(i+1):
				bodies[i].linearDamping = 10

				c_vec = ( (1/0.19634954631328583) * 0.2*(control_vec['x'][t] - self.bodies[i].position[0]), 
						(1/0.19634954631328583) * 0.2*(control_vec['y'][t] - self.bodies[i].position[1]))
				#Print the calculated values
				#print (t, i, 'control force', self.bodies[i].position[0], self.bodies[i].position[1], self.bodies[i].angle, c_vec)

				#Apply the force to the object
				bodies[i].ApplyLinearImpulse(impulse=c_vec, point=(0,0), wake=True)
				if t!=(len(control_vec['obj'])-1):
					if control_vec['obj'][t+1]==0:
						self.bodies[i].linearDamping = 0.05

				#Store the position and velocity of object i
            local_data[objname]['x'].append(bodies[i].position[0])
            local_data[objname]['y'].append(bodies[i].position[1])
            local_data[objname]['vx'].append(bodies[i].linearVelocity[0])
            local_data[objname]['vy'].append(bodies[i].linearVelocity[1])
            local_data[objname]['rotation'].append(bodies[i].angle)

            bodies[i].angularVelocity = 0 #Turned off all rotation but could include if we want
            bodies[i].angle = 0
        return local_data

	def Step10(self,control_vec,time_stamp = 10):
	    simulate_state_dic_list = []
	    true_state_dic_list = [];
	    for t in range(time_stamp):

	        cond = copy.deepcopy(self.cond)
	        bodies = self.bodies[:]
	        data = copy.deepcopy(self.data)

	        simulate_state_dic = {}
	        true_state_dic = {}
	        for m in self.mass_list:
	            for f in self.force_list:
	                cond['mass'] = m
	                cond['lf'] = f
	                simulate_state_dic[(tuple(m),tuple(np.array(f).flatten()))] = self.generate_states(cond,data,control_vec,t)
	                data = copy.deepcopy(self.data)
	        simulate_state_dic_list.append(simulate_state_dic)

	        cond = self.cond
	        bodies = self.bodies
	        data = self.data

	        #Loop over the dynamic objects
	        for i in range(0,len(bodies)):
	            #Grab and print current object name and location
	            objname = bodies[i].userData['name']
	            # print (objname, bodies[i].position)

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
	                    #Print the calculated values
	                    #if i==0:
	                    #    print (i,j, 'force', strength, m, d,
	                    #        'rounded distance y', round(bodies[i].position[1] - bodies[j].position[1], 3),
	                    #        'rounded distance x', round(bodies[i].position[0] - bodies[j].position[0], 3),
	                    #        'angle', angle, f_mag, f_vec)
	                    
	                    #Apply the force to the object
	                    bodies[j].ApplyForce(force=f_vec, point=(0,0), wake=True)

	            if control_vec['obj'][t]==(i+1):
	                bodies[i].linearDamping = 10

	                c_vec = ( (1/0.19634954631328583) * 0.2*(control_vec['x'][t] - bodies[i].position[0]), 
	                        (1/0.19634954631328583) * 0.2*(control_vec['y'][t] - bodies[i].position[1]))
	                #Print the calculated values
	                #print (t, i, 'control force', bodies[i].position[0], bodies[i].position[1], bodies[i].angle, c_vec)

	                #Apply the force to the object
	                bodies[i].ApplyLinearImpulse(impulse=c_vec, point=(0,0), wake=True)
	                if t!=(len(control_vec['obj'])-1):
	                    if control_vec['obj'][t+1]==0:
	                        bodies[i].linearDamping = 0.05

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
	        true_state_dic[(tuple(cond['mass']),tuple(np.array(cond['lf']).flatten()))] = copy.deepcopy(self.data)
	        #import ipdb;ipdb.set_trace()
	        true_state_dic_list.append(true_state_dic)
	    
	    if(simulate_state_dic_list):
	        diff_state = []
	        true_diff_state = []
	        for time in range(1,10):
	            diff_state_dic = {}
	            true_diff_state_dic = {}
	            for key in true_state_dic_list[0]:
	                true_diff_obj_dic = {}
	                for obj in ['o1','o2','o3','o4']:
	                    true_diff_r_theta = {}
	                    current_state = true_state_dic_list[time][key][obj]
	                    old_state = true_state_dic_list[time-1][key][obj]
	                    true_diff_r_theta['r'] = np.sqrt((current_state['x'][-1] - old_state['x'][-1])**2 + (current_state['y'][-1] - old_state['y'][-1])**2)
	                    true_diff_r_theta['rotation'] = current_state['rotation'][-1] - old_state['rotation'][-1]
	                    true_diff_r_theta['rotation'] = true_diff_r_theta['rotation'] + 2 * np.pi if true_diff_r_theta['rotation'] < 0 else true_diff_r_theta['rotation']
	                    true_diff_obj_dic[obj] = true_diff_r_theta
	                true_diff_state_dic[key] = true_diff_obj_dic
	            true_diff_state.append(true_diff_state_dic)
	            for key in simulate_state_dic_list[0]:
	                diff_obj_dic = {}
	                for obj in ['o1','o2','o3','o4']:
	                    diff_r_theta = {}
	                    current_state = simulate_state_dic_list[time][key][obj]
	                    diff_r_theta['r'] = np.sqrt((current_state['x'][-1] - old_state['x'][-1])**2 + (current_state['y'][-1] - old_state['y'][-1])**2)
	                    diff_r_theta['rotation'] = current_state['rotation'][-1] - old_state['rotation'][-1] 
	                    diff_r_theta['rotation'] = diff_r_theta['rotation'] + 2 * np.pi if diff_r_theta['rotation'] < 0 else diff_r_theta['rotation'] 
	                    diff_obj_dic[obj] = diff_r_theta
	                diff_state_dic[key] = diff_obj_dic
	            diff_state.append(diff_state_dic)          
	        #print "****************Rewards************:{}".format(get_reward(true_diff_state,diff_state))
	        #wreward,freward = get_reward(true_diff_state,diff_state)
	    	reward, self.prior = get_reward(true_diff_state,diff_state,SIGMA, self.prior)
            print "****************Rewards************:{}".format(reward)
	    return reward
	    
    
	def reset(self):
		#Update the world         
		self.world.Step(TIME_STEP, 3, 3)
		#Remove any forces applied at the previous timepoint (these will be recalculated and reapplied below)
		self.world.ClearForces()