#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
control_world.py
"""

#IMPORT LIBRARIES
import Box2D  # The main library
import numpy as np
import json
import random as rd
import math
import gizeh as gz
import numpy as np
import moviepy.editor as mpy
from os import listdir
import matplotlib
import matplotlib.pyplot as plt

#For js implementation
import pyduktape

from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody, vec2)
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)

# --- Set constants ---
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
WIDTH, HEIGHT = 6, 4
BALL_RADIUS = 0.25
BORDER = 0.20

# --- SET STARTING CONDITIONS --- 
# sls = starting locations, svs = starting velocities, lf = local forces, mass = object densities
cond = {'sls':[{'x':1, 'y':1}, {'x':2, 'y':1}, {'x':1, 'y':2}, {'x':2, 'y':2}],
        'svs':[{'x':0, 'y':0}, {'x':0, 'y':0}, {'x':0, 'y':0}, {'x':0, 'y':0}],
        'lf':[[0, 3, 0, -3],
              [3, 0, 0, 0],
              [0, 0, 0, -3],
              [-3, 0, -3, 0]],
        'mass':[1,2,1,1],
        'timeout': 240
    }
# cond = {'sls':[{'x':1., 'y':1.1}, {'x':2.0, 'y':1.0}, {'x':1.0, 'y':3.0}, {'x':2.0, 'y':3.0}],
#     'svs':[{'x':0.0, 'y':0.0}, {'x':0.0, 'y':0.0}, {'x':0.0, 'y':0.0}, {'x':0.0, 'y':0.0}],
#     'lf':[[0, 10, 0, 0],
#           [10, 0, 0, 0],
#           [0, 0, 0, 0],
#           [0, 0, 0, 0]],
#     'mass':[1,1,1,1],
#     'timeout': 240}
    
# Collisions but no forces
# cond = {'sls':[{'x':1.0, 'y':1.0}, {'x':2.0, 'y':1.0}, {'x':1.0, 'y':3.0}, {'x':2.0, 'y':3.0}],
#     'svs':[{'x':2.0, 'y':0.0}, {'x':-3.0, 'y':0.0}, {'x':2.0, 'y':0.1}, {'x':0.0, 'y':0.0}],
#     'lf':[[0.0, 0.0, 0.0, 0.0],
#           [0.0, 0.0, 0.0, 0.0],
#           [0.0, 0.0, 0.0, 0.0],
#           [0.0, 0.0, 0.0, 0.0]],
#     'mass':[1.0,1.0,1.0,1.0],
#     'timeout': 240
# } 

# --- pybox2d world setup ---

# Create the world
world = world(gravity=(0, 0), doSleep=True)
bodies = []
walls = []
data = {}

# --- add pucks (bodies) ---
for i in range(0, len(cond['sls'])):
    #Give each a unique name
    objname = 'o' + str(i + 1)
    #Create the body
    b = world.CreateDynamicBody(position=(cond['sls'][i]['x'], cond['sls'][i]['y']),
                                linearDamping = 0.05, fixedRotation=True,
                                userData = {'name': objname, 'bodyType': 'dynamic'})
    b.linearVelocity = vec2(cond['svs'][i]['x'], cond['svs'][i]['y'])
    #Add the the shape 'fixture'
    circle = b.CreateCircleFixture(radius=BALL_RADIUS,
                                   density=cond['mass'][i],
                                   friction=0.05, restitution=0.98)
    b.mass = cond['mass'][i]
    #Add it to our list of dynamic objects
    bodies.append(b)
    #Add a named entry in the data for this object
    data[objname] = {'x':[], 'y':[], 'vx':[], 'vy':[], 'rotation':[]};

data['co'] = [] #Add an entry for controlled object's ID (0 none, 1-4 are objects 'o1'--'o4')
data['mouse'] = {'x':[], 'y':[]} #Add an entry for the mouse position

# --- add static walls ---
w = world.CreateStaticBody(position=(WIDTH/2, 0),  shapes=polygonShape(box=(WIDTH/2, BORDER)), 
                           userData = {'name':'top_wall', 'bodyType':'static'})
w.CreateFixture(shape=polygonShape(box=(WIDTH/2, BORDER)), friction = 0.05, restitution = 0.98)
walls.append(w)
w = world.CreateStaticBody(position=(WIDTH/2, HEIGHT), #shapes=polygonShape(box=(WIDTH/2, BORDER)), 
                           userData = {'name':'bottom_wall', 'bodyType':'static'})
w.CreateFixture(shape=polygonShape(box=(WIDTH/2, BORDER)), friction = 0.05, restitution = 0.98)
walls.append(w)
w = world.CreateStaticBody(position=(0, HEIGHT/2), #shapes=polygonShape(box=(BORDER, HEIGHT/2)), 
                           userData = {'name':'left_wall', 'bodyType':'static'})
w.CreateFixture(shape=polygonShape(box=(BORDER, HEIGHT/2)), friction = 0.05, restitution = 0.98)
walls.append(w)
w = world.CreateStaticBody(position=(WIDTH, HEIGHT/2), #shapes=polygonShape(box=(BORDER, HEIGHT/2)), 
                           userData = {'name':'right_wall', 'bodyType':'static'})
w.CreateFixture(shape=polygonShape(box=(BORDER, HEIGHT/2)), friction = 0.05, restitution = 0.98)
walls.append(w)

#---For demonstration purposes, some random control---
control_vec = {'obj': np.append(np.repeat(0, 60), np.repeat(1, 180)), 'x':np.repeat(3, 240), 'y':np.repeat(3, 240)}
# control_vec = {'obj': np.repeat(0, 240), 'x':np.repeat(3, 240), 'y':np.repeat(3, 240)}

# --- Run the main game loop ---
for t in range(0,cond['timeout']):

    #TODO:
    #Store the current state of the world (object locations & velocities)
    #Loop over mass in [1,1,1,1], [1,2,1,1], [2,1,1,1] and local forces in (-3,0,3) for all 6 pairs of objects
    #Set the objects to have these properties (i.e. bodies['o1'].mass = 1 etc, and cond['lf][i][j]==3'])
    #Step forward 10 steps or so
    #Store the final simulated locations and r and theta of motion of the objects each time
    #return the objects to their current state and repeat for next setting
    #Use the simulated locations to compute PD and entropy as in equations in paper.  Return these as the reward value for this period
    #Move the actual simulation forward 10 more steps and repeat...
    #When this is working.  Add an action space for sampling the control for each 10 frame period (initially choosing random control actions each time)

    #Update the world         
    world.Step(TIME_STEP, 3, 3)
    #Remove any forces applied at the previous timepoint (these will be recalculated and reapplied below)
    world.ClearForces()

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
                if i==0:
                    print (i,j, 'force', strength, m, d,
                           'rounded distance y', round(bodies[i].position[1] - bodies[j].position[1], 3),
                           'rounded distance x', round(bodies[i].position[0] - bodies[j].position[0], 3),
                           'angle', angle, f_mag, f_vec)
                
                #Apply the force to the object
                bodies[j].ApplyForce(force=f_vec, point=(0,0), wake=True)

        if control_vec['obj'][t]==(i+1):
            bodies[i].linearDamping = 10

            c_vec = ( (1/0.19634954631328583) * 0.2*(control_vec['x'][t] - bodies[i].position[0]), 
                     (1/0.19634954631328583) * 0.2*(control_vec['y'][t] - bodies[i].position[1]))
            #Print the calculated values
            print (t, i, 'control force', bodies[i].position[0], bodies[i].position[1], bodies[i].angle, c_vec)

            #Apply the force to the object
            bodies[i].ApplyLinearImpulse(impulse=c_vec, point=(0,0), wake=True)
            if t!=(len(control_vec['obj'])-1):
                if control_vec['obj'][t+1]==0:
                    bodies[i].linearDamping = 0.05

        #Store the position and velocity of object i
        data[objname]['x'].append(bodies[i].position[0])
        data[objname]['y'].append(bodies[i].position[1])
        data[objname]['vx'].append(bodies[i].linearVelocity[0])
        data[objname]['vy'].append(bodies[i].linearVelocity[1])
        data[objname]['rotation'].append(bodies[i].angle)

        bodies[i].angularVelocity = 0 #Turned off all rotation but could include if we want
        bodies[i].angle = 0

    #Store the target of the controller (i.e. is one of the objects selected?)
    #And the current position of the controller (i.e. mouse)
    data['co'].append(control_vec['obj'][t])
    data['mouse']['x'].append(control_vec['x'][t])
    data['mouse']['y'].append(control_vec['y'][t])



 
#########################
## Below is the original js environment.  It gives slightly different results.
#If anyone can get to the bottom of why that would be great.
########################

#--- Create JS environment ---
context = pyduktape.DuktapeContext()


#--- Import JS library ---
js_file = open("../js/box2d.js")
js = file.read(js_file)
context.eval_js(js)
# --- Load the JS script --- 
js_file = open("../js/control_world.js")
js = file.read(js_file)
context.eval_js(js)
# --- Set the condition ---
context.set_globals(cond=cond)
context.set_globals(control_path=control_vec)
# --- Run the simulation --- 
json_data = context.eval_js("Run()")
js_data = json.loads(json_data)



############################
# Draw the scene
#############################

#Set constants
RATIO = 100
RAD = 25
W = 600
H = 400
# H_outer = 500
N_OBJ=4

this_data = data

colors = [(1,0,0,),(0,1,0),(0,0,1),(0,0,1)]

#print colors, np.random.rand(N_OBJ, 3)
labels = ['A','B','','']

centers = np.array(['o1','o2','o3','o4'])
#random.randint(0,W, (nballs,2))

def make_frame(t):
    
    frame = int(math.floor(t*60))#*60
    print frame

    #Essentially pauses the action if there are no more frames and but more clip duration
    if frame >= len(this_data["co"]):
        frame = len(this_data["co"])-1

    #White background
    surface = gz.Surface(W,H, bg_color=(1,1,1))            
    
    #Walls
    wt = gz.rectangle(lx=W, ly=20, xy=(W/2,10), fill=(0,0,0))#, angle=Pi/8
    wb = gz.rectangle(lx=W, ly=20, xy=(W/2,H-10), fill=(0,0,0))
    wl = gz.rectangle(lx=20, ly=H, xy=(10,H/2), fill=(0,0,0))
    wr = gz.rectangle(lx=20, ly=H, xy=(W-10,H/2), fill=(0,0,0))
    wt.draw(surface)
    wb.draw(surface)
    wl.draw(surface)
    wr.draw(surface)

    #Pucks
    for label, color, center in zip(labels, colors, centers):

        xy = np.array([this_data[center]['x'][frame]*RATIO, this_data[center]['y'][frame]*RATIO])

        ball = gz.circle(r=RAD, fill=color).translate(xy)
        ball.draw(surface)

        #Draw the letters
        text = gz.text(label, fontfamily="Helvetica",  fontsize=25, fontweight='bold', fill=(0,0,0), xy=xy) #, angle=Pi/12
        text.draw(surface)

        #Draw an orientation marker
        xym = (xy[0]+RAD*np.cos(this_data[center]['rotation'][frame]), xy[1]+RAD*np.sin(this_data[center]['rotation'][frame]))
        marker = gz.circle(r=RAD/5, fill=(0,0,0)).translate(xym)
        marker.draw(surface)

        #Overlay the results from the JS implementation
        xy = np.array([js_data['physics'][center]['x'][frame+1]*RATIO, js_data['physics'][center]['y'][frame+1]*RATIO])
        ball = gz.circle(r=RAD, fill=(color[0], color[1], color[2], 0.3)).translate(xy)
        ball.draw(surface)
        #Draw an orientation marker
        # xym = (xy[0]+RAD*np.cos(js_data['physics'][center]['rotation'][frame]),
        #        xy[1]+RAD*np.sin(js_data['physics'][center]['rotation'][frame]))
        # marker = gz.circle(r=RAD/5, fill=(0,0,0, .5)).translate(xym)
        # marker.draw(surface)

    #Mouse cursor
    cursor_xy = np.array([this_data['mouse']['x'][frame]*RATIO, this_data['mouse']['y'][frame]*RATIO])
    cursor = gz.text('+', fontfamily="Helvetica",  fontsize=25, fill=(0,0,0), xy=cursor_xy) #, angle=Pi/12
    cursor.draw(surface)
    
    #Control
    if this_data['co'][frame]!=0:
        if this_data['co'][frame]==1:
            xy = np.array([this_data['o1']['x'][frame]*RATIO, this_data['o1']['y'][frame]*RATIO])
        elif this_data['co'][frame]==2:
            xy = np.array([this_data['o2']['x'][frame]*RATIO, this_data['o2']['y'][frame]*RATIO])
        elif this_data['co'][frame]==3:
            xy = np.array([this_data['o3']['x'][frame]*RATIO, this_data['o3']['y'][frame]*RATIO])
        elif this_data['co'][frame]==4:
            xy = np.array([this_data['o4']['x'][frame]*RATIO, this_data['o4']['y'][frame]*RATIO])

        #control_border = gz.arc(r=RAD, a1=0, a2=np.pi, fill=(0,0,0)).translate(xy)
        control_border = gz.circle(r=RAD,  stroke_width= 2).translate(xy)
        control_border.draw(surface)
    
    return surface.get_npimage()  

#Create the clip
duration = len(this_data['co'])/60
clip = mpy.VideoClip(make_frame, duration=duration)#, fps=60?\

#Create the filename (adding 0s to ensure things are in a nice alphabetical order now)
writename = 'example_control.mp4' #+ str(i) + '.mp4'
print writename
#Write the clip to file
clip.write_videofile(writename, fps=24)#



# y1a=js_data['physics']['o1']['y'][1:241]
# y1b=data['o1']['y']
# y2a=js_data['physics']['o2']['y'][1:241]
# y2b=data['o2']['y']
# y3a=js_data['physics']['o3']['y'][1:241]
# y3b=data['o3']['y']
# y4a=js_data['physics']['o4']['y'][1:241]
# y4b=data['o4']['y']


# Compare x axis trajectories
# t = np.arange(0., 240., 1)
# plt.plot(t, y1a, 'r--', t, y1b, 'r-')
# plt.plot(t, y2a, 'g--', t, y2b, 'g-')
# plt.plot(t, y3a, 'b--', t, y3b, 'b-')
# plt.plot(t, y4a, 'y--', t, y4b, 'y-')
# plt.show()

# from operator import add, neg
# tmp = map(add, data['o1']['x'], map(neg, js_data['physics']['o1']['x'][1:241]))
# plt.plot(tmp)
# plt.show()

