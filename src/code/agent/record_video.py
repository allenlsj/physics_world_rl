#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
record_video.py
"""
import sys
sys.path.append('../simulator/')
import argparse
from environment import physic_env
from config import *
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from functools import partial
import math
import gizeh as gz
import moviepy.editor as mpy
import json

num_feats = 16

def get_action(nn, s):
    q_values = nn.predict(np.array(s)[None])[0]
    action = np.argmax(q_values)

    # thre = np.random.rand()
    # if thre < 0.8:
    #     action = np.random.choice(n_actions, 1)[0]
    # else:
    #     action = np.argmax(q_values)

    return action

def test(args, env):
    if args.mode == 1:
        name = 'mass'
    else:
        name = 'force'

    with open(args.model_json, 'rb') as json_file:
        lmodel_json = json_file.read()
    model = model_from_json(lmodel_json)
    model.load_weights(args.model_weights)
    print("Model loaded!")
    
    rewards = []
    rewards_other = []
    s = env.reset()
    if args.RQN:
        s = np.transpose(np.array(s).reshape(num_feats/2,T,2),[0,2,1]).flatten().reshape(num_feats, T).T

    for t in range(1000):
        a = get_action(model, s)
        s_next, r, is_done, r_ = env.step(a)
        if args.RQN:
            s_next = np.transpose(np.array(s_next).reshape(num_feats/2,T,2),[0,2,1]).flatten().reshape(num_feats, T).T

        rewards += [r]
        rewards_other += [r_]
        s = s_next
        if is_done:
            break

    report = {}
    report['target reward'] = np.sum(rewards)
    report['mismatched reward'] = np.sum(rewards_other)
    with open('test_reward_{}_{}.json'.format(name, args.game), 'w') as fp:
        json.dump(report, fp)

    with open('test_data_{}_{}.json'.format(name, args.game), 'w') as fp:
        json.dump(env.step_data(), fp)

    # plt.plot(rewards)
    # plt.ylabel("Reward")
    # plt.xlabel("Number of actions")
    # plt.title("Total rewards of the {} exploration game via RQN is: {:.3f}".format(name, np.sum(rewards)))
    # fig = plt.gcf()
    # fig.savefig('eval_{}.png'.format(name))

    # plt.show()

    return env.step_data()

def make_frame(this_data, t):
    labels = ['A','B','C','D']
    centers = np.array(['o1','o2','o3','o4'])
    colors = [(0.97,0.46,0.44),(0.48,0.68,0),(0,0.75,0.75),(0.78,0.48,1)]
    RATIO = 100
    RAD = 25
    W = 600
    H = 400
    # H_outer = 500
    N_OBJ=4

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

        #Letters
        text = gz.text(label, fontfamily="Helvetica",  fontsize=25, fontweight='bold', fill=(0,0,0), xy=xy) #, angle=Pi/12
        text.draw(surface)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='recording test video')
    parser.add_argument('--game', type=int, action='store', help='games order', default=1)
    parser.add_argument('--mode', type=int, action='store', help='type of intrinsic reward, 1 for mass, 2 for force', default=1)
    parser.add_argument('--model_json', type=str, action='store', help='trained model structure to load', default='RQN_mass.json')
    parser.add_argument('--model_weights', type=str, action='store', help='trained model weights to load', default='RQN_mass.h5')
    parser.add_argument('--RQN', type=bool, action='store', help='True if the loaded model is RQN', default=True)

    args = parser.parse_args()
    print(args)
    if args.mode == 1:
        name = 'mass'
    else:
        name = 'force'

    # initialize the environment
    new_env = physic_env(test_cond, mass_list, force_list, init_mouse, T, args.mode, prior, reward_stop)

    data_ = test(args, new_env)

    frame = partial(make_frame, data_)
    duration = len(data_['co'])/60
    clip = mpy.VideoClip(frame, duration=duration)
    writename = 'test_{}_{}.mp4'.format(name, args.game)
    clip.write_videofile(writename, fps=24)