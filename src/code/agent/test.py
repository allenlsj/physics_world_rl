#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test.py
"""
import sys
sys.path.append('../simulator/')
import argparse
from environment import physic_env
from config import *
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import json

num_feats = 16


def get_action(nn, s):
    q_values = nn.predict(np.array(s)[None])[0]
    action = np.argmax(q_values)

    # thre = np.random.rand()
    # if thre < 0.75:
    #     action = np.random.choice(n_actions, 1)[0]
    # else:
    #     action = np.argmax(q_values)

    return action


def test_loop(args, env):
    if args.mode == 1:
        name = 'mass'
    else:
        name = 'force'

    with open(args.model_json, 'rb') as json_file:
        lmodel_json = json_file.read()
    model = model_from_json(lmodel_json)
    model.load_weights(args.model_weights)
    print("Model loaded!")

    cum_rewards = []
    for i in range(args.epochs):
        rewards = []
        s = env.reset()
        if args.RQN:
            s = np.transpose(np.array(s).reshape(
                num_feats / 2, T, 2), [0, 2, 1]).flatten().reshape(num_feats, T).T

        for t in range(1000):
            a = get_action(model, s)
            s_next, r, is_done, _ = env.step(a)
            if args.RQN:
                s_next = np.transpose(np.array(s_next).reshape(
                    num_feats / 2, T, 2), [0, 2, 1]).flatten().reshape(num_feats, T).T

            rewards += [r]
            s = s_next
            if is_done:
                break

        cum_rewards.append(np.cumsum(rewards).tolist())
        #np.savetxt('Test_data_{}.txt'.format(name), env.step_data())
        with open('data_test.json', 'w') as fp:
            json.dump(env.step_data(), fp)
        plt.plot(np.cumsum(rewards))
        plt.ylabel('Cumulative rewards')
        plt.xlabel("Number of steps")
        plt.title("RQN test({})".format(name))
        #plt.pause(0.001)
    fig = plt.gcf()
    fig.savefig('test_{}.png'.format(name))
    np.savetxt('Test_{}.txt'.format(name), cum_rewards)
    plt.show()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='recording test video')
    parser.add_argument('--epochs', type=int, action='store',
                        help='number of games to test', default=1)
    parser.add_argument('--mode', type=int, action='store',
                        help='type of intrinsic reward, 1 for mass, 2 for force', default=1)
    parser.add_argument('--model_json', type=str, action='store',
                        help='trained model structure to load', default='RQN_mass.json')
    parser.add_argument('--model_weights', type=str, action='store',
                        help='trained model weights to load', default='RQN_mass.h5')
    parser.add_argument('--RQN', type=bool, action='store',
                        help='True if the loaded model is RQN', default=True)

    args = parser.parse_args()
    print(args)

    # initialize the environment
    new_env = physic_env(test_cond, mass_list, force_list,
                         init_mouse, T, args.mode, prior, reward_stop)

    test_loop(args, new_env)
