#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot.py
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot(args):
    if args.mode == 1:
        name = 'mass'
    else:
        name = 'force'

    qlearning = np.loadtxt(args.Qlearning)
    qlearning_reward = qlearning[0]
    qlearning_loss = qlearning[1]
    qlearning_cum_reward = qlearning[2]

    qagent = np.loadtxt(args.Qagent)
    qagent_reward = qagent[0]
    qagent_loss = qagent[1]
    qagent_cum_reward = qagent[2]

    RQN = np.loadtxt(args.RQN)
    RQN_reward = RQN[0]
    RQN_loss = RQN[1]
    RQN_cum_reward = RQN[2]

    plt.figure(1)
    plt.plot(qlearning_reward)
    plt.plot(qagent_reward)
    plt.plot(RQN_reward)
    plt.legend(['MLP', 'MLP target network', 'RQN'], loc='best')
    plt.ylabel("Rewards")
    plt.xlabel("Number of iterations")
    plt.title(name)

    plt.figure(2)
    plt.plot(qlearning_loss)
    plt.plot(qagent_loss)
    plt.plot(RQN_loss)
    plt.legend(['MLP', 'MLP target network', 'RQN'], loc='best')
    plt.ylabel("Loss")
    plt.xlabel("Number of iterations")
    plt.title(name)

    plt.figure(3)
    plt.plot(qlearning_cum_reward)
    plt.plot(qagent_cum_reward)
    plt.plot(RQN_cum_reward)
    plt.legend(['MLP', 'MLP target network', 'RQN'], loc='best')
    plt.ylabel("Cumulative Rewards")
    plt.xlabel("Number of iterations")
    plt.title(name)

    plt.show()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot training curves')
    parser.add_argument('--mode', type=int, action='store',
                        help='type of intrinsic reward, 1 for mass, 2 for force', default=1)
    parser.add_argument('--Qlearning', type=str, action='store', help='training details of Qlearning.py', default='Qlearning_mass.txt')
    parser.add_argument('--Qagent', type=str, action='store', help='training details of Qagent.py', default='Qagent_mass.txt')
    parser.add_argument('--RQN', type=str, action='store', help='training details of RQN.py', default='RQN_mass.txt')

    args = parser.parse_args()
    print(args)

    plot(args)