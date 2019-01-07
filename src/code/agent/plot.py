#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot.py
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot(args):
    # if args.mode == 1:
    #     name = 'mass'
    #     name_others = 'force'
    # else:
    #     name = 'force'
    #     name_others = 'name'

    # qlearning = np.loadtxt(args.Qlearning)
    # qlearning_reward = qlearning[0]
    # qlearning_loss = qlearning[1]
    # qlearning_cum_reward = qlearning[2]
    # print('Final loss of qlearning:{}'.format(qlearning_loss[-1]))

    # qagent = np.loadtxt(args.Qagent)
    # qagent_reward = qagent[0]
    # qagent_loss = qagent[1]
    # qagent_cum_reward = qagent[2]
    # print('Final loss of qagent:{}'.format(qagent_loss[-1]))

    RQN_mass = np.loadtxt('RQN_mass.txt')
    RQN_force = np.loadtxt('RQN_force.txt')
    RQN_mass_reward = RQN_mass[0]
    RQN_mass_reward_others = RQN_mass[1]
    RQN_mass_loss = RQN_mass[2]
    RQN_force_reward = RQN_force[0]
    RQN_force_reward_others = RQN_force[1]
    RQN_force_loss = RQN_force[2]

    plt.figure(1)
    plt.plot(RQN_mass_reward)
    plt.plot(RQN_mass_reward_others)
    plt.legend(['mass', 'force'], loc='best')
    plt.ylabel("RQN Training Rewards (mass exploration)")
    plt.xlabel("Number of games")
    plt.title(name)
    fig = plt.gcf()
    fig.savefig('RQN_mass_reward.png')

    plt.figure(2)
    plt.plot(RQN_force_reward)
    plt.plot(RQN_force_reward_others)
    plt.legend(['force', 'mass'], loc='best')
    plt.ylabel("RQN Training Rewards (force exploration)")
    plt.xlabel("Number of games")
    plt.title(name)
    fig = plt.gcf()
    fig.savefig('RQN_force_reward.png')

    plt.figure(3)
    plt.plot(RQN_mass_loss)
    plt.plot(RQN_force_loss)
    plt.legend(['mass', 'force'], loc='best')
    plt.ylabel("RQN Loss")
    plt.xlabel("Number of games")
    plt.title(name)
    fig = plt.gcf()
    fig.savefig('RQN_loss.png')

    plt.show()
    #print('Final loss of RQN:{}'.format(RQN_loss[-1]))

    # plt.figure(1)
    # plt.plot(qlearning_reward)
    # plt.plot(qagent_reward)
    # plt.plot(RQN_reward)
    # plt.legend(['MLP', 'MLP target network', 'RQN'], loc='best')
    # plt.ylabel("Rewards")
    # plt.xlabel("Number of iterations")
    # plt.title(name)
    # fig = plt.gcf()
    # fig.savefig('Final_reward_{}.png'.format(name))

    # plt.figure(2)
    # plt.plot(qlearning_loss)
    # plt.plot(qagent_loss)
    # plt.plot(RQN_loss)
    # plt.legend(['MLP', 'MLP target network', 'RQN'], loc='best')
    # plt.ylabel("Loss")
    # plt.xlabel("Number of iterations")
    # plt.title(name)
    # fig = plt.gcf()
    # fig.savefig('Final_loss_{}.png'.format(name))


    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot training curves')
    parser.add_argument('--mode', type=int, action='store',
                        help='type of intrinsic reward, 1 for mass, 2 for force', default=1)
    # parser.add_argument('--Qlearning', type=str, action='store', help='training details of Qlearning.py', default='Qlearning_mass.txt')
    # parser.add_argument('--Qagent', type=str, action='store', help='training details of Qagent.py', default='Qagent_mass.txt')
    # parser.add_argument('--RQN', type=str, action='store', help='training details of RQN.py', default='RQN_mass.txt')

    args = parser.parse_args()
    print(args)

    plot(args)