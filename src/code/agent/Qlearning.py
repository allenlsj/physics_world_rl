#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qlearning.py
"""
import sys
sys.path.append('../simulator/')
import argparse
from environment import physic_env
from config import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.layers as L
import time

tf.reset_default_graph()
sess = tf.InteractiveSession()
keras.backend.set_session(sess)

def Network(state_dim, n_actions, h1, h2, h3):
    nn = keras.models.Sequential()
    nn.add(L.InputLayer((state_dim,)))

    nn.add(L.Dense(h1, activation='relu'))
    nn.add(L.Dense(h2, activation='relu'))
    nn.add(L.Dense(h3, activation='relu'))
    nn.add(L.Dense(n_actions, activation='linear'))

    return nn

# initialize the function approximator
nn = Network(state_dim, n_actions, nn_h1, nn_h2, nn_h3)

def get_action(s, epsilon):
    q_values = nn.predict(np.array(s)[None])[0]

    thre = np.random.rand()
    if thre < epsilon:
        action = np.random.choice(n_actions, 1)[0]
    else:
        action = np.argmax(q_values)

    return action

class q_agent:
    # placeholders for <s, a, r, s'>
    s_ph = keras.backend.placeholder(
        dtype='float32', shape=(None,) + (state_dim,))
    a_ph = keras.backend.placeholder(dtype='int32', shape=[None])
    r_ph = keras.backend.placeholder(dtype='float32', shape=[None])
    s_next_ph = keras.backend.placeholder(
        dtype='float32', shape=(None,) + (state_dim,))
    is_done_ph = keras.backend.placeholder(dtype='bool', shape=[None])

    # predicted q-value
    q_pred = nn(s_ph)
    q_pred_a = tf.reduce_sum(q_pred * tf.one_hot(a_ph, n_actions), axis=1)

    # sample estimate of target q-value
    q_pred_next = nn(s_next_ph)
    q_target_a = r_ph + qlearning_gamma * tf.reduce_max(q_pred_next, axis=1)
    q_target_a = tf.where(is_done_ph, r_ph, q_target_a)

    # mse loss
    loss = (q_pred_a - tf.stop_gradient(q_target_a)) ** 2
    loss = tf.reduce_mean(loss)

train_step = tf.train.AdamOptimizer(1e-4).minimize(q_agent.loss)

def train_iteration(t_max, epsilon, train=False):
    #print(time.strftime("%H:%M:%S", time.localtime()))
    total_reward = 0
    s = new_env.reset()

    for t in range(t_max):
        a = get_action(s, epsilon)
        s_next, r, is_done = new_env.step(a)

        if train:
            sess.run(train_step,{q_agent.s_ph: [s],q_agent.a_ph: [a], q_agent.r_ph: [r], q_agent.s_next_ph: [s_next], q_agent.is_done_ph: [is_done]})

        total_reward += r
        s = s_next
        if is_done:
            break

    return total_reward

def train_loop(args):
    global epsilon
    rewards = []
    for i in range(args.epochs):
        epoch_rewards = [train_iteration(t_max=1000, epsilon=epsilon, train=True) for t in range(args.sessions)]
        rewards += epoch_rewards
        print("epoch {}\t mean reward = {:.3f}\t epsilon = {:.3f}".format(i, np.mean(epoch_rewards), epsilon))
        epsilon *= epsilon_decay

        plt.plot(rewards)
        plt.ylabel("Reward")
        plt.xlabel("Number of iteration")
        plt.pause(0.001)
    plt.show()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training q-function approximator')
    parser.add_argument('--epochs', type=int, action='store', help='number of epoches to train', default=1000)
    parser.add_argument('--mode', type=int, action='store', help='type of intrinsic reward, 1 for mass, 2 for force', default=1)
    parser.add_argument('--sessions', type=int, action='store', help='number of sessions to train per epoch', default=10)

    args = parser.parse_args()
    print(args)

    # initialize the environment
    new_env = physic_env(cond, mass_list, force_list, init_mouse, T, args.mode, prior)
    # train
    train_loop(args)