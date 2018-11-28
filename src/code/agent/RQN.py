#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RQN.py
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
num_feats = 8


class q_agent:
    def __init__(self, name, num_feats, n_actions, epsilon=0):
        with tf.variable_scope(name):
            inputs = L.Input(shape=(T, num_feats))
            lstm, state_h, state_c = L.LSTM(num_feats, return_sequences=True, return_state=True)(inputs)
            output = L.Dense(n_actions, activation='linear')(state_h)
            self.nn = keras.models.Model(inputs=inputs, outputs=output)

            self.state_t = tf.placeholder(
                'float32', [None, ] + list((T, num_feats,)))
            self.q_values_t = self.get_q_values_tensors(self.state_t)

        self.weights = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.epsilon = epsilon

    def get_q_values_tensors(self, state_t):
        return self.nn(state_t)

    def get_q_values(self, state_t):
        sess = tf.get_default_session()
        return sess.run(self.q_values_t, {self.state_t: state_t})

    def get_action(self, s):
        epsilon = self.epsilon
        q_values = self.nn.predict(np.array(s)[None])[0]

        thre = np.random.rand()
        if thre < epsilon:
            action = np.random.choice(n_actions, 1)[0]
        else:
            action = np.argmax(q_values)

        return action


# initialize q agent and target network
agent = q_agent("agent", num_feats, n_actions, epsilon)
target_network = q_agent("target_network", num_feats,
                         n_actions, epsilon)


def load_weigths_into_target_network(agent, target_network):
    assigns = []
    for w_agent, w_target in zip(agent.weights, target_network.weights):
        assigns.append(tf.assign(w_target, w_agent, validate_shape=True))
    tf.get_default_session().run(assigns)


class train_agent:
    # placeholders for <s, a, r, s'>
    s_ph = keras.backend.placeholder(
        dtype='float32', shape=(None,) + (T, num_feats,))
    a_ph = keras.backend.placeholder(dtype='int32', shape=[None])
    r_ph = keras.backend.placeholder(dtype='float32', shape=[None])
    s_next_ph = keras.backend.placeholder(
        dtype='float32', shape=(None,) + (T, num_feats,))
    is_done_ph = keras.backend.placeholder(dtype='bool', shape=[None])

    # predicted q-value
    q_pred = agent.get_q_values_tensors(s_ph)
    q_pred_a = tf.reduce_sum(q_pred * tf.one_hot(a_ph, n_actions), axis=1)

    # sample estimate of target q-value
    q_pred_next = target_network.get_q_values_tensors(s_next_ph)
    q_target_a = r_ph + qlearning_gamma * tf.reduce_max(q_pred_next, axis=1)
    q_target_a = tf.where(is_done_ph, r_ph, q_target_a)

    # mse loss
    loss = (q_pred_a - q_target_a) ** 2
    loss = tf.reduce_mean(loss)


train_step = tf.train.AdamOptimizer(
    1e-4).minimize(train_agent.loss, var_list=agent.weights)

sess.run(tf.global_variables_initializer())


def train_iteration(t_max, train=False):
    #print(time.strftime("%H:%M:%S", time.localtime()))
    total_reward = 0
    td_loss = 0
    s = new_env.reset()
    s = np.transpose(np.array(s).reshape(num_feats/2,T,2),[0,2,1]).flatten().reshape(num_feats, T).T

    for t in range(t_max):
        a = agent.get_action(s)
        s_next, r, is_done = new_env.step(a)
        s_next = np.transpose(np.array(s_next).reshape(num_feats/2,T,2),[0,2,1]).flatten().reshape(num_feats, T).T

        if train:
            _, loss_t = sess.run([train_step, train_agent.loss], {train_agent.s_ph: [s], train_agent.a_ph: [a], train_agent.r_ph: [
                     r], train_agent.s_next_ph: [s_next], train_agent.is_done_ph: [is_done]})

        total_reward += r
        td_loss += loss_t
        s = s_next
        if is_done:
            break

    return [total_reward, td_loss]


def train_loop(args):
    rewards = []
    loss = []
    if args.mode == 1:
        name = 'mass'
    else:
        name = 'force'

    for i in range(args.epochs):
        #print(time.strftime("%H:%M:%S", time.localtime()))
        results = [train_iteration(
            t_max=1000, train=True) for t in range(args.sessions)]
        epoch_rewards = [r[0] for r in results]
        epoch_loss = [l[1] for l in results]
        rewards += epoch_rewards
        loss += epoch_loss
        print("epoch {}\t mean reward = {:.4f}\t mean loss = {:.4f}\t epsilon = {:.4f}".format(
            i, np.mean(epoch_rewards), np.mean(epoch_loss), agent.epsilon))
        # adjust agent parameters
        if i % 2 == 0:
            load_weigths_into_target_network(agent, target_network)
            agent.epsilon = max(agent.epsilon * epsilon_decay, 0.01)

        plt.figure(1)
        plt.plot(rewards)
        plt.ylabel("Reward")
        plt.xlabel("Number of iteration")
        plt.title("Recurrent Q learning with target network (" + name + ")")
        plt.pause(0.001)
        fig = plt.gcf()
        fig.savefig('RQN_{}_reward.png'.format(name))

        plt.figure(2)
        plt.plot(loss)
        plt.ylabel("Loss")
        plt.xlabel("Number of iteration")
        plt.title("Recurrent Q learning with target network (" + name + ")")
        plt.pause(0.001)
        fig = plt.gcf()
        fig.savefig('RQN_{}_loss.png'.format(name))

    plt.show()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='training a recurrent q-network')
    parser.add_argument('--epochs', type=int, action='store',
                        help='number of epoches to train', default=1000)
    parser.add_argument('--mode', type=int, action='store',
                        help='type of intrinsic reward, 1 for mass, 2 for force', default=1)
    parser.add_argument('--sessions', type=int, action='store',
                        help='number of sessions to train per epoch', default=10)

    args = parser.parse_args()
    print(args)

    # initialize the environment
    new_env = physic_env(cond, mass_list, force_list,
                         init_mouse, T, args.mode, prior)
    # train
    train_loop(args)