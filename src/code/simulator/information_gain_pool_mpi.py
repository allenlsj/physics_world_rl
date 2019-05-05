#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
information_gain.py
"""

# IMPORT LIBRARIES
import numpy as np
from config import *
from utility import gaussian
from scipy.stats import entropy
import copy

from multiprocessing import Pool
from functools import partial
import time
#--- Set functions ---
# @autojit(parallel=True)
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# @autojit(parallel=True)
def weighted_sum(x):
    return x / np.sum(x)

# # @njit(parallel=True)
# @autojit(parallel=True)
# def np.sum(x):
#     return np.sum(x)
#
# @autojit(parallel=True)
# def np.mean(x):
#     return np.mean(x)

# @autojit(parallel=True)
def marginalize_prior(prior, axis):
    prior_marg = []
    # marginalize over mass
    if axis == 0:
        prior_marg = [np.sum([prior[j] for j in prior if tuple(i) == j[0]]) for i in mass_list]

    # marginalize over force
    elif axis == 1:
        prior_marg = [np.sum([prior[j] for j in prior if tuple(
        np.array(i).flatten()) == j[1]]) for i in force_list]
    return prior_marg

# @autojit(parallel=True)
def get_likelihood(true_dict, est_dict, Sigma):
    s_obj = true_dict.values()[0]

    error = []
    for obj in est_dict:
        r_list = est_dict[obj]['r']
        theta_list =  est_dict[obj]['rotation']
        d_list = [(r_list[i], theta_list[i]) for i in range(len(r_list))]
        r_s_list = s_obj[obj]['r']
        theta_s_list = s_obj[obj]['rotation']
        s_list = [(r_s_list[i], theta_s_list[i]) for i in range(len(r_s_list))]
        errors = [gaussian(s_list[i], d_list[i], Sigma) for i in range(len(s_list))]
        error.append(np.prod(errors)) # likelihood over T

    return np.mean(error) # average over four objects

# def marg_post_helper(mass_list_split,true_dict, est_dict, Sigma, prior, mode):
#     posterior = []
#     for i in mass_list_split:
#
#         ll = []
#         prior_mass = []
#         for j in force_list:
#             ll.append(get_likelihood(true_dict, est_dict[tuple(
#                 i), tuple(np.array(j).flatten())], Sigma))
#             prior_mass.append(prior[tuple(
#                 i), tuple(np.array(j).flatten())])
#         posterior.append([ll[k]*prior_mass[k] for k in range(len(force_list))])
#     return posterior

def marg_post_helper(mass_list_split):
    posterior = []
    for i in mass_list_split:

        ll = []
        prior_mass = []
        for j in force_list:
            ll.append(get_likelihood(glob_dict, glob_est[str(BACKHASH[(tuple(
                i), tuple(np.array(j).flatten()))])], glob_sigma))
            prior_mass.append(glob_prior[tuple(
                i), tuple(np.array(j).flatten())])
        posterior.append([ll[k]*prior_mass[k] for k in range(len(force_list))])
    return posterior

def marginalize_posterior(true_dict, est_dict, Sigma, prior, mode):


    global glob_dict
    global glob_est
    global glob_sigma
    global glob_prior
    global glob_mode

    glob_dict = true_dict
    glob_est = est_dict
    glob_sigma = Sigma
    glob_prior = prior
    glob_mode = mode

    if mode == 'mass':
        posterior = []

        n_proc=4
        n=0
        step_size=len(mass_list)/n_proc
        mass_list_splits=[]
        while n<len(mass_list):
            mass_list_splits.append(mass_list[n:n+step_size])
            n+=step_size

        p = Pool(n_proc)
        marg_mass_part=partial(marg_post_helper,true_dict=true_dict, est_dict=est_dict, Sigma=Sigma, prior=prior, mode=mode)

        tt=time.time()
        results = p.map(marg_post_helper, mass_list_splits)
        print('pure time',time.time()-tt)
        ll=[]
        prior_mass=[]
        for i in results:
            posterior+=i

        # for i in mass_list:
        #     ll = []
        #     prior_mass = []
        #     for j in force_list:
        #         ll.append(get_likelihood(true_dict, est_dict[tuple(
        #             i), tuple(np.array(j).flatten())], Sigma))
        #         prior_mass.append(prior[tuple(
        #             i), tuple(np.array(j).flatten())])
        #     posterior.append([ll[k]*prior_mass[k] for k in range(len(force_list))])

        posterior_marg = [np.sum(post) for post in posterior] # marginalize over force

        return weighted_sum(posterior_marg), posterior

    elif mode == 'force':
        posterior = []
        for i in force_list:
            ll = []
            prior_force = []
            for j in mass_list:
                key = (tuple(
                    j), tuple(np.array(i).flatten()))
                # print(key in BACKHASH.keys())
                # print(BACKHASH[(tuple(
                    # j), tuple(np.array(i).flatten()))])
                #print(est_dict.keys())
                ll.append(get_likelihood(true_dict, est_dict[str(BACKHASH[(tuple(
                    j), tuple(np.array(i).flatten()))])], Sigma))
                prior_force.append(prior[tuple(
                    j), tuple(np.array(i).flatten())])
            posterior.append([ll[k]*prior_force[k] for k in range(len(mass_list))])

        posterior_marg = [np.sum(post) for post in posterior] # marginalize over mass

        return weighted_sum(posterior_marg), posterior

# @autojit(parallel=True)
def get_reward_ig(true_dict, est_dict, Sigma, prior, mode=1, update_prior=True):
    if mode == 1:

        t0=time.time()
        posterior_mass,posterior = marginalize_posterior(true_dict, est_dict, Sigma, prior, 'mass')
        tx=time.time()
        print('ig post',tx-t0)
        posterior_ent_mass = entropy(posterior_mass)
        prior_ent_mass = entropy(marginalize_prior(prior, 0))

        t1=time.time()
        print('ig entropy',t1-t0)
        # update prior
        #prior_ = copy.deepcopy(prior)
        if update_prior:
            new_prior = weighted_sum(posterior)
            for index, mass in enumerate(mass_list):
                for index2, force in enumerate(force_list):
                    prior[tuple(mass), tuple(np.array(force).flatten())] = new_prior[index][index2]
        t2=time.time()
        print('ig update',t2-t1)
        print('ig total',t2-t0)
        return prior_ent_mass - posterior_ent_mass, prior

    elif mode == 2:
        posterior_force, posterior = marginalize_posterior(true_dict, est_dict, Sigma, prior, 'force')
        posterior_ent_force = entropy(posterior_force)
        prior_ent_force = entropy(marginalize_prior(prior, 1))

        # update prior
        #prior_ = copy.deepcopy(prior)
        if update_prior:
            new_prior = weighted_sum(posterior)
            for index, force in enumerate(force_list):
                for index2, mass in enumerate(mass_list):
                    prior[tuple(mass), tuple(np.array(force).flatten())] = new_prior[index][index2]

        return prior_ent_force - posterior_ent_force, prior

    else:
        raise ValueError("Cannot recognize input mode value. Currently only support 1 (for mass) and 2 (for force)")
