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

#--- Set functions ---
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def weighted_sum(x):
    return x / np.sum(x)


def marginalize_prior(prior, axis):
    prior_marg = []
    if axis == 0:
        prior_marg = [
            np.sum([prior[j] for j in prior if tuple(i) == j[0]]) for i in mass_list]

    elif axis == 1:
        prior_marg = [np.sum([prior[j] for j in prior if tuple(
            np.array(i).flatten()) == j[1]]) for i in force_list]
    return prior_marg


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


def marginalize_posterior(true_dict, est_dict, Sigma, prior, mode):
    if mode == 'mass':
        posterior = []
        for i in mass_list:
            ll = []
            prior_mass = []
            for j in force_list:
                ll.append(get_likelihood(true_dict, est_dict[tuple(
                    i), tuple(np.array(j).flatten())], Sigma))
                prior_mass.append(prior[tuple(
                    i), tuple(np.array(j).flatten())])
            posterior.append([ll[k]*prior_mass[k] for k in range(len(force_list))])
        
        posterior_marg = [np.sum(post) for post in posterior] # marginalize over force

        new_prior = weighted_sum(posterior)
        for index, mass in enumerate(mass_list):
            for index2, force in enumerate(force_list):
                prior[tuple(mass), tuple(np.array(force).flatten())] = new_prior[index][index2]

        return weighted_sum(posterior_marg), prior

    elif mode == 'force':
        posterior = []
        for i in force_list:
            ll = []
            prior_force = []
            for j in mass_list:
                ll.append(get_likelihood(true_dict, est_dict[tuple(
                    j), tuple(np.array(i).flatten())], Sigma))
                prior_mass.append(prior[tuple(
                    j), tuple(np.array(i).flatten())])
            posterior.append([ll[k]*prior_force[k] for k in range(len(mass_list))])
        
        posterior_marg = [np.sum(post) for post in posterior] # marginalize over mass

        new_prior = weighted_sum(posterior)
        for index, force in enumerate(force_list):
            for index2, mass in enumerate(mass_list):
                prior[tuple(mass), tuple(np.array(force).flatten())] = new_prior[index][index2]

        return weighted_sum(posterior_marg), prior


def get_reward_ig(true_dict, est_dict, Sigma, prior, mode=1):
    if mode == 1:
        posterior_mass, prior = marginalize_posterior(true_dict, est_dict, Sigma, prior, 'mass')
        posterior_ent_mass = entropy(posterior_mass)
        prior_ent_mass = entropy(marginalize_prior(prior, 0))
        return prior_ent_mass - posterior_ent_mass, prior
    elif mode == 2:
        posterior_force, prior = marginalize_posterior(true_dict, est_dict, Sigma, prior, 'force')
        posterior_ent_force = entropy(posterior_force)
        prior_ent_force = entropy(marginalize_prior(prior, 1))
        
        return prior_ent_force - posterior_ent_force, prior