#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prior.py
"""

#IMPORT LIBRARIES
import numpy as np
from config import *
from utility import gaussian

#--- Set functions ---
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def weighted_sum(x):
    return x/np.sum(x)

def marginalize_prior(prior, axis):
    prior_marg = []
    if axis == 0:
        prior_marg = [np.sum([prior[j] for j in prior if tuple(i) == j[0]]) for i in mass_list]
  
    elif axis == 1:
        prior_marg = [np.sum([prior[j] for j in prior if tuple(np.array(i).flatten()) == j[1]]) for i in force_list]
    return prior_marg
            
def inner_expectation(true_dict, est_dict_ls, Sigma, prior_marg):
    if not isinstance(est_dict_ls, list):
        raise TypeError("Input est_dict_ls must be a list of dictionary object. Your input object is a {}"
                        .format(type(est_dict_ls)))
    if len(true_dict.values()) != 1:
        raise ValueError("For each timestamp, the true trajectory dictionary is fixed. Check the dimension first!")
    
    s_obj = true_dict.values()[0]
    gaussian_ls = []
    posterior = []
    for i, est_dict in enumerate(est_dict_ls):
        gaussian_obj = 1
        gaussian_sum = []
        for obj in est_dict:
            r = est_dict[obj]['r']
            theta =  est_dict[obj]['rotation']
            d = (r, theta)
            r_s = s_obj[obj]['r']
            theta_s = s_obj[obj]['rotation']
            s = (r_s, theta_s)
            error = gaussian(s, d, Sigma)
            gaussian_sum.append(error)
        gaussian_ls.append(np.mean(gaussian_sum))
        gaussian_obj *= np.prod(softmax(gaussian_sum))
        posterior.append(gaussian_obj*prior_marg[i])
    return np.average(gaussian_ls, weights=prior_marg), posterior

def outer_expectation(true_dict, est_dict, Sigma, prior, update, out_type):
    if out_type == 'mass':
        inner_e = []
        prior_mass = []
        for i in mass_list:
            inner_force = []
            prior_force = []
            for j in force_list:
                inner_force.append(est_dict[tuple(i),tuple(np.array(j).flatten())])
                prior_force.append(prior[tuple(i),tuple(np.array(j).flatten())])
            
            inner_expect = inner_expectation(true_dict, inner_force, Sigma, prior_force)
            inner_e.append(inner_expect[0])
            prior_mass.append(inner_expect[1])
        
        mass_expect = np.average(inner_e, weights=marginalize_prior(prior, 0))
        
        if update:
            prior_mass = weighted_sum(prior_mass)
            for index, mass in enumerate(mass_list):
                for index2, force in enumerate(force_list):
                    prior[tuple(mass), tuple(np.array(force).flatten())] = prior_mass[index][index2]
            return mass_expect, prior
                
        return mass_expect
    
    elif out_type == 'force':
        inner_e = []
        prior_force = []
        for i in force_list:
            inner_mass = []
            prior_mass = []
            for j in mass_list:
                inner_mass.append(est_dict[tuple(j), tuple(np.array(i).flatten())])
                prior_mass.append(prior[tuple(j), tuple(np.array(i).flatten())])
            
            inner_expect = inner_expectation(true_dict, inner_mass, Sigma, prior_mass)
            inner_e.append(inner_expect[0])
            prior_force.append(inner_expect[1])
        
        force_expect = np.average(inner_e, weights=marginalize_prior(prior, 1))
        
        if update:
            prior_force = softmax(prior_force)
            for index, force in enumerate(force_list):
                for index2, mass in enumerate(mass_list):
                    prior[tuple(mass), tuple(np.array(force).flatten())] = prior_force[index][index2]
            return force_expect, prior
            
        return force_expect
    
    else:
        raise ValueError("Cannot recognize input out_type value. Currently only support computing predictive \
                         divergence of 'mass' and 'force'")

def get_reward(true_ls_dict, est_ls_dict, Sigma, prior, mod=1):
    if mod == 1:
        rewards = []
        for i in range(len(est_ls_dict)):
            if i == len(est_ls_dict)-1:
                pd_mass, prior = outer_expectation(true_ls_dict[i], est_ls_dict[i], Sigma, prior, True, "mass")
                pd_force, prior = outer_expectation(true_ls_dict[i], est_ls_dict[i], Sigma, prior, True, "force")
            else:
                pd_mass = outer_expectation(true_ls_dict[i], est_ls_dict[i], Sigma, prior, False, "mass")
                pd_force = outer_expectation(true_ls_dict[i], est_ls_dict[i], Sigma, prior, False, "force")
            rewards.append((pd_mass+pd_force)/2)
        return 1.0-np.mean(rewards), prior
    
    elif mod == 2:
        reward_force = []
        reward_mass = []
        for i in range(len(est_ls_dict)):
            if i == len(est_ls_dict)-1:
                pd_mass, prior = outer_expectation(true_ls_dict[i], est_ls_dict[i], Sigma, prior, True, "mass")
                pd_force, prior = outer_expectation(true_ls_dict[i], est_ls_dict[i], Sigma, prior, True, "force")
            else:
                pd_mass = outer_expectation(true_ls_dict[i], est_ls_dict[i], Sigma, prior, False, "mass")
                pd_force = outer_expectation(true_ls_dict[i], est_ls_dict[i], Sigma, prior, False, "force")
            reward_force.append(pd_force)
            reward_mass.append(pd_mass)
        return 1.0-np.mean(reward_force), 1.0-np.mean(reward_mass), prior