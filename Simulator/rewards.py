import numpy as np

def gaussian(s, d, Sigma, eta=10):
    diff = np.subtract(s,d)
    return np.exp(-eta/2 * np.dot(np.dot(diff.T, Sigma), diff))

def inner_expectation(true_dict, est_dict_ls, Sigma):
    if not isinstance(est_dict_ls, list):
        raise TypeError("Input est_dict_ls must be a list of dictionary object. Your input object is a {}"
                        .format(type(est_dict_ls)))
    if len(true_dict.values()) != 1:
        raise ValueError("For each timestamp, the true trajectory dictionary is fixed. Check the dimension first!")
    
    s_obj = true_dict.values()[0]
    gaussian_ls = []
    for est_dict in est_dict_ls:
        for obj in est_obj:
            r = est_dict[obj]['r']
            theta =  est_dict[obj]['rotation']
            d = (r, theta)
            r_s = s_obj[obj]['r']
            theta_s = s_obj[obj]['rotation']
            s = (r_s, theta_s)
            gaussian_ls.append(gaussian(s, d, Sigma))         
    return np.mean(gaussian_ls)

def outer_expectation(true_dict, est_dict, Sigma, out_type):
    if out_type == 'mass':
        inner_e = []
        dict_ls = [i[0] for i in est_dict.keys()]
        for i in set(dict_ls):
            inner_force = []
            for j in est_dict:
                if i == j[0]:
                    inner_force.append(est_dict[j])
            inner_e.append(inner_expectation(true_dict, inner_force, Sigma))
        return np.mean(inner_e)
    
    elif out_type == 'force':
        inner_e = []
        dict_ls = [i[1] for i in est_dict.keys()]
        for i in set(dict_ls):
            inner_mass = []
            for j in est_dict:
                if i == j[1]:
                    inner_mass.append(est_dict[j])
            inner_e.append(inner_expectation(true_dict, inner_mass, Sigma))        
        return np.mean(inner_e)
    
    else:
        raise ValueError("Cannot recognize input out_type value. Currently only support computing predictive \
                         divergence of 'mass' and 'force'")
        
def get_reward(true_ls_dict, est_ls_dict, Sigma):
    rewards = []
    for i in range(len(est_ls_dict)):
        pd_mass = outer_expectation(true_ls_dict[i], est_ls_dict[i], Sigma, "mass")
        pd_force = outer_expectation(true_ls_dict[i], est_ls_dict[i], Sigma, "force")
        rewards.append((pd_mass+pd_force)/2)
    return 1-np.mean(rewards)