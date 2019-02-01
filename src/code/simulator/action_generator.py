#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
action_generator.py
"""

from math import sqrt, pow, sin, cos
from math import pi as M_PI
import numpy as np
import matplotlib.pyplot as plt
import random
M_PI_2 = M_PI * 2
num_obj = 4 + 1
'''
Adapted from: 
https://gist.github.com/zeffii/c1e14dd6620ad855d81ec2e89a859719
'''

def generate_action(m_x, m_y, index, beta=1.0, T=15):
    #  Modeled after no click action
    def NoClick(p):
        return None

    #  Modeled after the line y=0
    def rightward(p):
        return 0

    #..Modeled after the line x=0
    def upward():
        return np.arange(0, beta, beta/T)

    #  Modeled after the line y = x
    def LinearInterpolation(p):
        return p

    # Modeled after the parabola y = x^2
    def QuadraticEaseIn(p):
        return p * p

    # Modeled after the parabola y = -x^2 + 2x
    def QuadraticEaseOut(p):
        return -(p * (p - 2))

    # Modeled after the piecewise quadratic
    # y = (1/2)((2x)^2)             ; [0, 0.5)
    # y = -(1/2)((2x-1)*(2x-3) - 1) ; [0.5, 1]
    def QuadraticEaseInOut(p):
        if (p < 0.5):
            return 2 * p * p
        return (-2 * p * p) + (4 * p) - 1

    # Modeled after the cubic y = x^3
    def CubicEaseIn(p):
        return p * p * p

    # Modeled after the cubic y = (x - 1)^3 + 1
    def CubicEaseOut(p):
        f = (p - 1)
        return f * f * f + 1

    # Modeled after the piecewise cubic
    # y = (1/2)((2x)^3)       ; [0, 0.5)
    # y = (1/2)((2x-2)^3 + 2) ; [0.5, 1]
    def CubicEaseInOut(p):
        if (p < 0.5):
            return 4 * p * p * p
        else:
            f = ((2 * p) - 2)
            return 0.5 * f * f * f + 1

    # Modeled after the quartic x^4
    def QuarticEaseIn(p):
        return p * p * p * p

    # Modeled after the quartic y = 1 - (x - 1)^4
    def QuarticEaseOut(p):
        f = (p - 1)
        return f * f * f * (1 - p) + 1

    # Modeled after the piecewise quartic
    # y = (1/2)((2x)^4)        ; [0, 0.5)
    # y = -(1/2)((2x-2)^4 - 2) ; [0.5, 1]
    def QuarticEaseInOut(p) :
        if (p < 0.5):
            return 8 * p * p * p * p
        else:
            f = (p - 1)
            return -8 * f * f * f * f + 1
        


    # Modeled after the quintic y = x^5
    def QuinticEaseIn(p):
        return p * p * p * p * p

    # Modeled after the quintic y = (x - 1)^5 + 1
    def QuinticEaseOut(p):
        f = (p - 1)
        return f * f * f * f * f + 1


    # Modeled after the piecewise quintic
    # y = (1/2)((2x)^5)       ; [0, 0.5)
    # y = (1/2)((2x-2)^5 + 2) ; [0.5, 1]
    def QuinticEaseInOut(p):
        if (p < 0.5):
            return 16 * p * p * p * p * p
        else:
            f = ((2 * p) - 2)
            return  0.5 * f * f * f * f * f + 1

    # Modeled after quarter-cycle of sine wave
    def SineEaseIn(p):
        return sin((p - 1) * M_PI_2) + 1

    # Modeled after quarter-cycle of sine wave (different phase)
    def SineEaseOut(p):
        return sin(p * M_PI_2)

    # Modeled after half sine wave
    def SineEaseInOut(p):
        return 0.5 * (1 - cos(p * M_PI))

    # Modeled after shifted quadrant IV of unit circle
    def CircularEaseIn(p):
        return 1 - sqrt(1 - (p * p))

    # Modeled after shifted quadrant II of unit circle
    def CircularEaseOut(p):
        return sqrt((2 - p) * p)

    # Modeled after the piecewise circular function
    # y = (1/2)(1 - sqrt(1 - 4x^2))           ; [0, 0.5)
    # y = (1/2)(sqrt(-(2x - 3)*(2x - 1)) + 1) ; [0.5, 1]
    def CircularEaseInOut(p):
        if(p < 0.5):
            return 0.5 * (1 - sqrt(1 - 4 * (p * p)))
        else:
            return 0.5 * (sqrt(-((2 * p) - 3) * ((2 * p) - 1)) + 1)

    # Modeled after the exponential function y = 2^(10(x - 1))
    def ExponentialEaseIn(p):
        return p if (p == 0.0) else pow(2, 10 * (p - 1))

    # Modeled after the exponential function y = -2^(-10x) + 1
    def ExponentialEaseOut(p):
        return p if (p == 1.0) else 1 - pow(2, -10 * p)

    # Modeled after the piecewise exponential
    # y = (1/2)2^(10(2x - 1))         ; [0,0.5)
    # y = -(1/2)*2^(-10(2x - 1))) + 1 ; [0.5,1]
    def ExponentialEaseInOut(p):
        if(p == 0.0 or p == 1.0):
            return p
        
        if(p < 0.5):
            return 0.5 * pow(2, (20 * p) - 10)
        else:
            return -0.5 * pow(2, (-20 * p) + 10) + 1

    # Modeled after the damped sine wave y = sin(13pi/2*x)*pow(2, 10 * (x - 1))
    def ElasticEaseIn(p):
        return sin(13 * M_PI_2 * p) * pow(2, 10 * (p - 1))

    # Modeled after the damped sine wave y = sin(-13pi/2*(x + 1))*pow(2, -10x) + 1
    def ElasticEaseOut(p):
        return sin(-13 * M_PI_2 * (p + 1)) * pow(2, -10 * p) + 1

    # Modeled after the piecewise exponentially-damped sine wave:
    # y = (1/2)*sin(13pi/2*(2*x))*pow(2, 10 * ((2*x) - 1))      ; [0,0.5)
    # y = (1/2)*(sin(-13pi/2*((2x-1)+1))*pow(2,-10(2*x-1)) + 2) ; [0.5, 1]
    def ElasticEaseInOut(p):
        if (p < 0.5):
            return 0.5 * sin(13 * M_PI_2 * (2 * p)) * pow(2, 10 * ((2 * p) - 1))
        else:
            return 0.5 * (sin(-13 * M_PI_2 * ((2 * p - 1) + 1)) * pow(2, -10 * (2 * p - 1)) + 2)

    # Modeled after the overshooting cubic y = x^3-x*sin(x*pi)
    def BackEaseIn(p):
        return p * p * p - p * sin(p * M_PI)

    # Modeled after overshooting cubic y = 1-((1-x)^3-(1-x)*sin((1-x)*pi))
    def BackEaseOut(p):
        f = (1 - p)
        return 1 - (f * f * f - f * sin(f * M_PI))

    # Modeled after the piecewise overshooting cubic function:
    # y = (1/2)*((2x)^3-(2x)*sin(2*x*pi))           ; [0, 0.5)
    # y = (1/2)*(1-((1-x)^3-(1-x)*sin((1-x)*pi))+1) ; [0.5, 1]
    def BackEaseInOut(p):
        if (p < 0.5):
            f = 2 * p
            return 0.5 * (f * f * f - f * sin(f * M_PI))
        else:
            f = (1 - (2*p - 1))
            return 0.5 * (1 - (f * f * f - f * sin(f * M_PI))) + 0.5

    def BounceEaseIn(p):
        return 1 - BounceEaseOut(1 - p)

    def BounceEaseOut(p):
        if(p < 4/11.0):
            return (121 * p * p)/16.0
        
        elif(p < 8/11.0):
            return (363/40.0 * p * p) - (99/10.0 * p) + 17/5.0
        
        elif(p < 9/10.0):
            return (4356/361.0 * p * p) - (35442/1805.0 * p) + 16061/1805.0
        
        else:
            return (54/5.0 * p * p) - (513/25.0 * p) + 268/25.0

    def BounceEaseInOut(p):
        if(p < 0.5):
            return 0.5 * BounceEaseIn(p*2)
        else:
            return 0.5 * BounceEaseOut(p * 2 - 1) + 0.5

    def find_direction_index(index, index_action_fun):
        return int((index-index_action_fun*num_obj)//num_obj)

    fns = [NoClick, upward, rightward, LinearInterpolation, QuadraticEaseIn, QuadraticEaseOut, QuadraticEaseInOut, CubicEaseIn, CubicEaseOut, CubicEaseInOut, QuarticEaseIn, QuarticEaseOut, QuarticEaseInOut, QuinticEaseIn, QuinticEaseOut, QuinticEaseInOut, SineEaseIn, SineEaseOut,SineEaseInOut, CircularEaseIn, CircularEaseOut, CircularEaseInOut, ExponentialEaseIn, ExponentialEaseOut, ExponentialEaseInOut, ElasticEaseIn, ElasticEaseOut, ElasticEaseInOut, BackEaseIn, BackEaseOut, BackEaseInOut, BounceEaseIn, BounceEaseOut, BounceEaseInOut]
    index_list = np.arange(0, num_obj*((len(fns)-3)*4+2*2+1))
    obj_list = np.arange(0, num_obj)

    # find corresponding index of fns
    index_action = index // num_obj
    obj = index % num_obj
    if index_action == 0:
        index_action_fun = 0
    elif index_action in [1,2,3,4]:
        index_action_fun = (index_action-1)//2+1
    else:
        index_action_fun = (index_action-5)//4+3

    # generate new action paths based on the input index
    if index_action_fun == 0:
        fn = NoClick
        new_m_x = [m_x] * T
        new_m_y = [m_y] * T
    elif index_action_fun == 1:
        fn = upward
        new_m_x = [m_x] * T
        dir_y = find_direction_index(index, index_action_fun)
        if dir_y == 0:
            y = fn()
        else:
            y = -fn()
        y = y.tolist()
        new_m_y = [m_y + j for j in y]
    else:
        fn = fns[index_action_fun]
        x = np.arange(0, beta, beta/T)
        y = list(map(fn, x))
        if index_action_fun == 2:
            dir_x = find_direction_index(index, index_action_fun + 1)
            if dir_x == 1:
                x = -x
        else:
            dir_ = find_direction_index(index, index_action_fun*4-7)
            if dir_ == 2:
                y = [-i for i in y]
            elif dir_ == 3:
                x = -x
            elif dir_ == 1:
                x = -x
                y = [-i for i in y]
        x = x.tolist()
        new_m_x = [m_x + i for i in x]
        new_m_y = [m_y + j for j in y]

    # reset the mouse position if necessary
    new_m_x = [5.75 if i>5.75 else 0.25 if i<0.25 else i for i in new_m_x]
    new_m_y = [3.75 if i>3.75 else 0.25 if i<0.25 else i for i in new_m_y]

    # randomly generate new paths
    # fn = random.choice(fns)
    # if fn.__name__ == 'NoClick':
    #     new_m_x = [m_x] * T
    #     new_m_y = [m_y] * T
    # elif fn.__name__ == 'upward':
    #     dir_y = np.random.randint(2)
    #     new_m_x = [m_x] * T
    #     if dir_y == 0:
    #         y = fn()
    #     else:
    #         y = -fn()
    #     y = y.tolist()
    #     new_m_y = [m_y + j for j in y]
    # else:
    #     dir_x = np.random.randint(2)
    #     dir_y = np.random.randint(2)
    #     x = np.arange(0, beta, beta/T)
    #     y = list(map(fn, x))
    #     if dir_x == 0:
    #         if dir_y == 1:
    #             y = [-i for i in y]
    #     else:
    #         x = -x
    #         if dir_y == 1:
    #             y = [-i for i in y]
    #     x = x.tolist()
    #     new_m_x = [m_x + i for i in x]
    #     new_m_y = [m_y + j for j in y]
    return obj, new_m_x, new_m_y

if __name__ == "__main__":
    obj,x,y = generate_action(3,2,577)
    plt.plot(x,y)
    plt.show()
