# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:32:00 2020

@author: mapp2
"""


import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd

class target:
    def __init__(self, f, is_map = False):
        self.f_ = f
        if is_map:
            self.f = self.map_
        else:
            self.f = self.f_
    def map_(self, x):
        return list(map(self.f_ , x))
        
def ackley(x):
    a = 20
    b = 0.2
    c = 2*np.pi
    d = len(x)
    x = np.array(x)
    return -a*np.exp(-b*np.sqrt(sum(x**2)/d))-np.exp(sum(np.cos(c*x))/d)+a+np.exp(1)

def bukin(x):
    return 100*np.sqrt(np.abs(x[1]-0.01*x[0]**2)) + 0.01*np.abs(x[0]+10)

def cross_in_tray(x):
    x = np.array(x)
    sum_ = sum(x**2)
    prod_ = np.prod(np.sin(x))
    return -0.0001*(np.abs(prod_*np.exp(np.abs(100-np.sqrt(sum_)/np.pi)))+1)**0.1
    
def drop_wave(x):
    x = np.array(x)
    sum_ = sum(x**2)
    return (1+np.cos(12*np.sqrt(sum_)))/(0.5*sum_+2)

def eggholder(x):
    return -1*(x[1]+47)*np.sin(np.sqrt(np.abs(x[1]+x[0]/2+47))) - x[0]*np.sin(np.sqrt(np.abs(x[0]-x[1]-47)))

def griewank(x):
    x = np.array(x)
    sum_ = sum(x**2)/4000
    prod_ = np.prod(np.cos(np.array([x[i]/np.sqrt(i+1) for i in range(len(x))])))
    return sum_ - prod_ + 1

def holder(x):
    x = np.array(x)
    sum_ = sum(x**2)
    return -1*np.abs(np.sin(x[0])*np.cos(x[1])*np.exp(np.abs(1-np.sqrt(sum_)/np.pi)))

def langermann(x):
    x = np.array(x)
    m = 5
    c = np.array([1, 2, 5, 2, 3])
    xA2 = ((x - np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]]))**2).sum(axis=1)
    return sum(c*np.exp(-1/np.pi*xA2)*np.cos(np.pi*xA2))

def levy(x):
    w = np.array([1+(i-1)/4 for i in x])
    x = np.array(x)
    sum_ = (w-1)**2*(1+10*np.sin(np.pi*w+1)**2)
    sum_ = sum(sum_[:-1])
    return np.sin(np.pi*w[0])**2 + sum_ + (w[-1]-1)**2*(1+np.sin(2*np.pi*w[-1])**2)

def levy13(x):
    return np.sin(3*np.pi*x[0])**2 + (x[0]-1)**2*(1*np.sin(3*np.pi*x[1])**2) + (x[1]-1)**2*(1+np.sin(2*np.pi*x[1])**2)

def rastrigin(x):
    x = np.array(x)
    d = len(x)
    sum_ = sum(x**2-10*np.cos(2*np.pi*x))
    return 10*d + sum_

def schaffer2(x):
    x = np.array(x)
    sum_ = sum(x**2)
    return 0.5 + (np.sin(sum_)**2 - 0.5)/((1+0.001*sum_)**2)
    
def schaffer4(x):
    x = np.array(x)
    sum_0 = sum(np.array( [x[i]**2*(-1)**i for i in range(len(x))]))
    sum_1 =sum(x**2)
    return 0.5 + (np.cos(np.sin(np.abs(sum_0)))-0.5)/(1+0.001*sum_1)**2

def schwefel(x):
    x  = np.array(x)
    sum_ = sum(x*np.sin(np.sqrt(np.abs(x))))
    d = len(x)
    return 418.9829*d - sum_

def shubert(x):
    sum_ = np.array([[i*np.cos( (i+1)*x[j] + i) for i in range(5)] for j in range(len(x))]).sum(axis=1)
    return np.prod(sum_)

if __name__ == '__main__':

    # print(ackley([0,0,0]))
    # print(bukin([-10,1]))
    # print(cross_in_tray([1.3491,-1.3491]))
    # print(drop_wave([0,0]))
    # print(eggholder([512, 404.2319]))
    # print(griewank([0,0,0,0]))
    # print(holder([8.05502,9.66459]))
    # print(langermann([8.05502,9.66459]))
    # print(levy([1,1]))
    # print(levy13([1,1]))
    # print(rastrigin([0,0]))
    # print(schaffer2([0,0]))
    # print(schaffer4([0,0]))
    # print(schwefel([420.9687, 420.9687]))
    # print(shubert([0,0]))
    
    f_ackley = target(ackley, True)
    
    print(f_ackley.f([[0,0], [1,1]]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    