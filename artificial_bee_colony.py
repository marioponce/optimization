# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:49:27 2020

@author: mapp2
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
import untitled1 as ff

class DE:
    def __init__(self, pars):
        try:
            self.NP = pars["NP"]
        except:
            self.NP = 10
        try:
            self.NG = pars["NG"]
        except:
            self.NG = 10
        try: 
            self.F = pars["F"]
        except:
            self.F = 0.5
        try: 
            self.GR = pars["GR"]
        except:
            self.GR = 0.5
        try: 
            self.f = pars["func"]
        except:
            self.f = ff.target(ff.ackley, True).f
        try:
            self.lob = np.array(pars["lob"])
        except:
            self.lob = np.array([-1,-1])
        try:
            self.upb = np.array(pars["upb"])
        except:
            self.upb = np.array([1,1])
        try:
            self.tol = pars["tol"]
        except:
            self.tol = 1e-6
        if len(self.upb) < len(self.lob): 
            self.upb = self.upb[:len(self.lob)]
        if len(self.upb) > len(self.lob): 
            self.lob = self.lob[:len(self.upb)]
        self.n = len(self.upb)
        self.x = None
        self.m = None
    def init(self):
        self.x = np.array([self.lob + rd.random(self.n)*(self.upb - self.lob) for i in range(self.NP)])
        self.fx = np.array(self.f(self.x))
    def mut(self):
        a,b,c = rd.randint(self.NP, size= [3, self.NP])
        self.m = self.x[a] + self.F*(self.x[b] - self.x[c])
        for i in range(self.NP):
            self.m[i][self.m[i]>self.upb] = self.upb[self.m[i]>self.upb]
            self.m[i][self.m[i]<self.lob] = self.lob[self.m[i]<self.lob]
    def cross(self):
        gr = rd.random(self.x.shape) < self.GR
        self.t = self.x[:]
        self.t[gr] = self.m[gr][:]
        self.m = None
        del gr
        self.ft = np.array(self.f(self.t))
    def sel(self):
        sel = self.ft < self.fx
        self.x[sel] = self.t[sel][:]
        self.fx[sel] = self.ft[sel][:]
        self.t = None
        self.ft = None
    def run_(self):
        self.init()
        g = 0
        self.min_x = [np.min(self.fx)]
        while g < self.NG or self.min_x < self.tol: 
            self.mut()
            self.cross()
            self.sel()
            self.min_x.append(np.min(self.fx))
            g += 1
            
class ABC:
    def __init__(self, pars):
        try:
            self.SN = pars["SN"]
        except:
            self.SN = 10
        try:
            self.MCN = pars["MCN"]
        except:
            self.MCN = 10
        try: 
            self.limit = pars["limit"]
        except:
            self.limite = 5
        try: 
            self.f = pars["func"]
        except:
            self.f = ff.target(ff.ackley, True).f
        try:
            self.lob = np.array(pars["lob"])
        except:
            self.lob = np.array([-1,-1])
        try:
            self.upb = np.array(pars["upb"])
        except:
            self.upb = np.array([1,1])
        try:
            self.tol = pars["tol"]
        except:
            self.tol = 1e-6
        if len(self.upb) < len(self.lob): 
            self.upb = self.upb[:len(self.lob)]
        if len(self.upb) > len(self.lob): 
            self.lob = self.lob[:len(self.upb)]
        self.n = len(self.upb)
        self.x = None
        self.m = None
    def init(self):
        self.x = np.array([self.lob + rd.random(self.n)*(self.upb - self.lob) for i in range(self.SN)])
        self.fx = np.array(self.f(self.x))
        self.i = np.zeros(self.fx.shape)
    def employee(self):
        j = rd.randint(self.SN, size= [ self.SN])
        phi = -1 + rd.random(self.SN)*2
        phi.shape =len(phi),1
        v = self.x + phi*(self.x - self.x[j])
        for i in range(self.SN):
            v[i][v[i]>self.upb] = self.upb[v[i]>self.upb]
            v[i][v[i]<self.lob] = self.lob[v[i]<self.lob]
        
        fv = np.array(self.f(v))
        sel = fv < self.fx
        self.x[sel] = v[sel][:]
        self.fx[sel] = fv[sel][:]
        self.i += 1
        self.i[sel] = 0
    def observer(self):
        fit = 0.01*max((max(self.fx)-self.fx)) + (max(self.fx)-self.fx)
        if sum(fit) > 0 :
            p = fit/fit.sum()
        else: 
            p = np.ones(fit.shape)
            p = p/p.sum()

        a = rd.choice(self.SN, self.SN, p = p)
        j = rd.randint(self.SN, size= [self.SN])
        phi = -1 + rd.random(self.SN)*2
        phi.shape =len(phi),1
        v = self.x[a] + phi*(self.x[a] - self.x[j])
        for i in range(self.SN):
            v[i][v[i]>self.upb] = self.upb[v[i]>self.upb]
            v[i][v[i]<self.lob] = self.lob[v[i]<self.lob]
        
        fv = np.array(self.f(v))
        sel = fv < self.fx
        self.x[sel] = v[sel][:]
        self.fx[sel] = fv[sel][:]
        self.i += 1
        self.i[sel] = 0
    def explorer(self):
        #print(self.i >= self.limit)
        to_explore = self.i >= self.limit
        nn = sum(to_explore)
        #print(nn)
        if nn > 0:
            self.x[to_explore] = np.array([self.lob + rd.random(self.n)*(self.upb - self.lob) for i in range(nn)])
            self.fx[to_explore] = np.array(self.f(self.x[to_explore]))
            self.i[to_explore] = 0
    def run_(self):
        self.init()
        c = 0
        self.min_fx = [np.min(self.fx)]
        self.min_x = self.x[np.where(self.fx==self.min_fx[-1])[0][0]]
        while c < self.MCN or self.min_x[-1] > self.tol: 
            self.employee()
            self.observer()
            print(c)
            self.min_fx.append(np.min(self.fx))
            self.min_x = self.x[np.where(self.fx==self.min_fx[-1])[0][0]]
            self.explorer()
            c += 1

        
if __name__ == '__main__':

    # pars = {
    #     "NP" : 100,
    #     "NG" : 100000,
    #     "F" : 0.5,
    #     "GR" : 0.75,
    #     "func" : ff.target(ff.ackley, True).f,
    #     "lob" : [-1]*3,
    #     "upb" : [1]*3,
    #     "tol" : 1e-6
    #     }
    
    # ins_0 = DE(pars)
    # ins_0.run_()
    # print(ins_0.min_x)
    
    pars = {
        "SN" : 5,
        "MCN" : 100000,
        "limit" : 50,
        "func" : ff.target(ff.ackley, True).f,
        "lob" : [-1]*3,
        "upb" : [1]*3,
        "tol" : 1e-6
        }
    
    
    ins_1 = ABC(pars)
    ins_1.run_()
    print(ins_1.min_fx)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    