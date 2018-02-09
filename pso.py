from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import random

def HCEFunc(x):
    D = len(x)
    R = 0
    for i in range(D):
        R += ((10 ** 6) ** (i / (D-1))) * (x[i] ** 2)
    return R 

def BCFunc(x):
    D = len(x)
    R = x[0]**2
    for i in range(1, D):
        R += (10**6) * (x[i]**2)
    return R

def DFunc(x):
    D = len(x)
    R = (10**6) * (x[0]**2)
    for i in range(1, D):
        R += x[i]**2
    return R

def RoFunc(x):
    D = len(x)
    R = 0
    for i in range(D-1):
        R += (100*(((x[i]**2)-x[i+1])**2) + (x[i]-1)**2)
    return R

def AFunc(x):
    D = len(x)
    R = 0
    a = 0
    b = 0
    for i in range(D):
        a += x[i]**2
        b += np.cos(2*np.pi*x[i])
    R = -20*np.exp(-0.2*np.sqrt(a/D)) - np.exp(b/D) + 20 + np.e
    return R

def WFunc(x):
    D = len(x)
    R = 0
    a = 0.5
    b = 3
    kmax = 20
    p1 = 0
    p2 = 0
    for i in range(D):
        for k in range(0, kmax):
            p1 += (a**k) * np.cos(2*np.pi*(b**k)*(x[i] + 0.5))
    for k in range(0, kmax):
        p2 += (a**k) * np.cos(2*np.pi*(b**k)*0.5)
    R = p1 - (D*p2)
    return R

def GFunc(x):
    D = len(x)
    a = 0
    b = 1
    for i in range(D):
        a += x[i]**2
    for i in range(D):
        b *= np.cos(float(x[i]) / np.sqrt(i+1))
    R = 1 + (float(a)/4000.0) - float(b)
    return R

def RaFunc(x):
    D = len(x)
    R = 0
    for i in range(D):
        R += x[i]**2 - (10*np.cos(2*np.pi*x[i])) + 10
    return R

def KFunc(x):
    D = len(x)
    abcd = 10/(D**2)
    p1 = 0
    p2 = 1
    R = 0
    for i in range(D):
        for j in range(1, 33):
            p1 += (abs((2**j)*x[i]-int((2**j)*x[i]))/(2**j))
        p2 *= (1 + i * p1) ** (10/(D**1.2))
    R = (abcd * p2) - abcd
    return R

class Particle:
    def __init__(self, x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0, num_dimensions):
            self.velocity_i.append(random.uniform(-10, 10))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self, costFunc):
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i


    # update new particle velocity
    def update_velocity(self, pos_best_g, w):
        c1=2.05         # cognative constant
        c2=2.05         # social constant

        for i in range(0, num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]

class PSO():
    def __init__(self, costFunc, x0, bounds, num_particles, maxiter):
        global num_dimensions
        temp = []
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0, num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            w = .9 - (i / maxiter) * (0.5)         # inertia weight (how much to weigh the previous velocity)
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0, num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)
                    temp.append(err_best_g)

            # cycle through swarm and update velocities and position
            for j in range(0, num_particles):
                swarm[j].update_velocity(pos_best_g, w)
                swarm[j].update_position(bounds)
            i += 1

        self.result = err_best_g
        err_best_g_ar.append(temp)

if __name__ == "__PSO__":
    main()

#--- RUN ----------------------------------------------------------------------+

global a
global err_best_g_ar

cost_func = [KFunc, RaFunc, GFunc, WFunc, AFunc, RoFunc, DFunc, BCFunc, HCEFunc]

for f in cost_func:
    print(str(f))
    d = 5                   # must be changed for the number of dimensions (2, 5, 10)
    counta = 0
    countb = 0
    temp_long = 0
    temp_count = 0
    bound = (-10, 10)
    temp_short = 3000
    err_best_g_ar = []
    plotting = []
    initial = []            # initial starting location [x1,x2...]
    bounds = []             # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
    result = []
    for a in range(10):
        for o in range(d):
            rando = random.uniform(-10, 10)
            initial.append(rando)
            bounds.append(bound)
        print("test {}".format(a+1))
        p = PSO(f, initial, bounds, num_particles=100, maxiter = 5000*d)
        result.append(p.result)
        initial.clear()     # must clear initial or there will be too many and the ones that will be compared will always be the same
        bounds.clear()      # must clear bounds so that it is random each time

    for each in err_best_g_ar:
        if len(each) < temp_short:
            counta = temp_count
            temp_short = len(each)
        elif len(each) > temp_long:
            countb = temp_count
            temp_long = len(each)
        else:
            '''nothing to be done'''
        temp_count += 1
    
    print("longest {}: {}".format(countb, len(err_best_g_ar[countb])))
    
    #--- PLOT ---------------------------------------------------------------------+
    
    data_lengthb = len(err_best_g_ar[countb])
    X = np.arange(0, data_lengthb, 1)
    Y = err_best_g_ar[countb]
    plt.plot(X, Y)
    plt.show()

#--- END ----------------------------------------------------------------------+

