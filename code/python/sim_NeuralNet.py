import numpy as np
from na_python_version import *
neqn = 5
import os

class plane():
    def __init__(self):
        os.chdir(r'C:\Users\GreenFluids_VR\Documents\GitHub\CUDA__Project\code\python')
        
    def render(self, eps=999, state=None):
        fname = "../geometry/glider_187k.stl"
        self.fresults = "../data/Run 9/trajNA187_PY_RK4_NN_" + str(eps) + ".txt"
        rho, P_inf, L_D = 0.0, 0.0, 0.0
        self.t = 0
        self.dt = 0.5
        self.t1 = np.copy(self.dt)
    
        y = np.zeros(5)
        yp = np.zeros(5)
        ntri=187140
        na = NA(ntri)
        read_STL(fname,na)
        self.cumreward = 0
        if state is None:
            y[0] = 0.0    # gamma (0 rads)
            y[1] = np.random.randint(5000, 8000) #6000.0  # vel (6 km/s) 
            y[2] = np.random.randint(50000, 76200) #76200  # altitude (60 km)
            y[3] = 0.0    # downrange (0 km)
            y[4] = np.random.uniform(0.10, 0.35) #0.35  # alpha
        else:
            y = np.copy(state)

        self.na = na
        self.y = y
        self.yp = yp
        EOM(self.na, self.t, self.y, self.yp)
        self.L_D = na.L / na.D
        self.startwrite = False
        self.write()

        return self.y

    def step(self, action):
        del_alpha = np.pi/180 if action == 1 else -np.pi/180
        state = np.copy(self.y)
        self.y[4] += del_alpha
        done = False
        self.y = runge_kutta_4(self.na, self.t, self.y, self.dt, self.yp)
        reward = -(180/np.pi * self.y[0]) ** 2  
        if self.y[1] < 1e3 or self.y[2] > 85e3 or self.y[4] < -np.pi/2 or self.y[4] > np.pi/2:
            done = True
            reward += -1e5
        if self.y[3] < 1e3 or self.y[2] > 85e3:
            reward += -1e6
        self.t = self.t1
        self.t1 += self.dt
        self.L_D = self.na.L / self.na.D
        self.cumreward += reward
        self.write()
        return state, self.y, reward, done
    
    def write(self):
        if self.startwrite == False:
            with open(self.fresults, 'w') as outputFile:
                outputFile.write(f"{self.t} {self.y[0]} {self.y[1]} {self.y[2]} {self.y[3]} {self.y[4]} {self.L_D} {self.cumreward}\n")
            self.startwrite = True
        else:
            with open(self.fresults, 'a') as outputFile:
                outputFile.write(f"{self.t} {self.y[0]} {self.y[1]} {self.y[2]} {self.y[3]} {self.y[4]} {self.L_D} {self.cumreward}\n")