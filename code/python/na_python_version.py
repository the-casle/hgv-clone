import numpy as np
from math import sin, cos
from concurrent.futures import ThreadPoolExecutor
import os

from  atmosphere import *
neqn=5

class NA:
    def __init__(self, ntri):
        self.ntri = ntri
        self.area = np.zeros(ntri) 
        self.normals = np.zeros((ntri,3)) 
        self.L= 0.0
        self.D= 0.0
        self.ref_area = 0.0

def read_STL(fname, na):
    vec1 = np.zeros(3)
    vec2 = np.zeros(3)
    area_ = np.zeros(3)
    r1 = np.zeros(3)
    r2 = np.zeros(3)
    r3 = np.zeros(3)

    na.ref_area = 0.0

    with open(fname, 'r') as f:
        # Skip the first line
        next(f)
        
        for i,line in enumerate(f):
            if i >=(na.ntri):
                break
            values = line.split()
            na.normals[i, :] = [float(val) for val in values[2:5]]
            next(f)  # Skip non-numbers
            # next(f)  # Skip non-numbers

            values = next(f).split()
            r1[:] = [float(val) for val in values[1:4]]

            values = next(f).split()
            r2[:] = [float(val) for val in values[1:4]]

            values = next(f).split()
            # print(values)
            r3[:] = [float(val) for val in values[1:4]]

            next(f)  # Skip non-numbers
            next(f)  # Skip non-numbers

            vec1 = r1 - r3
            vec2 = r1 - r2

            area_ = np.cross(vec1, vec2)
            na.area[i] = 0.5 * np.linalg.norm(area_)

            na.ref_area += na.area[i]
            

def calc_LD_single(na, alpha, vel, rho, P_inf):
    u = np.array([np.cos(alpha), np.sin(alpha), 0.0], dtype=np.float64)
    q = 0.5 * rho * vel**2

    vel_ = np.dot(na.normals, u)
    cp = np.where(vel_ < 0.0, 2 * vel_**2, 0.0)
    P = (cp * q) + P_inf
    F_Total = np.sum((P * na.area)[:, np.newaxis] * (-na.normals), axis=0)
    return F_Total

def calc_LD(na, alpha, vel, rho, P_inf):
    F_Total = calc_LD_single(na, alpha, vel, rho, P_inf)

    na.L = F_Total[1] * np.cos(alpha) - F_Total[0] * np.sin(alpha)
    na.D = F_Total[1] * np.sin(alpha) + F_Total[0] * np.cos(alpha)
# def calc_LD_single(na, alpha, vel, rho, P_inf, i):
#     u = np.array([cos(alpha), sin(alpha), 0.0])
#     F_Total = np.zeros(3)
#     q = 0.5 * rho * vel**2

#     vel_ = np.dot(u, na.normals[i, :])
#     cp = 2 * vel_**2 if vel_ < 0.0 else 0.0
#     P = (cp * q) + P_inf
#     F_Total += (P * na.area[i]) * (-na.normals[i, :])

#     return F_Total

# def calc_LD(na, alpha, vel, rho, P_inf):
#     max_workers = os.cpu_count()
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [executor.submit(calc_LD_single, na, alpha, vel, rho, P_inf, i) for i in range(na.ntri)]

#         F_Total = np.zeros(3)
#         for future in futures:
#             F_Total += future.result()

#     na.L = F_Total[1] * cos(alpha) - F_Total[0] * sin(alpha)
#     na.D = F_Total[1] * sin(alpha) + F_Total[0] * cos(alpha)

def EOM(na, t, y, yp):
    gamma, v, h, x,alpha, g = y[0], y[1], y[2], y[3], y[4], 0.0  # Initialize g as a placeholder
    mass=1e3
    rho, P_inf = compute_atmosphere(h)
    calc_LD(na, alpha, v, rho, P_inf)
    g = compute_g(h)
    R = 6.371e6
    yp[0] = (1.0 / v) * (-na.L / mass + g * cos(gamma) - (v * v / (R + h)) * cos(gamma))  # flight path angle
    yp[1] = -na.D / mass + g * sin(gamma)  # Velocity (truely air speed)
    yp[2] = -v * sin(gamma)  # height in meters
    yp[3] = v * cos(gamma)  # cross range distance
    yp[4] = 0.0  # alpha doesn't change here

def runge_kutta_4(na, t, y, h, yp):
    k1 = yp

    k2 = (y + h / 2.0 * k1)
    EOM(na, t + h / 2.0, k2, k2)

    k3 = (y + h / 2.0 * k2)
    EOM(na, t + h / 2.0, k3, k3)

    k4 = (y + h * k3)
    EOM(na, t + h, k4, k4)

    yp= (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    y+=yp*h

    return y
