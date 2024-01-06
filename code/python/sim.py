import numpy as np
from na_python_version import *
neqn = 5
import os


def run_traj():
    fname = "../geometry/glider_187k.stl"
    fresults = "../data/trajNA187_PY_RK4_2.txt"

    rho, P_inf, L_D = 0.0, 0.0, 0.0
    t, t1, dt = 0, 1, 1
    y = np.zeros(neqn)
    yp = np.zeros(neqn)
    ntri=187140
    print("reading the file")
    na = NA(ntri)
    # Initialize STL data and read STL file
    read_STL(fname,na)
    print("finished reading the file")
    with open(fresults, 'w') as outputFile:
        # Set initial values for the simulation
        y[0] = 0.0    # gamma (0 rads)
        y[1] = 6000.0  # vel (6 km/s)
        y[2] = 60000.0  # altitude (60 km)
        y[3] = 0.0    # downrange (0 km)
        y[4] = 0.35   # alpha (0.35 rads)

        # compute_atmosphere(y[2], rho, P_inf)  # Initialize atmospheric values
        EOM(na, t, y, yp)
        L_D = na.L / na.D
        outputFile.write(f"{t} {y[0]} {y[1]} {y[2]} {y[3]} {y[4]} {L_D}\n")
        print("starting iterations")
        while not (y[1] < 1e3 or y[2] < 1e3 or y[2] > 80e3):
            runge_kutta_4(na, t, y, dt,yp)
            # EOM(na, t, y, yp)
            # y=y+yp*dt
            # L_D = na.L / na.D
            # compute_atmosphere(y[2], rho, P_inf)
            t = t1
            t1 += dt
            outputFile.write(f"{t} {y[0]} {y[1]} {y[2]} {y[3]} {y[4]} {L_D}\n")
            # print(t)
            # if not(t%10):
            #     print(t)
        print("finished iterations")


if __name__ == "__main__":
    os.chdir(r'C:\Users\GreenFluids_VR\Documents\GitHub\CUDA__Project\code\python')
    print(os.cpu_count())
    run_traj()