#include <stdio.h>
#include <stdlib.h>
#include "trajectory.h"
#include "cuda_traj.h"

#include <time.h>

// #pragma GCC diagnostic error "-Wframe-larger-than="
void run_traj(AeroAnalysis *aero, FILE *out, bool use_cuda) {
    // const int neqn = 5;
    double rho, P_inf, L_D;
    // int flag, err;
    double relerr, t, t1, dt;
    double y[neqn], yp[neqn];

    // Have some default values for integration and starting values for simulation.
    // abserr = 1e-9;
    // relerr = 1e-9;
    // flag = 1;
    t = 0.0;
    dt = 1;  // Usually 1 s for RK4, else 0.1 for standard Euler. Can get away with 2 seconds
    t1 = dt + t;
    y[0] = 0.0;      // gamma (0 rads)
    y[1] = 6000.0;   // vel (6 km/s)
    y[2] = 76200.0;  // altitude (60 km)
    y[3] = 0.0;      // downrange (0 km)
    y[4] = 0.35;     // alpha (0.35 rads)

    compute_atmosphere(y[2], &rho, &P_inf);  // Initialize the correct atmospheric values.

    EOM(aero,t, y, yp);  // Initialize the yp or y' or dy/dt vector
    L_D= aero->L/aero->D;
    fprintf(out, "%lf %lf %lf %lf %lf %lf %lf\n", t, y[0], y[1],y[2],y[3],y[4], L_D);//, rho, 0.5 * rho * y[1] * y[1]);
    
    if(use_cuda){
        double *normals_dev;
        double *area_dev;
        double *F_dev;
        cudaMemcpyAreasNormals(aero, &normals_dev, &area_dev, &F_dev);
        do {
            runge_kutta_4_cuda(aero,normals_dev,area_dev, F_dev, t,y,dt,yp);
            L_D= aero->L/aero->D;
            compute_atmosphere(y[2], &rho, &P_inf);  // Get atmospheric values at new height
            fprintf(out, "%lf %lf %lf %lf %lf %lf %lf\n", t, y[0], y[1],y[2],y[3],y[4], L_D);//, rho, 0.5 * rho * y[1] * y[1]);
            // Update the time t
            t = t1;
            t1 = t1 + dt;

        } while (!(y[1] < 1e3 || y[2] < 1e3 || y[2] > 80e3));
        cudaFree_AreasNormals(&normals_dev, &area_dev, &F_dev);
    }else{
        do {

            runge_kutta_4(aero,t,y,dt,yp);
            // ! preform time integration using Runga-Kutta with 4 stages.
            // ! else we can do a neive approach
            // ! y=y+yp*dt->
            // ! call EOM(t,y,yp) to get new derivative values for next time step
            // y=y+yp*dt // standard euler approach we need to possibly move to RK-4.


            L_D= aero->L/aero->D;
            compute_atmosphere(y[2], &rho, &P_inf);  // Get atmospheric values at new height
            fprintf(out, "%lf %lf %lf %lf %lf %lf %lf\n", t, y[0], y[1],y[2],y[3],y[4], L_D);//, rho, 0.5 * rho * y[1] * y[1]);


            // Update the time t
            t = t1;
            t1 = t1 + dt;

        } while (!(y[1] < 1e3 || y[2] < 1e3 || y[2] > 80e3));
    }


    // ! If the altitude y(3) is less than 1km exit the loop/ stop the simulation
    // ! If the altitude y(3) is greater than 80km exit the loop/ stop the simulation. You are about to be in space my guy.
    // ! If the speed y(2) is less than 1km/s exit the loop/ stop the simulation

}




void EOM(AeroAnalysis *aero, double t, double y[neqn], double yp[neqn]) {
    double gamma, v, alpha, h, x,g;
    double rho, P_inf, T_inf;
    const double R = 6.371e6;  // radius of the earth in meters
    const double mass = 1.0e3;  // assumed mass is fixed
    int err;

    gamma = y[0];
    v = y[1];
    h = y[2];
    x = y[3];
    alpha = y[4];

    compute_atmosphere(h, &rho, &P_inf);  // return rho, and P_inf
    calc_LD(aero, alpha, v, rho, P_inf); // computes the cd and L_D where L_D is a global variable from NA
    compute_g(h, &g);
    // printf("EOM:%lf, %lf,%lf, %lf\n", aero->L, g, rho,P_inf);
    yp[0] = (1.0 / v) * (-aero->L / mass + g * cos(gamma) - (v * v / (R + h)) * cos(gamma));  // flight path angle
    yp[1] = -aero->D / mass + g * sin(gamma);  // Velocity (truely air speed)
    yp[2] = -v * sin(gamma);  // height in meters
    yp[3] = v * cos(gamma);  // cross range distance
    yp[4] = 0.0;  // alpha doesn't change here
}
void runge_kutta_4(AeroAnalysis *aero,double t, double y[neqn], double h, double yp[neqn]) {
    double k1[neqn], k2[neqn], k3[neqn], k4[neqn];

    // EOM(aero,t, y, k1);
     for (int i = 0; i < neqn; i++) {
        k1[i] = yp[i];
    }
    for (int i = 0; i < neqn; i++) {
        k2[i] = y[i] + h / 2.0 * k1[i];
    }
    EOM(aero,t + h / 2.0, k2, k2);

    for (int i = 0; i < neqn; i++) {
        k3[i] = y[i] + h / 2.0 * k2[i];
    }
    EOM(aero,t + h / 2.0, k3, k3);

    for (int i = 0; i < neqn; i++) {
        k4[i] = y[i] + h * k3[i];
    }
    EOM(aero,t + h, k4, k4);

    for (int i = 0; i < neqn; i++) {
         yp[i] =  (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])/6.0;
         y[i] = y[i]+yp[i]*h;
    }

    // EOM(aero,t+h, y, yp);
}




