#include <math.h>
#include <stdio.h>
#include <stdlib.h>  
#include "../trajectory.h"

void compute_g(double h, double *g) {
    const double R = 6.371e6;  // radius of the earth in meters
    const double g_e = 9.80665;  // sea level gravity acceleration

    *g = g_e * pow(R / (R + h), 2);
}
void compute_atmosphere(double h, double *rho, double *P_inf) {
    /*
    rho = rho_0 exp( -Beta h). Where Beta = 0.1378 or g/RT for earth.
    Can then use ideal gas law assumption for remaining freestream conditions. 
    p=rho*R*T
    diverges is an issue with these small functions. 
    */

    const double rho_0 = 1.22500;
    const double Beta = 0.1378; // 1/km
    const double R = 287.00;
    const double T = 270.0; // Lets assume this is constant for simplicity

    *rho = rho_0 * exp(-Beta*h/1e3);
    *P_inf = (* rho)*(R*T);
    double temp=(-Beta*h/1e3);
}