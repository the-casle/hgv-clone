#ifndef trajectory_
#define trajectory_

#define MAX_TRIANGLES 417420
//187140  //417420! Adjust this value as needed
#define neqn 5
#include <math.h>

typedef struct STLData {
    /*
    Get this data as constant or cached chached.
    */
    double (*normals)[3];
    double (*area);
    double ref_area;
} STLData;


typedef struct AeroAnalysis {
    STLData stlData;
    double L,D;
} AeroAnalysis;

void run_traj(AeroAnalysis *aero, FILE *out, bool use_cuda);
void compute_g(double h, double *g);
void compute_atmosphere(double h, double *rho, double *P_inf);
void read_STL(AeroAnalysis *aero, const char *fname, bool use_cuda);
void calc_LD(AeroAnalysis *aero, double alpha, double vel, double rho, double P_inf);
void EOM(AeroAnalysis *aero, double t, double y[neqn], double yp[neqn]);
void initializeSTLData(STLData *stlData);
void freeSTLData(STLData *stlData);
void runge_kutta_4(AeroAnalysis *aero,double t, double y[neqn], double h, double yp[neqn]);
#endif