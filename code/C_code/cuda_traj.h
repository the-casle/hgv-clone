//
// Created by Jake Kaslewicz on 11/28/23.
//

#ifndef FINAL_PROJECT_TRAJECTORY_H
#define FINAL_PROJECT_TRAJECTORY_H

#endif //FINAL_PROJECT_TRAJECTORY_H
void read_STL_cuda(AeroAnalysis *aero, const char *fname);
void calc_LD_cuda(AeroAnalysis *aero, double* normals_device, double* area_device, double *F_dev, double alpha, double vel, double rho, double P_inf);
void cudaMemcpyAreasNormals(AeroAnalysis *aero, double** normals_dev, double** area_dev,  double **F_dev);
void cudaFree_AreasNormals(double** normals_dev, double** area_dev, double **F_dev);

void runge_kutta_4_cuda(AeroAnalysis *aero, double* normals_dev, double* area_dev, double *F_dev, double t, double y[neqn], double h, double yp[neqn]);
void EOM_cuda(AeroAnalysis *aero,double* normals_dev,double* area_dev, double *F_dev, double t, double y[neqn], double yp[neqn]);
