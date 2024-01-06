#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../trajectory.h"
#include "../cuda_traj.h"


void initializeSTLData(STLData *stlData) {
    stlData->normals = (double (*)[3])malloc(MAX_TRIANGLES * sizeof(double[3]));
    stlData->area = (double *)malloc(MAX_TRIANGLES * sizeof(double));
    if (stlData->normals == NULL || stlData->area == NULL) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }
}

void freeSTLData(STLData *stlData) {
    free(stlData->normals);
    free(stlData->area);
}

void read_STL_ref(AeroAnalysis *aero, const char *fname) {
     /*
    Maybe find a way to accomplish this on the GPU?
    */

    FILE *file = fopen(fname, "r");
    if (!file) {
        perror("Error opening file. Input file doesn't exist.");
        exit(EXIT_FAILURE);
    }

    char void_chars[10];
    double vec1[3], vec2[3], area_[3], r1[3], r2[3], r3[3];

    aero->stlData.ref_area = 0.0;

    // Skip the first line
    fscanf(file, "%*[^\n]\n");

    for (int i = 0; i < MAX_TRIANGLES; i++) {
        fscanf(file, "%s %s %lf %lf %lf\n", void_chars, void_chars,         \
                &aero->stlData.normals[i][0], &aero->stlData.normals[i][1], \
                &aero->stlData.normals[i][2]);
        fscanf(file, "%*s\n%*s\n");
        fscanf(file, "%s %lf %lf %lf\n", void_chars, &r1[0], &r1[1], &r1[2]);
        fscanf(file, "%s %lf %lf %lf\n", void_chars, &r2[0], &r2[1], &r2[2]);
        fscanf(file, "%s %lf %lf %lf\n", void_chars, &r3[0], &r3[1], &r3[2]);
        fscanf(file, "%*s\n%*s\n");

        // Compute the areas
        vec1[0] = r1[0] - r3[0];
        vec1[1] = r1[1] - r3[1];
        vec1[2] = r1[2] - r3[2];

        vec2[0] = r1[0] - r2[0];
        vec2[1] = r1[1] - r2[1];
        vec2[2] = r1[2] - r2[2];
        // Perform the cross product to get area.
        area_[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
        area_[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
        area_[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];

        aero->stlData.area[i] = 0.5 * sqrt(area_[0] * area_[0] + area_[1] * area_[1] + area_[2] * area_[2]);

        // Compute reference area
        aero->stlData.ref_area += aero->stlData.area[i];
    }

    fclose(file);
}

void read_STL(AeroAnalysis *aero, const char *fname, bool use_cuda) {
    if(use_cuda){
        read_STL_cuda(aero, fname);
    } else {
        read_STL_ref(aero, fname);
    }
}

void calc_LD_ref(AeroAnalysis *aero, double alpha, double vel, double rho, double P_inf) {
    double u[3], F_Total[3];
    double P, q, vel_, cp;

    q = 0.5 * rho * vel * vel;
    u[0] = cos(alpha);
    u[1] = sin(alpha);
    u[2] = 0.0;
    F_Total[0] = 0.0;
    F_Total[1] = 0.0;
    F_Total[2] = 0.0;

    for (int i = 0; i < MAX_TRIANGLES ; i++) {
        vel_ = u[0] * aero->stlData.normals[i][0] + u[1] * aero->stlData.normals[i][1] + u[2] * aero->stlData.normals[i][2];

        if (vel_ < 0.0) {
            cp = 2 * vel_ * vel_;
        } else {
            cp = 0;
        }

        P = (cp * q) + P_inf;
        F_Total[0] += (P * aero->stlData.area[i]) * (-aero->stlData.normals[i][0]);
        F_Total[1] += (P * aero->stlData.area[i]) * (-aero->stlData.normals[i][1]);
        F_Total[2] += (P * aero->stlData.area[i]) * (-aero->stlData.normals[i][2]);
    }
    /*
    A reduction algo will have to be done here. Then an attomic operaiton to get L and D. Also, don't forget about hardware accelearation for 
    trig functions. 
    */
    
    aero->L = F_Total[1] * cos(alpha) - F_Total[0] * sin(alpha);
    aero->D = F_Total[1] * sin(alpha) + F_Total[0] * cos(alpha);
    // printf("calc_LD:%lf,%lf,%lf,%lf ,%lf \n", rho,P_inf,aero->stlData.ref_area,aero->L,aero->D);
}

void calc_LD(AeroAnalysis *aero, double alpha, double vel, double rho, double P_inf) {
    calc_LD_ref(aero, alpha, vel, rho, P_inf);
}
