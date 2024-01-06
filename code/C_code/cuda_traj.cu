#include <stdio.h>
#include <stdlib.h>
#include "trajectory.h"
#include<iostream>

#define BLOCK_SIZE 64

// Reducing large data array onto one grid of concurrent threads.
__global__ void big_reduction(double *in_data, int n){
    unsigned int t = blockDim.y*blockIdx.y + threadIdx.y;
    unsigned int ele = threadIdx.x;

    // Each stride is the height of the grid launch.
    int stride = gridDim.y * blockDim.y;

    // Local register to hold reduction value for this thread.
    double collapse = 0;
    for(int i = t; i < n; i += stride){
        collapse += in_data[i * 3 + ele];
    }

    // Write register value back to global memory.
    in_data[t * 3 + ele] = collapse;
}

// Tree reduction that reduces within the grid to reach a single value.
// The reduction is performed on the x,y,z depending on the ele index.
// The x,y,z reductions are independent from each other.
// Memory accesses are coalesced in this pattern because the 2D grid
__global__ void tree_reduction(double *in_data, int n)
{
    __shared__ double partialSum[2 * BLOCK_SIZE * 3];

    unsigned int t = threadIdx.y;
    unsigned int ele = threadIdx.x;
    unsigned int start = 2*blockDim.y*blockIdx.y;

    // Loading first value of this thread's reduction into shared memory.
    if (start + t < n) {
        partialSum[t * 3 + ele] = in_data[(start + t) * 3 + ele];
    } else {
        partialSum[t * 3 + ele] = 0;
    }
    // Loading second value of this thread's reduction into shared memory.
    if (start + blockDim.y + t < n) {
        partialSum[(blockDim.y + t) * 3 + ele] = in_data[(start + blockDim.y + t) * 3 + ele];
    } else {
        partialSum[(blockDim.y + t) * 3 + ele] = 0;
    }

    // Striding across the block, dropping threads off that are no longer needed (t < stride)
    for (unsigned int stride = blockDim.y; stride >= 1;  stride >>= 1) {
        __syncthreads();
        if (t < stride){
            partialSum[t * 3 + ele] += partialSum[(t+stride) * 3 + ele];
        }
    }

    // No sync needed because only one thread (0) will be writing to this location.
    if(t == 0){
        in_data[blockIdx.y * 3 + ele] = partialSum[0 * 3 + ele];
    }
}

// Calculating lift over drag.
__global__ void calc_LD_kernel( double *normals, double* area, double u0, double u1, double u2, double* F_Total, double q, double P_inf){
    int i = blockDim.x * blockIdx.x + threadIdx.x; 
    double vel_, P, cp;

    // We are using the tranposed accesses below so that we get coalesced memory accesses.
    // Necessary when using a 1D grid (which we need to because using x,y,z norms in one calc).
    if(i<MAX_TRIANGLES){

        // Placing global memory reads into registers to minimize reads to global.
        double norm0 = normals[0 * MAX_TRIANGLES + i];
        double norm1 = normals[1 * MAX_TRIANGLES + i];
        double norm2 = normals[2 * MAX_TRIANGLES + i];

        vel_ = u0 * norm0 + u1 * norm1 + u2 * norm2;

        if (vel_ < 0.0) {
            cp = 2 * vel_ * vel_;
        } else {
            cp = 0;
        }

        P = (cp * q) + P_inf;

        // Again loading into register before subsequent calculations to avoid multiple reads from global.
        double p_area = P * area[i];
        F_Total[i * 3 + 0] = p_area * -norm0;
        F_Total[i * 3 + 1] = p_area * -norm1;
        F_Total[i * 3 + 2] =  p_area * -norm2;
    }
}

// Transposing data. Better than corner turning into shared memory because we reuse the data.
// Therefore one transpose at beginning instead corner turning each time step.
void transpose(const double* src, double* dst, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

// Alloc and copying data to the gpu. Performed once at beginning of calculation to minimize data transfers.
void cudaMemcpyAreasNormals(AeroAnalysis *aero, double** normals_dev, double** area_dev, double **F_dev){
    cudaMalloc((void**)normals_dev, MAX_TRIANGLES * sizeof(double) * 3);
    cudaMalloc((void**)F_dev, MAX_TRIANGLES * sizeof(double) * 3);
    cudaMalloc((void**)area_dev, sizeof(double)*MAX_TRIANGLES);
    double *transposed = (double *) malloc(sizeof(double) * MAX_TRIANGLES * 3 * 2);
    transpose((double *)aero->stlData.normals, (double *)transposed, MAX_TRIANGLES, 3);
    cudaMemcpy(*normals_dev, transposed, MAX_TRIANGLES * sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(*area_dev, aero->stlData.area, sizeof(double)*MAX_TRIANGLES, cudaMemcpyHostToDevice);
    free(transposed);
}

// Freeing gpu data
void cudaFree_AreasNormals(double** normals_dev, double** area_dev, double **F_dev){
    cudaFree(*normals_dev);
    cudaFree(*area_dev);
    cudaFree(*F_dev);
}

// Static accesses to cudaDeviceProp to avoid calling cudaGetDeviceProperties many times.
cudaDeviceProp getProp(void){
    static cudaDeviceProp prop;
    static bool populated;
    if(!populated){
        cudaGetDeviceProperties(&prop, 0);
        populated = true;
    }
    return prop;
}

// Calculating lift over drag. Responsible for launching kernels.
void calc_LD_cuda(AeroAnalysis *aero, double* normals_device, double* area_device, double *F_device, double alpha, double vel, double rho, double P_inf) {
    double u[3], F_Total[3];
    double q = 0.5 * rho * vel * vel;
    u[0] = cos(alpha);
    u[1] = sin(alpha);
    u[2] = 0.0;
    F_Total[0] = 0.0;
    F_Total[1] = 0.0;
    F_Total[2] = 0.0;

    // Launching the initial calc_LD kernel to find vector at each mesh triangle.
    dim3 gridDim((MAX_TRIANGLES + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE);
    calc_LD_kernel<<<gridDim, blockDim>>>(normals_device, area_device, u[0], u[1], u[2], F_device, q, P_inf);

    // Getting dev properties to maximize occupancy of the big reduction.
    cudaDeviceProp prop = getProp();
    int  multi_count = prop.multiProcessorCount;
    int  threads_per_multi = prop.maxThreadsPerMultiProcessor;
    int threads_at_same_time = threads_per_multi * multi_count;

    // We round down so that there are no warps waiting in queue slowing down reduction.
    // Every thread will always be active.
    int grid_dim = threads_at_same_time / (BLOCK_SIZE * 3); // Rounding down
    int elem_reduced_to = grid_dim * BLOCK_SIZE;

    dim3 big_grid(1, grid_dim, 1);
    dim3 big_block(3, BLOCK_SIZE, 1);

    // Launching the big reduction kernel with calc_LD results.
    big_reduction<<<big_grid, big_block>>>(F_device, MAX_TRIANGLES);
    cudaDeviceSynchronize();

    // Starting the tree reduction algorithm.
    int remaining_elements = elem_reduced_to;

    // Continuously launch kernels until data fits into one block.
    // Remaining elements is in terms of each triangle, not each x,y,z.
    while(remaining_elements > 1){
        int half_elements = (remaining_elements + 2 - 1) / 2;
        int number_blocks = (half_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 rud_grid(1, number_blocks, 1);
        dim3 rud_block(3, BLOCK_SIZE, 1);
        tree_reduction<<<rud_grid, rud_block, sizeof(double) * BLOCK_SIZE * 3 * 2>>>(F_device, remaining_elements);
        cudaDeviceSynchronize();
        remaining_elements = number_blocks;
    }

    //cudamemcpy results from device to host
    cudaMemcpy(F_Total, F_device, sizeof(double) * 3, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // Finish calculating Lift over Drag from the reduced vector.
    aero->L = F_Total[1] * cos(alpha) - F_Total[0] * sin(alpha);
    aero->D = F_Total[1] * sin(alpha) + F_Total[0] * cos(alpha);
}
void EOM_cuda(AeroAnalysis *aero,double* normals_dev,double* area_dev, double *F_dev, double t, double y[neqn], double yp[neqn]){
    double gamma, v, alpha, h,g;
    double rho, P_inf;
    const double R = 6.371e6;  // radius of the earth in meters
    const double mass = 1.0e3;  // assumed mass is fixed

    gamma = y[0];
    v = y[1];
    h = y[2];
    alpha = y[4];

    compute_atmosphere(h, &rho, &P_inf);  // return rho, and P_inf
    calc_LD_cuda(aero,normals_dev, area_dev, F_dev, alpha, v, rho, P_inf); // computes the cd and L_D where L_D is a global variable from NA
    compute_g(h, &g);
    yp[0] = (1.0 / v) * (-aero->L / mass + g * cos(gamma) - (v * v / (R + h)) * cos(gamma));  // flight path angle
    yp[1] = -aero->D / mass + g * sin(gamma);  // Velocity (truely air speed)
    yp[2] = -v * sin(gamma);  // height in meters
    yp[3] = v * cos(gamma);  // cross range distance
    yp[4] = 0.0;  // alpha doesn't change here
}
void runge_kutta_4_cuda(AeroAnalysis *aero, double* normals_dev, double* area_dev, double *F_dev, double t, double y[neqn], double h, double yp[neqn]){
    double k1[neqn], k2[neqn], k3[neqn], k4[neqn];

    // EOM(aero,t, y, k1);
     for (int i = 0; i < neqn; i++) {
        k1[i] = yp[i];
    }
    for (int i = 0; i < neqn; i++) {
        k2[i] = y[i] + h / 2.0 * k1[i];
    }
    EOM_cuda(aero,normals_dev,area_dev, F_dev, t + h / 2.0, k2, k2);

    for (int i = 0; i < neqn; i++) {
        k3[i] = y[i] + h / 2.0 * k2[i];
    }
    EOM_cuda(aero,normals_dev,area_dev,F_dev, t + h / 2.0, k3, k3);

    for (int i = 0; i < neqn; i++) {
        k4[i] = y[i] + h * k3[i];
    }
    EOM_cuda(aero,normals_dev,area_dev,F_dev, t + h, k4, k4);

    for (int i = 0; i < neqn; i++) {
         yp[i] =  (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])/6.0;
         y[i] = y[i]+yp[i]*h;
    }

}

// Kernel we did not time/use because the data transfer took too long.
__global__ void area_kernel(double* r1, double* r2, double* r3, double* stl_area, double* stl_ref_area){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<MAX_TRIANGLES){
        // printf("%d ", i);
        double vec1[3], vec2[3], area_[3];

        // Compute the areas
        vec1[0] = r1[i*3+0] - r3[i*3+0];
        vec1[1] = r1[i*3+1] - r3[i*3+1];
        vec1[2] = r1[i*3+2] - r3[i*3+2];

        vec2[0] = r1[i*3+0] - r2[i*3+0];
        vec2[1] = r1[i*3+1] - r2[i*3+1];
        vec2[2] = r1[i*3+2] - r2[i*3+2];

        // Perform the cross product to get area
        area_[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
        area_[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
        area_[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];

        stl_area[i] = 0.5 * sqrt(area_[0] * area_[0] + area_[1] * area_[1] + area_[2] * area_[2]);

        // Compute reference area -  reduction
        // atomicAdd(stl_ref_area, stl_area[i]);
    }
}

// Similarly, cuda kernel launch we did not use.
void read_STL_cuda(AeroAnalysis *aero, const char *fname) { //reads inputs and converts to area

    FILE *file = fopen(fname, "r");
    if (!file) {
        perror("Error opening file. Input file doesn't exist.");
        exit(EXIT_FAILURE);
    }

    char void_chars[10];

    double* r1 = (double*)malloc(sizeof(double)*MAX_TRIANGLES*3);
    double* r2 = (double*)malloc(sizeof(double)*MAX_TRIANGLES*3);
    double* r3 = (double*)malloc(sizeof(double)*MAX_TRIANGLES*3);

    aero->stlData.ref_area = 0.0;

    // Skip the first line
    fscanf(file, "%*[^\n]\n");

    // First loop: Read and store data
    for (int i = 0; i < MAX_TRIANGLES; i++) {
        fscanf(file, "%s %s %lf %lf %lf\n", void_chars, void_chars, \
                &aero->stlData.normals[i][0], &aero->stlData.normals[i][1], \
                &aero->stlData.normals[i][2]);
        fscanf(file, "%*s\n%*s\n");
        fscanf(file, "%s %lf %lf %lf\n", void_chars, &r1[i*3+0], &r1[i*3+1], &r1[i*3+2]);
        fscanf(file, "%s %lf %lf %lf\n", void_chars, &r2[i*3+0], &r2[i*3+1], &r2[i*3+2]);
        fscanf(file, "%s %lf %lf %lf\n", void_chars, &r3[i*3+0], &r3[i*3+1], &r3[i*3+2]);
        fscanf(file, "%*s\n%*s\n");
    }

    double* r1dev;
    double* r2dev;
    double* r3dev;
    double* stl_area_dev;
    double* stl_ref_area_dev;

    cudaMalloc((void**)&r1dev,sizeof(double)*MAX_TRIANGLES*3);
    cudaMalloc((void**)&r2dev,sizeof(double)*MAX_TRIANGLES*3);
    cudaMalloc((void**)&r3dev,sizeof(double)*MAX_TRIANGLES*3);
    cudaMalloc((void**)&stl_area_dev,sizeof(double)*MAX_TRIANGLES);
    cudaMalloc((void**)&stl_ref_area_dev,sizeof(double));

    cudaMemcpy(r1dev, r1, sizeof(double)*MAX_TRIANGLES*3, cudaMemcpyHostToDevice);
    cudaMemcpy(r2dev, r2, sizeof(double)*MAX_TRIANGLES*3, cudaMemcpyHostToDevice);
    cudaMemcpy(r3dev, r3, sizeof(double)*MAX_TRIANGLES*3, cudaMemcpyHostToDevice);

    dim3 gridDim((MAX_TRIANGLES + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE);
    area_kernel<<<gridDim, blockDim>>>(r1dev, r2dev, r3dev, stl_area_dev, stl_ref_area_dev);

    cudaMemcpy(aero->stlData.area, stl_area_dev, sizeof(double)*MAX_TRIANGLES, cudaMemcpyDeviceToHost);
    cudaMemcpy(&aero->stlData.ref_area, stl_ref_area_dev, sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    free(r1);
    free(r2);
    free(r3);
    cudaFree(r1dev);
    cudaFree(r2dev);
    cudaFree(r3dev);

    fclose(file);
}
