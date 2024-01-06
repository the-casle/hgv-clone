#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <math.h>
#include "trajectory.h"
#include <string.h>

#include <time.h>



bool areAlmostEqual(double a, double b, double tolerance) {
    return fabs(a - b) < tolerance;
}

void compare_outputs(FILE *file1, FILE *file2) {

    const int number_outs = 7;

    double values1[number_outs], values2[number_outs];
    int line = 1;
    bool mismatchFound = false;
    double tolerance[7] = {1e-6, 1e-6, 1e0, 1e0, 1e0, 1e-6, 1e-6};  // Set your tolerance

    while (!feof(file1) && !feof(file2)) {
        if (fscanf(file1, "%lf %lf %lf %lf %lf %lf %lf",
                   &values1[0], &values1[1], &values1[2], &values1[3],
                   &values1[4], &values1[5], &values1[6]) != number_outs) {
            break; // Incomplete or corrupt line
        }

        if (fscanf(file2, "%lf %lf %lf %lf %lf %lf %lf",
                   &values2[0], &values2[1], &values2[2], &values2[3],
                   &values2[4], &values2[5], &values2[6]) != number_outs) {
            break; // Incomplete or corrupt line
        }

        for (int i = 0; i < number_outs; i++) {
            if (!areAlmostEqual(values1[i], values2[i], tolerance[i])) {
                printf("Mismatch found at line %d, value %d: CPU=%lf, GPU=%lf, with tolerance: %lf\n",
                       line, i+1, values1[i], values2[i], tolerance[i]);
                mismatchFound = true;
                break;
            }
        }

        line++;
    }

    if (!mismatchFound) {
        printf("Tests Passed.\n");
    }
}

int main(int argc, char* argv[]){
    printf("Starting Test harness...\n");
    char fname[] = "../geometry/glider_417k.stl";
    char freference_out[] = "../data/trajNA417_C_RK4.txt";
    char fcuda_out[] = "../data/trajNA417_C_RK4_cuda.txt";
    if(argc>=2){
        if(strcmp(argv[1], "large")==0){
            strcpy(fname, "../geometry/glider_417k.stl");
            strcpy(freference_out, "../data/trajNA417_C_RK4.txt");
            strcpy(fcuda_out, "../data/trajNA417_C_RK4_cuda.txt");
        }else if(strcmp(argv[1], "small")==0){
            strcpy(fname, "../geometry/glider_187k.stl");
            strcpy(freference_out, "../data/trajNA187_C_RK4.txt");
            strcpy(fcuda_out, "../data/trajNA187_C_RK4_cuda.txt");
        }
    }
    printf("Using %s\n", fname);
    AeroAnalysis *aero = (AeroAnalysis *)malloc(sizeof(AeroAnalysis));
    initializeSTLData(&aero->stlData);

    clock_t t = clock();
    read_STL(aero, fname, false);
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
    printf("Reading from file took %f seconds to execute \n", time_taken);

    const char dirPath[] = "../data";

    // Creating output directory
    struct stat st = {0};

    if (stat(dirPath, &st) == -1) {
        if (mkdir(dirPath, 0700) == -1) {
            perror("Error creating directory");
        }
    }

    FILE *outputFile = fopen(freference_out, "w");
    if (!outputFile) {
        perror("Error opening output file. Unable to open outputfile");
    }


    t = clock();

    run_traj(aero, outputFile, false);

    t = clock() - t;
    time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

    printf("CPU reference took %f seconds to execute \n", time_taken);

    //start CUDA
    AeroAnalysis* aero2 = (AeroAnalysis *)malloc(sizeof(AeroAnalysis));
    initializeSTLData(&aero2->stlData);

    t = clock();
    read_STL(aero2, fname, false);
    t = clock() - t;
    time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
    printf("Reading from file took %f seconds to execute \n", time_taken);

    FILE *cuda_outputFile = fopen(fcuda_out, "w");
    if (!cuda_outputFile) {
        perror("Error opening output file. Unable to open outputfile");
    }

    t = clock();

    run_traj(aero2, cuda_outputFile, true);

    t = clock() - t;
    time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

    printf("Cuda took %f seconds to execute \n", time_taken);

    FILE *file1 = fopen(freference_out, "r");
    FILE *file2 = fopen(fcuda_out, "r");

    if (file1 == NULL || file2 == NULL) {
        perror("Error opening files");
        return -1;
    }

    // compare cuda output with reference
    compare_outputs(file1, file2);


    fclose(file1);
    fclose(file2);
    fclose(cuda_outputFile);
    fclose(outputFile);
    free(aero);
    return 0;
}
