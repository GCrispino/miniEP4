#include <iostream>
#include <algorithm>
#include <sys/time.h>
#include "utils.h"

#define THS_PER_BLOCK 256

// return range from 0 to size - 1 - [0, size - 1)
int* range(int size){
    int *arr = new int[size];

    for (int i = 0; i < size; ++i){
        arr[i] = i;
    }
    
    return arr;
}

// sort array in ascending order and return indices of sorted elements
template <typename T>
int * argsort(T* arr, int size){
    int *indices = range(size);

    std::sort(indices, indices + size,
       [&arr](size_t i1, size_t i2) { return arr[i1] < arr[i2]; });

    return indices;
}

__global__
void gpu_work_v2(double *arr, int *map)
{
        const int id = blockDim.x * blockIdx.x + threadIdx.x;

        if (id >= ARR_SIZE) return;

        // mapping calculated ids to sorted indices (arr[map[id]])
        for(int i = 0; i < GPU_WORK_ITERATIONS; ++i) {
                if (arr[map[id]] <= 0.5) {
                        arr[map[id]] = laborious_func_le_half(arr[map[id]]);
                } else {
                        arr[map[id]] = laborious_func_gt_half(arr[map[id]]);
                }
        }
}

// Launch the work on arr and return it at results;
void launch_gpu_work_v2(double *arr, double **results)
{
        struct timeval start, end;
        double *d_arr;
        gettimeofday(&start, NULL);
        int *map = argsort(arr, ARR_SIZE), *d_map;

        gettimeofday(&end, NULL);
        double elapsed_v1 = (end.tv_sec - start.tv_sec) +
                     (end.tv_usec - start.tv_usec) / 1000000.0;

        cudaAssert(cudaMalloc(&d_arr, ARR_SIZE * sizeof(double)));
        cudaAssert(cudaMemcpy(d_arr, arr, ARR_SIZE * sizeof(double),
                              cudaMemcpyHostToDevice));
        cudaAssert(cudaMalloc(&d_map, ARR_SIZE * sizeof(int)));
        cudaAssert(cudaMemcpy(d_map, map, ARR_SIZE * sizeof(int),
                              cudaMemcpyHostToDevice));

        printf("\nelapsed to get map: %.4fs\n", elapsed_v1);
        gpu_work_v2<<<DIV_CEIL_INT(ARR_SIZE, THS_PER_BLOCK), THS_PER_BLOCK>>>(d_arr, d_map);
        cudaAssert(cudaDeviceSynchronize());

        cudaAssert(cudaMemcpy(*results, d_arr, ARR_SIZE * sizeof(double),
                              cudaMemcpyDeviceToHost));

        delete [] map;
        cudaAssert(cudaFree(d_arr));
        cudaAssert(cudaFree(d_map));
}


