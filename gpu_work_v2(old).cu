#include <iostream>
#include <algorithm>
#include <numeric>
#include <sys/time.h>
#include "utils.h"

#define THS_PER_BLOCK 256

template<typename T>
void swap(T *x, T *y){
    T aux = *x;
    *x = *y;
    *y = aux;
}

template<typename T>
void print_array(T *v, int size){
    std::cout << '[' << std::endl;
    for (int i = 0; i < size; ++i){
        std::cout << v[i] << ',';
    }
    
    std::cout << ']' << std::endl;
}

template<typename T>
T* copy_array(T *arr, int size){
    T* new_array = new T[size];

    for (int i = 0; i < size; ++i){
        new_array[i] = arr[i];
    }

    return new_array;
}

int* range(int size){
    int *arr = new int[size];

    for (int i = 0; i < size; ++i){
        arr[i] = i;
    }
    
    return arr;
}

int find_first(int i, int size, double *arr, int group){
    for (int j = i + 1; j < size; ++j){
        if (group == 0){
            if (arr[j] > 0.5){
                return j;
            }
        }
        else{
            if (arr[j] < 0.5){
                return j;
            }
        }
    }

    return -1;
}

int * rearrange(int step_size, int *init_map, int size, double *arr){
    double *arr_ = copy_array(arr, size);
    int *new_arr = copy_array(init_map, size);

    for (int i = 0; i < size; ++i){
        if (i / step_size % 2){
            if (arr_[i] > 0.5){
                int j = find_first(i, size, arr_, 1);
                if (j == -1)
                    continue;

                swap(&arr_[i], &arr_[j]);
                swap(&new_arr[i], &new_arr[j]);
            }
        }
        else{
            if (arr_[i] < 0.5){
                int j = find_first(i, size, arr_, 0);

                if (j == -1)
                    continue;

                swap(&arr_[i], &arr_[j]);
                swap(&new_arr[i], &new_arr[j]);
            }
        }
    }
    
    return new_arr;
}

template <typename T>
int * argsort(T* arr, int size){
    int *indices = new int[size];
    std::iota(indices, indices + size, 0);

    std::sort(indices, indices + size,
       [&arr](size_t i1, size_t i2) {return arr[i1] < arr[i2];});

    return indices;
}

// TODO: Não esquecer de liberar recursos alocados (procurar onde foi dado "new")

/**
 * - Números <= 0.5 demoram mais para "trocar de lado" quando sao proximos de 0
 * - Números > 0.5 demoram mais para "trocar de lado" quando sao proximos de 1
 * IDEIAS:
 *      - Alinhar em uma warp numeros mais proximos de 0 ou 1 dependendo do seu "grupo"
 *      - Alinhar em uma warp números que estão próximos um dos outros
        - Pra fazer talvez as duas ideias seja preciso ordenar o array primeiro, antes do rearrange
            - Aí depois, desordenar ou conseguir extrair do ordenado um array map
 */
__global__
void gpu_work_v2(double *arr, int *map)
{
        const int id = blockDim.x * blockIdx.x + threadIdx.x;

        if (id >= ARR_SIZE) return;

        // if (id < 1000)
        //     printf("arr[%d] before = %.5f\n", id, arr[id]);
        for(int i = 0; i < GPU_WORK_ITERATIONS; ++i) {
                if (arr[map[id]] <= 0.5) {
                        arr[map[id]] = laborious_func_le_half(arr[map[id]]);
                } else {
                        arr[map[id]] = laborious_func_gt_half(arr[map[id]]);
                }
        }
        // if (id < 1000)
        //     printf("arr[%d] after = %.5f\n", id, arr[id]);
}

// Launch the work on arr and return it at results;
void launch_gpu_work_v2(double *arr, double **results)
{
        struct timeval start, end;
        int step_size = 32;
        double *d_arr;
        gettimeofday(&start, NULL);
        int *init_map = argsort(arr, ARR_SIZE);
        int *map = rearrange(step_size, init_map, ARR_SIZE, arr), *d_map;

        gettimeofday(&end, NULL);
        double elapsed_v1 = (end.tv_sec - start.tv_sec) +
                     (end.tv_usec - start.tv_usec) / 1000000.0;

        cudaAssert(cudaMalloc(&d_arr, ARR_SIZE * sizeof(double)));
        cudaAssert(cudaMemcpy(d_arr, arr, ARR_SIZE * sizeof(double),
                              cudaMemcpyHostToDevice));
        cudaAssert(cudaMalloc(&d_map, ARR_SIZE * sizeof(int)));
        cudaAssert(cudaMemcpy(d_map, map, ARR_SIZE * sizeof(int),
                              cudaMemcpyHostToDevice));

        printf("elapsed to get map: %.4fs\n", elapsed_v1);
        gpu_work_v2<<<DIV_CEIL_INT(ARR_SIZE, THS_PER_BLOCK), THS_PER_BLOCK>>>(d_arr, d_map);
        cudaAssert(cudaDeviceSynchronize());

        cudaAssert(cudaMemcpy(*results, d_arr, ARR_SIZE * sizeof(double),
                              cudaMemcpyDeviceToHost));
        cudaAssert(cudaFree(d_arr));
        cudaAssert(cudaFree(d_map));
}
