#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>

using namespace std;

template<typename T>
void print_array(T *v, int size){
    for (int i = 0; i < size; ++i){
        std::cout << v[i];
        if (i < size - 1)
             std::cout << ',';
    }
    std::cout << std::endl;
}

void randomly_fill_array(double *v, int size)
{
        srand(1373); // arbitrary initialization
        for (int i = 0; i < size; ++i)
                v[i] = (double)rand() / RAND_MAX;
}

template <typename T>
int * argsort(T* arr, int size){
    int *indices = new int[size];
    iota(indices, indices + size, 0);

    std::sort(indices, indices + size,
       [&arr](size_t i1, size_t i2) {return arr[i1] < arr[i2];});

    return indices;
}

int main(){
    int n = 20;
    double *v = new double[n];

    randomly_fill_array(v,n);

    print_array(v, n);
    int * indices = argsort<double>(v, n);

    print_array(indices, n);
    return 0;
}
