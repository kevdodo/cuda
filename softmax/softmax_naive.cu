#include <cmath>
#include <driver_types.h>
#include <random>
#include <cuda_runtime_api.h>
#include <iostream>

__global__ void softmax(float *arr, float *out, int size){
  /*
   * Softmax formula:
   * softmax_i(x) =  e ^(x_i-C) / sum(e^x_j-C)
   *
   * Where C is the max of the softmax array
   *
   * */
  

  // Get the max
  float max_arr = -INFINITY;
  for (int i=0; i < size; i++){
    max_arr = max(max_arr, arr[i]);
  }
  // calc denom
  float denom = 0;
  for (int i=0; i < size; i++) {
    denom += expf(arr[i] - max_arr);
  }
  for (int i=0; i < size; i++){
    out[i] = expf(arr[i]-max_arr) / denom;
  }
} 

void print_first_five(float *arr){
  std::cout << "printing first five :D" << std::endl;
  for (int i=0; i < 5; i++){
    std::cout << arr[i] << " ";
  }
  std::cout << std::endl;
}

int main(){
  int N = 5;

  int bytes = sizeof(float) * N;
  float *arr = (float *) malloc(bytes); 
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, 100);

  for (int i = 0; i < N; i++){
    arr[i] = dis(gen);
  }
  std::cout <<" we allocated them" << std::endl;
  print_first_five(arr);
  float *output; float *arr_d; 
  cudaMalloc(&output, bytes);
  cudaMalloc(&arr_d, bytes);

  cudaMemcpy(arr_d, arr, bytes, cudaMemcpyHostToDevice);

  softmax<<<1, 1>>>(arr_d, output, N);
  cudaDeviceSynchronize();
  // std::cout<< "finished calcs" << std::endl;
  cudaMemcpy(arr, output, bytes, cudaMemcpyDeviceToHost);
  print_first_five(arr);
  cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
  free(arr);

  cudaFree(arr_d);
  cudaFree(output);

  return 0;
}
