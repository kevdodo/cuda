#include <cmath>
#include <driver_types.h>
#include <random>
#include <cuda_runtime_api.h>
#include <iostream>

#ifndef BUILD_STANDALONE
#include <torch/extension.h>
#endif

const int TILE_DIM = 256;
__global__ void softmax(float *arr, float *out, int size){
  /*
   * Softmax formula:
   * softmax_i(x) =  e ^(x_i-C) / sum(e^x_j-C)
   *
   * Where C is the max of the softmax array
   *
   * */

  // Get the max
  float thread_max = -INFINITY;
  float thread_denom = 0.0f;

  int idx = threadIdx.x;

  // Pt. 1 get the max and denoms
  if (idx < size) {
    thread_max = arr[idx];
    thread_denom = 1.0f;
    for (int i=idx + TILE_DIM; i < size; i+=TILE_DIM){
      float val = arr[i];
      if (val > thread_max){
        thread_denom = thread_denom * expf(thread_max - val) + 1.0f;
        thread_max = val;
      } else{
        thread_denom += expf(val - thread_max);
      }
    }
  }
  

  __shared__ float maxes[TILE_DIM];
  __shared__ float denoms[TILE_DIM];
  
  // Pt 2. fill up the shared mem
  
  maxes[idx] = thread_max;
  denoms[idx] = thread_denom;

  __syncthreads();
  
  // Pt 3 merge part
  int stride = TILE_DIM / 2;
  while (stride > 0){
    //printf("thread_idx %d beforemaxes: %f, %f, %f, %f, %f \n\n", idx, maxes[0], maxes[1], maxes[2], maxes[3], maxes[4]);

    // idx + stride is not enough, as that allows the end guy of the tree to work
    //
    // We always want half workers (0 -> stride-1) to be executed
    if (idx < stride){
      float new_max = fmaxf(maxes[idx], maxes[idx + stride]);
      float dnew = denoms[idx] * expf(maxes[idx] - new_max) + denoms[idx + stride] * expf(maxes[idx+stride] - new_max);

      maxes[idx] = new_max;
      denoms[idx] = dnew;
    }
    __syncthreads();
    stride /=2;

    //printf("thread_idx %d  maxes: %f, %f, %f, %f, %f \n\n", idx, maxes[0], maxes[1], maxes[2], maxes[3], maxes[4]);
  }
  
  // now that we have global max and d...
  for (int i=idx; i < size; i += TILE_DIM){
    out[i] = expf(arr[i] - maxes[0]) / denoms[0];
  }
} 


#ifndef BUILD_STANDALONE
torch::Tensor run_softmax(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "input must be on cuda");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
  
  int size = input.numel();
  auto output = torch::empty_like(input);
  softmax<<<1, TILE_DIM>>>(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      size
  );
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax", &run_softmax, "Online softmax kernel");
}

#endif

#ifdef BUILD_STANDALONE
void print_first_five(float *arr){
  std::cout << "printing first five :D" << std::endl;
  for (int i=0; i < 5; i++){
    std::cout << arr[i] << " ";
  }
  std::cout << std::endl;
}

int main(){
  int N = 10<<20;

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

  softmax<<<1, TILE_DIM>>>(arr_d, output, N);
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
#endif
