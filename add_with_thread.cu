#include <iostream>
#include <math.h>

// standard way to add
// void add(int n, float *x, float *y){
//   for (int i=0; i < n; i++){
//       y[i] = x[i] + y[i];
//   }
// }

__global__
void add(int n, float *x, float *y){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i=index; i < n; i += stride){
      y[i] = x[i] + y[i];
  }
}


int main() {
  int N = 1<<20; // this is 1 million elements

  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  for (int i=0; i< N; i++){
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  // Prefetch the x and y arrays to the GPU
  cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0);
  cudaMemPrefetchAsync(y, N*sizeof(float), 0, 0);

  int block_size = 256;
  int num_blocks = (N + block_size - 1) / block_size;

 
  add <<<num_blocks, block_size>>>(N, x, y);
  cudaDeviceSynchronize();

  float max_error = 0.0f;
  for (int i=0; i< N; i++){
    max_error = std::fmax(std::fabs(y[i] - 3.0f), max_error);
  }
  cudaFree(x);
  cudaFree(y);
  std::cout << "Max error: " << max_error << std::endl;
  std::cout << "hello world" <<std::endl;

  return 0;
}
