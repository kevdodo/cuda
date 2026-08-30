#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

// Generates a contiguous block of memory a m by n matrix 
float* gen_matrix(int m, int n){
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, 100);
  
  float *lol = (float*) malloc(sizeof(float) * m * n);
  for (int i=0; i< m*n; i++){
    lol[i] = dis(gen);
  }
  return lol;
}

__device__ int get_idx(const int row, const int col, const int num_cols){
  return row * num_cols + col;
}

__global__
void matmul(float *d_A, float *d_B, float*ans, int m_a, int n_b, int k){
  /*
   * Computes the matrix multiplication of the two matricies and 
   * stores the result in ans
   *
   * m_a -> number of rows in a
   * n_b -> number of cols in b
   * k -> what we gotta iterate over for that dot product
   *
   */
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  // Edge guard so that you don't compute anything out of bounds
  if (row >= m_a || col >= n_b){
    return;
  }
  float acc = 0.0f;
  // compute the dot product loop through num cols in A or rows in b 
  for (int i=0; i < k; i++){
    int idx_a = get_idx(row, i, k);
    int idx_b = get_idx(i, col, n_b);
    acc += d_A[idx_a] * d_B[idx_b];
  }
  int final_idx =get_idx(row, col, n_b); 
  ans[final_idx] = acc;
}
bool verify_results(const float *d_ans, const float *h_ans, int size) {
    for (int i = 0; i < size; ++i) {
        float diff =std::abs(d_ans[i] - h_ans[i]); 
        if (diff > 1e-4 * std::max(1.0f, std::abs(h_ans[i]))) {
            std::cout << "Mismatch at index " << i << ": GPU=" << d_ans[i] << ", CPU=" << h_ans[i] << std::endl;
            return false;
        }
    }
    return true;
}
void matmul_cpu_reference(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[r * K + k] * B[k * N + c];
            }
            C[r * N + c] = sum;
        }
    }
}
int main(){
  int M_a = 3;
  int N_a = 3;

  int M_b = 3;
  int N_b = 3;

  // make sure that number of cols = num of rows in b
  if (N_a != M_b) {
    throw std::invalid_argument(
          "Matrix A columns (" + std::to_string(N_a) + 
          ") must equal Matrix B rows (" + std::to_string(M_b) + ")"
      );
  }

  int bytes_a = N_a * M_a * sizeof(float); 
  int bytes_b = N_b * M_b * sizeof(float); 
  int bytes_ans = N_b * M_a * sizeof(float); 

  float *h_A = gen_matrix(M_a, N_a);
  float *h_B = gen_matrix(M_b, N_b);
  // 2. Allocate device memory directly in GPU VRAM
  float *d_A, *d_B, *ans;
  cudaMalloc(&d_A, bytes_a);
  cudaMalloc(&d_B, bytes_b);
  cudaMalloc(&ans, bytes_ans);

  cudaMemcpy(d_A, h_A, bytes_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytes_b, cudaMemcpyHostToDevice);

  // 256 threads per block size 16 x 16;
  // each block is 16 tall, 16 wide
  int block_size = 16;

  // integer ceil trick, gonna have 
  int num_block_rows = (M_a + block_size -1) / block_size;
  int num_block_cols = (N_b + block_size -1) / block_size; 
  
  dim3 threadsPerBlock(block_size, block_size);
  dim3 numBlocks(num_block_cols, num_block_rows);
  matmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, ans, M_a, N_b, N_a);
  cudaDeviceSynchronize();

  float *h_ans = (float *) malloc(bytes_ans);
  float *d_ans = (float *) malloc(bytes_ans);
  cudaMemcpy(d_ans, ans, bytes_ans, cudaMemcpyDeviceToHost);
  
  matmul_cpu_reference(h_A, h_B, h_ans, M_a, N_b,M_b);
  if (verify_results(d_ans, h_ans, M_a * N_b)){
    std::cout << "yippeee" << std::endl;
  }
  
  cudaFree(d_A);
  cudaFree(d_B);
  free(h_A);
  free(h_B);
  return 0;
}
