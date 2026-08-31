/*
 *
 * This just for testing how fast matmul is for cpu
 *
 */
#include <random>
#include <stdexcept>
#include <iostream>
#include <omp.h>

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


void matmul_cpu_reference(const float *A, const float *B, float *C, int M, int N, int K) {
  /*
   * Does accesses C sequentially for better memory accesses and caches a_val
   */

  for (int i=0; i < M*N; i++){
    C[i] = 0.0f;
  }

  for (int r = 0; r < M; ++r) {
    for (int k = 0; k < K; ++k) {
      float a_val = A[r * K + k];
        for (int c = 0; c < N; ++c) {
            C[r * N + c] += a_val + B[k * N + c];
        }
      }
    }
}


void matmul_cpu_tiled_omp(const float *A, const float *B, float *C, int M, int N, int K) {
    // 1. Zero out result matrix C
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0.0f;
        }
    }

    // Tile size chosen to keep sub-matrices inside L1/L2 cache (e.g., 64x64 floats = 16KB)
    constexpr int BLOCK_SIZE = 64;

    // 2. Parallelize outer tile loops using OpenMP
    #pragma omp parallel for collapse(2) schedule(static)
    for (int sj = 0; sj < N; sj += BLOCK_SIZE) {
        for (int si = 0; si < M; si += BLOCK_SIZE) {
            for (int sk = 0; sk < K; sk += BLOCK_SIZE) {

                // Micro-kernel operating on the tile blocks
                for (int i = si; i < std::min(si + BLOCK_SIZE, M); ++i) {
                    for (int k = sk; k < std::min(sk + BLOCK_SIZE, K); ++k) {
                        float a_val = A[i * K + k];
                        
                        // Tell the compiler to auto-vectorize the innermost contiguous loop
                        #pragma omp simd
                        for (int j = sj; j < std::min(sj + BLOCK_SIZE, N); ++j) {
                            C[i * N + j] += a_val * B[k * N + j];
                        }
                    }
                }

            }
        }
    }
}
int main (){
  int M_a = 2048;
  int N_a = 2048;

  int M_b = 2048;
  int N_b = 2048;

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
  float *h_ans = (float *) malloc(bytes_ans);
  std::cout << "matrices inittialized peforming matmul :D" << std::endl;
  matmul_cpu_tiled_omp(h_A, h_B, h_ans, M_a, N_b,M_b);
  free(h_A);
  free(h_B);
  free(h_ans);
  return 0;
}
