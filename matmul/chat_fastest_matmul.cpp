#include <iostream>
#include <random>
#include <cstdlib>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <immintrin.h> // AVX2 / FMA intrinsics
#include <omp.h>       // OpenMP

// --- Micro-Kernel & Tile Parameters ---
#define MC 64  // Cache Tile Height
#define KC 128 // Cache Tile Depth
#define NC 64  // Cache Tile Width
#define MR 8   // Micro-kernel height (AVX2 register stride)
#define NR 8   // Micro-kernel width (AVX2 register stride)

// Generates a 32-byte aligned contiguous block of memory for an m x n matrix
float* gen_matrix(int m, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    
    size_t total_bytes = sizeof(float) * m * n;
    
    // 32-byte alignment is required for AVX _mm256_load_ps instructions
    float *matrix = (float*) std::aligned_alloc(32, total_bytes);
    if (!matrix) {
        throw std::bad_alloc();
    }
    
    for (int i = 0; i < m * n; i++) {
        matrix[i] = dis(gen);
    }
    return matrix;
}

// --- 8x8 AVX2 + FMA Micro-Kernel ---
inline void gemm_micro_kernel_avx_8x8(int K, const float *A, int lda,
                                       const float *B, int ldb,
                                       float *C, int ldc) {
    // Load 8 intermediate C vectors into 256-bit AVX registers
    __m256 c0 = _mm256_load_ps(C + 0 * ldc);
    __m256 c1 = _mm256_load_ps(C + 1 * ldc);
    __m256 c2 = _mm256_load_ps(C + 2 * ldc);
    __m256 c3 = _mm256_load_ps(C + 3 * ldc);
    __m256 c4 = _mm256_load_ps(C + 4 * ldc);
    __m256 c5 = _mm256_load_ps(C + 5 * ldc);
    __m256 c6 = _mm256_load_ps(C + 6 * ldc);
    __m256 c7 = _mm256_load_ps(C + 7 * ldc);

    for (int k = 0; k < K; ++k) {
        __m256 b_vec = _mm256_load_ps(B + k * ldb);

        c0 = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * lda + k]), b_vec, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * lda + k]), b_vec, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * lda + k]), b_vec, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * lda + k]), b_vec, c3);
        c4 = _mm256_fmadd_ps(_mm256_set1_ps(A[4 * lda + k]), b_vec, c4);
        c5 = _mm256_fmadd_ps(_mm256_set1_ps(A[5 * lda + k]), b_vec, c5);
        c6 = _mm256_fmadd_ps(_mm256_set1_ps(A[6 * lda + k]), b_vec, c6);
        c7 = _mm256_fmadd_ps(_mm256_set1_ps(A[7 * lda + k]), b_vec, c7);
    }

    // Store updated values back to C memory
    _mm256_store_ps(C + 0 * ldc, c0);
    _mm256_store_ps(C + 1 * ldc, c1);
    _mm256_store_ps(C + 2 * ldc, c2);
    _mm256_store_ps(C + 3 * ldc, c3);
    _mm256_store_ps(C + 4 * ldc, c4);
    _mm256_store_ps(C + 5 * ldc, c5);
    _mm256_store_ps(C + 6 * ldc, c6);
    _mm256_store_ps(C + 7 * ldc, c7);
}

// --- Multithreaded, Cache-Tiled AVX2 GEMM ---
void matmul_cpu_fastest(const float *A, const float *B, float *C, int M, int N, int K) {
    // Parallel zero initialization
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M * N; ++i) {
        C[i] = 0.0f;
    }

    #pragma omp parallel for schedule(static)
    for (int jc = 0; jc < N; jc += NC) {
        int nc = std::min(N - jc, NC);

        for (int ic = 0; ic < M; ic += MC) {
            int mc = std::min(M - ic, MC);

            for (int kc = 0; kc < K; kc += KC) {
                int k_tile_depth = std::min(K - kc, KC);

                for (int ir = 0; ir < mc; ir += MR) {
                    for (int jr = 0; jr < nc; jr += NR) {
                        const float *A_block = A + (ic + ir) * K + kc;
                        const float *B_block = B + kc * N + (jc + jr);
                        float *C_block = C + (ic + ir) * N + (jc + jr);

                        gemm_micro_kernel_avx_8x8(k_tile_depth, A_block, K, B_block, N, C_block, N);
                    }
                }
            }
        }
    }
}

int main() {
    int M_a = 2048, N_a = 2048;
    int M_b = 2048, N_b = 2048;

    if (N_a != M_b) {
        throw std::invalid_argument("Matrix inner dimensions must match!");
    }

    std::cout << "Allocating and generating 2048x2048 matrices..." << std::endl;
    float *h_A = gen_matrix(M_a, N_a);
    float *h_B = gen_matrix(M_b, N_b);
    
    size_t bytes_ans = sizeof(float) * M_a * N_b;
    float *h_ans = (float*) std::aligned_alloc(32, bytes_ans);

    if (!h_ans) {
        std::cerr << "Failed to allocate memory for result matrix." << std::endl;
        free(h_A);
        free(h_B);
        return 1;
    }

    std::cout << "Starting fast AVX2 + OpenMP matrix multiplication..." << std::endl;
    
    // Warm-up run (helps stabilize thread pool and CPU clock speeds)
    matmul_cpu_fastest(h_A, h_B, h_ans, M_a, N_b, M_b);

    // Timed benchmark run
    auto start = std::chrono::high_resolution_clock::now();

    matmul_cpu_fastest(h_A, h_B, h_ans, M_a, N_b, M_b);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Total Floating Point Operations for GEMM: 2 * M * N * K
    double gflops = (2.0 * M_a * N_b * M_b) / (duration.count() * 1e9);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Completed in: " << duration.count() << " seconds" << std::endl;
    std::cout << "Throughput:   " << gflops << " GFLOPS" << std::endl;
    std::cout << "========================================" << std::endl;

    free(h_A);
    free(h_B);
    free(h_ans);

    return 0;
}
