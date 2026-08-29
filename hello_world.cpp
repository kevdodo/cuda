#include <iostream>
#include <math.h>

// standard way to add
//
// void add(int n, float *x, float *y){
//   for (int i=0; i < n; i++){
//       y[i] = x[i] + y[i];
//   }
// }


__global__
void add(int n, float *sum, float *x, float *y){
  for (int i=0; i < n; i++){
      y[i] = x[i] + y[i];
  }
}


int main() {
  int N = 1<<20; // this is 1 million elements

  float *x = new float[N];
  float *y = new float[N];

  for (int i=0; i< N; i++){
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  add(N, x, y);
 
  float max_error = 0.0f;
  for (int i=0; i< N; i++){
    max_error = std::fmax(std::fabs(y[i] - 3.0f), max_error);
  }

  std::cout << "Max error: " << max_error << std::endl;
  std::cout << "hello world" <<std::endl;
  delete [] x;
  delete [] y;
  return 0;
}
