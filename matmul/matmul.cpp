#include <iostream>
#include <vector>
#include <random>
#include <stdexcept>
#include <string>

using Matrix = std::vector<std::vector<float>>; 

Matrix gen_matrix(int m, int n){
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, 100);
  
  Matrix matrix(m, std::vector<float>(n));
 
  for (int i=0; i < m; i++){
    for (int j=0; j < n; j++){
      matrix[i][j] = dis(gen);
    }
  }
  return matrix;
}

float get_value(const Matrix &a, const Matrix &b, int row, int col){
  float sum = 0.0f;
  for (int k=0; k < a[0].size(); k++){
    sum += a[row][k] * b[k][col];
  }
  return sum;
}

Matrix matmul(const Matrix &a, const Matrix &b){
  
  int rows_a = a.size();
  int cols_a = a[0].size();
  int rows_b = b.size();
  int cols_b = b[0].size();
  if (cols_a != rows_b) {
      throw std::invalid_argument(
          "Matrix A columns (" + std::to_string(cols_a) + 
          ") must equal Matrix B rows (" + std::to_string(rows_b) + ")"
      );
  }
  Matrix ans(rows_a, std::vector<float>(cols_b));
  for (int row=0; row < rows_a; row++){
    for (int col=0; col < cols_b; col++){
      ans[row][col] = get_value(a, b, row, col);   
    }
  } 
  return ans;
}

void print_matrix(const Matrix &a){
  for (int i=0; i < a.size(); i++){
    for (int j=0; j < a[0].size(); j++){
      std::cout << a[i][j] << ' ';
    }
    std::cout << std::endl;
  }
}

void test_mat_mul(){
  Matrix a = {
      {1, 0, 0},
      {0, 1, 0},
      {0, 0, 1}
  };
  Matrix b = {
      {1, 0, 0},
      {0, 2, 0},
      {0, 0, 3}
  };
  
  Matrix ans = matmul(a, b);
  print_matrix(ans);

}

int main(){
  int n = 3;
  int m = 3;
  test_mat_mul();
  
  return 0;
}
