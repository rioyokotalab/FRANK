#include <starsh.h>
#include <starsh-randtlr.h>
#include <starsh-electrodynamics.h>
#include <starsh-spatial.h>

#include <iostream>
#include <vector>

int main(int argc, char *argv[])
{
  int ndim = 3;
  int info = 0;
  int ret = 0;
  double beta = 0.1;
  double nu = 0.5;//in matern, nu=0.5 exp (half smooth), nu=inf sqexp (inifinetly smooth)
  //nu is only used in matern kernel
  //double noise = 1.e-2; // did not work for 10M in Lorapo
  double noise = 1.e-1;
  double sigma = 1.0;
  double wave_k = 1.0;
  int add_diag = 27000;

  STARSH_kernel *kernel;
  void *starsh_data;
  std::vector<STARSH_int> starsh_index;
  
  enum STARSH_PARTICLES_PLACEMENT place = STARSH_PARTICLES_UNIFORM;

  int N = 8192;
  int A_cols = 1024;
  int A_rows = 1024;
  double A[A_rows * A_cols];
      
  kernel = starsh_ssdata_block_exp_kernel_3d;
  info = starsh_ssdata_generate((STARSH_ssdata **)&starsh_data, N, ndim, beta, nu, noise, place, sigma);
  for (int j = 0; j < N; ++j) {
    starsh_index.push_back(j);
  }

  std::cout << "filling kernel r: " << A_rows << " c: " << A_cols << std::endl;
  kernel(A_cols, A_rows, starsh_index.data(), starsh_index.data(), starsh_data, starsh_data,
         A, A_cols);


  return 0;
}
