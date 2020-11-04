#include <starsh.h>
#include <starsh-randtlr.h>
#include <starsh-electrodynamics.h>
#include <starsh-spatial.h>

#include <vector>


namespace starsh {
  STARSH_kernel *kernel;
  void *starsh_data;
  std::vector<STARSH_int> starsh_index;
  void matern_kernel_prepare(int64_t N, double beta, double nu, double noise,
                              double sigma, double wave_k, int64_t add_diag) {
    // ./testing_dpotrf -N 27000 -t 2700 -e 1e-8 -u 200 -j 27000 -v -c 19 -G 200 -U 200 -D 2 -z 30 -Z 10 -Y 1
    int ndim = 3;
    int info = 0;
    int ret = 0;
    enum STARSH_PARTICLES_PLACEMENT place = STARSH_PARTICLES_UNIFORM;
    kernel = starsh_ssdata_block_exp_kernel_3d;
    info = starsh_ssdata_generate((STARSH_ssdata **)&starsh_data, N, ndim,
                                  beta, nu, noise,
                                  place, sigma);
    for (int j = 0; j < N; ++j) {
      starsh_index.push_back(j);
    }
  }
  void matern_kernel_fill(double* A, uint64_t A_rows, uint64_t A_cols, uint64_t A_stride,
                          const std::vector<std::vector<double>>& x,
                          int64_t row_start, int64_t col_start) {
    kernel(A_cols, A_rows, starsh_index.data(), starsh_index.data(), starsh_data, starsh_data,
            A, A_cols);
  }
}

int main() {

}