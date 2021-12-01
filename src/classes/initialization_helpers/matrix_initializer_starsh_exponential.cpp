#include "hicma/classes/initialization_helpers/matrix_initializer_starsh_exponential.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/initialization_helpers/cluster_tree.h"
#include "hicma/classes/initialization_helpers/index_range.h"
#include "hicma/util.h"

#include <cstdint>
#include <cmath>
#include <utility>
#include <vector>

namespace hicma
{
  MatrixInitializerStarshExponential::MatrixInitializerStarshExponential(
    int64_t N, double beta, double nu, double noise, double sigma, int ndim,
    double admis, int64_t rank, int admis_type
  ) : MatrixInitializer(admis, rank, std::vector<std::vector<double>>(), admis_type),
    N(N), beta(beta), nu(nu), noise(noise), sigma(sigma), ndim(ndim) {
    enum STARSH_PARTICLES_PLACEMENT place = STARSH_PARTICLES_UNIFORM;
    if (ndim == 2) {
      s_kernel = starsh_ssdata_block_exp_kernel_2d;
    } else if (ndim == 3) {
      s_kernel = starsh_ssdata_block_exp_kernel_3d;
    }
    starsh_ssdata_generate((STARSH_ssdata **)&starsh_data, N, ndim, beta,
                           nu, noise, place, sigma);
    starsh_index = (STARSH_int*)malloc(sizeof(STARSH_int) * N);
    for (int j = 0; j < N; ++j) {
      starsh_index[j] = j;
    }
    
    //Convert StarsH geometry to 2D vector
    STARSH_ssdata *data = (STARSH_ssdata*)starsh_data;
    double *range_data;
    double *points_range_ptr[ndim];
    range_data = data->particles.point;
    // store co-ordinates offsets for 0th dimension.
    points_range_ptr[0] = range_data;
    // store co-ordinates offsets for the rest of the dimensions.
    for (int i = 1; i < ndim; ++i) {
      // offset along the row data that corresponds to the points
      // in the ith dimension.
      points_range_ptr[i] = range_data + i * N;
    }
    //Convert pointer to vector
    for(int d=0; d<ndim; d++) {
      params.push_back(std::vector<double>(points_range_ptr[d], points_range_ptr[d] + N));
    }
  }

  MatrixInitializerStarshExponential::~MatrixInitializerStarshExponential()  {
    starsh_ssdata_free((STARSH_ssdata*) starsh_data);
    free(starsh_index);
  }

  void MatrixInitializerStarshExponential::fill_dense_representation(
    Dense& A, const IndexRange& row_range, const IndexRange& col_range
  ) const {
    s_kernel(A.dim[0], A.dim[1], starsh_index + row_range.start,
           starsh_index + col_range.start, starsh_data,
           starsh_data, &A, A.stride);
  }

}
