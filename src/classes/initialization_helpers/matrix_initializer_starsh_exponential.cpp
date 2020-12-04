#include "hicma/classes/initialization_helpers/matrix_initializer_starsh_exponential.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/initialization_helpers/cluster_tree.h"
#include "hicma/classes/initialization_helpers/index_range.h"
#include "hicma/util.h"

#include <cstdint>
#include <cmath>
#include <utility>
#include <iostream>

namespace hicma
{
  MatrixInitializerStarshExponential::MatrixInitializerStarshExponential(
    int64_t N, double beta, double nu, double noise, double sigma, int ndim,
    double admis, int64_t rank, BasisType basis_type, int admis_type
  ) : MatrixInitializer(admis, rank, basis_type, admis_type),
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
  }

  void MatrixInitializerStarshExponential::fill_dense_representation(
    Dense& A,
    const ClusterTree& node
  ) const {
    fill_dense_representation(A, node.rows, node.cols);
  }

  void MatrixInitializerStarshExponential::fill_dense_representation(
    Dense& A, const IndexRange& row_range, const IndexRange& col_range
  ) const {
    s_kernel(A.dim[0], A.dim[1], starsh_index + row_range.start,
           starsh_index + col_range.start, starsh_data,
           starsh_data, &A, A.stride);
  }

  Dense MatrixInitializerStarshExponential::get_dense_representation(
    const ClusterTree& node
  ) const {
    Dense representation(node.rows.n, node.cols.n);
    fill_dense_representation(representation, node.rows, node.cols);
    return representation;
  }

  std::vector<std::vector<double>> MatrixInitializerStarshExponential::get_coords_range(const IndexRange& range) const {
    STARSH_ssdata *data = (STARSH_ssdata*)starsh_data;
    double *range_data;
    double *coords_range_ptr[ndim];

    range_data = data->particles.point + range.start;
    // store co-ordinates offsets for 0th dimension.
    coords_range_ptr[0] = range_data;
    // store co-ordinates offsets for the rest of the dimensions.
    for (int i = 1; i < ndim; ++i) {
      // offset along the row data that corresponds to the points
      // in the ith dimension.
      coords_range_ptr[i] = range_data + i * N;
    }
    //Convert pointer to vector
    std::vector<std::vector<double>> coords_range;
    for(int d=0; d<ndim; d++) {
      coords_range.push_back(std::vector<double>(coords_range_ptr[d], coords_range_ptr[d]+range.n));
    }
    return coords_range;
  }

  MatrixInitializerStarshExponential::~MatrixInitializerStarshExponential()  {
    starsh_ssdata_free((STARSH_ssdata*) starsh_data);
    free(starsh_index);
  }

}
