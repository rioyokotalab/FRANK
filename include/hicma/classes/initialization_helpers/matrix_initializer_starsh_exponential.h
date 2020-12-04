#ifndef MATRIX_INITIALIZER_STARSH_EXPONENTIAL_H
#define MATRIX_INITIALIZER_STARSH_EXPONENTIAL_H

#include "hicma/classes/dense.h"
#include "hicma/classes/initialization_helpers/matrix_initializer.h"

#include <starsh.h>
#include <starsh-randtlr.h>
#include <starsh-electrodynamics.h>
#include <starsh-spatial.h>

#include <cstdint>

namespace hicma
{

class ClusterTree;
class IndexRange;

class MatrixInitializerStarshExponential : public MatrixInitializer {
  int64_t N;
  double beta, nu, noise, sigma;
  int ndim;
  STARSH_kernel *s_kernel;
  void *starsh_data;
  STARSH_int * starsh_index;

  public:
  MatrixInitializerStarshExponential() = delete;

  MatrixInitializerStarshExponential(
    int64_t N, double beta, double nu, double noise, double sigma, int ndim,
    double admis, int64_t rank, int basis_type, int admis_type
  );

  ~MatrixInitializerStarshExponential();

  // Utility methods
  void fill_dense_representation(
    Dense& A, const ClusterTree& node
  ) const override;

  void fill_dense_representation(
    Dense& A, const IndexRange& row_range, const IndexRange& col_range
  ) const override;

  Dense get_dense_representation(const ClusterTree& node) const override;

  std::vector<std::vector<double>> get_coords_range(const IndexRange& range) const override;
};
}

#endif /* MATRIX_INITIALIZER_STARSH_EXPONENTIAL_H */
