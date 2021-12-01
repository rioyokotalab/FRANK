#ifndef hicma_classes_initialization_helpers_matrix_initializer_starsh_exponential_h
#define hicma_classes_initialization_helpers_matrix_initializer_starsh_exponential_h

#include "hicma/definitions.h"
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
class Dense;
class IndexRange;

class MatrixInitializerStarshExponential : public MatrixInitializer {
 private:
  int64_t N;
  double beta, nu, noise, sigma;
  int ndim;
  STARSH_kernel *s_kernel;
  void *starsh_data;
  STARSH_int * starsh_index;

 public:
  // Special member function
  MatrixInitializerStarshExponential();

  ~MatrixInitializerStarshExponential();

  MatrixInitializerStarshExponential(const MatrixInitializerStarshExponential& A) = delete;

  MatrixInitializerStarshExponential& operator=(const MatrixInitializerStarshExponential& A) = delete;

  MatrixInitializerStarshExponential(MatrixInitializerStarshExponential&& A) = delete;

  MatrixInitializerStarshExponential& operator=(MatrixInitializerStarshExponential&& A) = default;
  
  // Additional constructors
  MatrixInitializerStarshExponential(
    int64_t N, double beta, double nu, double noise, double sigma, int ndim,
    double admis, int64_t rank, int admis_type
  );

  void fill_dense_representation(
    Dense& A, const IndexRange& row_range, const IndexRange& col_range
  ) const override;
  
};

} // namespace hicma

#endif //hicma_classes_initialization_helpers_matrix_initializer_starsh_exponential_h
