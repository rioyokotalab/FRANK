#ifndef hicma_classes_initialization_helpers_matrix_initializer_h
#define hicma_classes_initialization_helpers_matrix_initializer_h

#include "hicma/definitions.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/initialization_helpers/index_range.h"

#include <cstdint>
#include <tuple>
#include <vector>


namespace hicma
{

class ClusterTree;

class MatrixInitializer {
 private:
  double admis;
  int64_t rank;
  int admis_type;
 protected:
  const std::vector<std::vector<double>>& coords;

  void find_admissible_blocks(const ClusterTree& node);

 public:
  // Special member functions
  MatrixInitializer() = delete;

  ~MatrixInitializer() = default;

  MatrixInitializer(const MatrixInitializer& A) = delete;

  MatrixInitializer& operator=(const MatrixInitializer& A) = delete;

  MatrixInitializer(MatrixInitializer&& A) = delete;

  MatrixInitializer& operator=(MatrixInitializer&& A) = default;

  // Additional constructors
  MatrixInitializer(
    double admis, int64_t rank,
    int admis_type=POSITION_BASED_ADMIS,
    const std::vector<std::vector<double>>& coords=std::vector<std::vector<double>>()
  );

  // Utility methods
  virtual void fill_dense_representation(
    Dense& A, const ClusterTree& node
  ) const = 0;

  virtual void fill_dense_representation(
    Dense& A, const IndexRange& row_range, const IndexRange& col_range
  ) const = 0;

  virtual Dense get_dense_representation(const ClusterTree& node) const = 0;

  LowRank get_compressed_representation(const ClusterTree& node);

  bool is_admissible(const ClusterTree& node) const;

  virtual std::vector<std::vector<double>> get_coords_range(const IndexRange& range) const;

};

} // namespace hicma


#endif // hicma_classes_initialization_helpers_matrix_initializer_h
