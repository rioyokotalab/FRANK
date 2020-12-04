#ifndef hicma_classes_initialization_helpers_matrix_initializer_file_h
#define hicma_classes_initialization_helpers_matrix_initializer_file_h

#include "hicma/definitions.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/initialization_helpers/matrix_initializer.h"

#include <cstdint>


namespace hicma
{

class ClusterTree;
class IndexRange;

class MatrixInitializerFile : public MatrixInitializer {
 private:
  std::string filename;
  MatrixLayout ordering;
 public:
  // Special member functions
  MatrixInitializerFile() = delete;

  ~MatrixInitializerFile() = default;

  MatrixInitializerFile(const MatrixInitializerFile& A) = delete;

  MatrixInitializerFile& operator=(const MatrixInitializerFile& A) = delete;

  MatrixInitializerFile(MatrixInitializerFile&& A) = delete;

  MatrixInitializerFile& operator=(MatrixInitializerFile&& A) = default;

  // Additional constructors
  MatrixInitializerFile(
    std::string filename, MatrixLayout ordering, double admis, int64_t rank, BasisType basis_type,
    int admis_type, const std::vector<std::vector<double>>& coords
  );

  // Utility methods
  void fill_dense_representation(
    Dense& A, const ClusterTree& node
  ) const override;

  void fill_dense_representation(
    Dense& A, const IndexRange& row_range, const IndexRange& col_range
  ) const override;

  Dense get_dense_representation(const ClusterTree& node) const override;

};

} // namespace hicma

#endif // hicma_classes_initialization_helpers_matrix_initializer_file_h
