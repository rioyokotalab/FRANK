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
    std::string filename, MatrixLayout ordering, double admis, int64_t rank,
    const std::vector<std::vector<double>>& params, int admis_type
  );

  void fill_dense_representation(
    Dense& A, const IndexRange& row_range, const IndexRange& col_range
  ) const override;

};

} // namespace hicma

#endif // hicma_classes_initialization_helpers_matrix_initializer_file_h
