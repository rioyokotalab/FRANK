#ifndef hicma_classes_initialization_helpers_matrix_initializer_file_h
#define hicma_classes_initialization_helpers_matrix_initializer_file_h

#include "hicma/definitions.h"
#include "hicma/classes/dense.h"
#include "hicma/classes/initialization_helpers/matrix_initializer.h"

#include <cstdint>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class ClusterTree;
class IndexRange;

/**
 * @brief `MatrixInitializer` specialization that initializes matrix elements from a
 * dense matrix written in a text file
 */
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

  /**
   * @brief Construct a new `MatrixInitializerFile` object
   *
   * @param filename
   * Path to file that contain the `Dense` matrix
   * @param ordering
   * Ordering of matrix elements within the text file
   */
  MatrixInitializerFile(std::string filename, MatrixLayout ordering);

  /**
   * @brief Specialization for assigning matrix elements
   *
   * @param A
   * Matrix whose elements are to be assigned.
   * @param row_range
   * Row range of \p A. The start of the `IndexRange` within the dense matrix file
   * @param col_range
   * Column range of \p A. The start of the `IndexRange` within the dense matrix file
   *
   * Traverse the text file and read elements of the corresponding \p row_range and \p col_range.
   * This method can fetch a specific submatrix from a textfile by skipping certain elements when needed.
   */
  void fill_dense_representation(
    Dense& A, const IndexRange& row_range, const IndexRange& col_range
  ) const override;

};

} // namespace hicma

#endif // hicma_classes_initialization_helpers_matrix_initializer_file_h
