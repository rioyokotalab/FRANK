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
   * @param admis
   * Admissibility in terms of distance from the diagonal of the matrix on the
   * current recursion level (for `AdmisType::PositionBased`) or admissibility constant
   * (for `AdmisType::GeometryBased`)
   * @param eps
   * Fixed error threshold used for approximating admissible submatrices.
   * @param rank
   * Fixed rank to be used for approximating admissible submatrices. Ignored if eps &ne; 0
   * @param params
   * Vector containing the underlying geometry information of the input `Dense` matrix
   * @param admis_type
   * Either `AdmisType::PositionBased` or `AdmisType::GeometryBased`
   */
  MatrixInitializerFile(
    std::string filename, MatrixLayout ordering, double admis, double eps, int64_t rank,
    std::vector<std::vector<double>> params, AdmisType admis_type
  );

  /**
   * @brief Specialization for assigning matrix elements
   *
   * @param A
   * Matrix whose elements are to be assigned.
   * @param row_range
   * Row range of \p A. The start of the `IndexRange` within the root
   * level `Hierarchical` matrix.
   * @param col_range
   * Column range of \p A. The start of the `IndexRange` within the root
   * level `Hierarchical` matrix.
   *
   * Traverse the text file and read elements of the corresponding \p row_range and \p col_range.
   *
   * This method can fetch a specific submatrix from a textfile. However, it is quite slow when used to fetch submatrix from a big text file.
   * So instead of constructing a `Hierarchical` matrix directly from a big text file, consider reading the whole file as a `Dense` matrix and create `Hierarchical` matrix using `MatrixInitializerBlock`, which may consumes more memory but faster in most cases
   */
  void fill_dense_representation(
    Dense& A, const IndexRange& row_range, const IndexRange& col_range
  ) const override;

};

} // namespace hicma

#endif // hicma_classes_initialization_helpers_matrix_initializer_file_h
