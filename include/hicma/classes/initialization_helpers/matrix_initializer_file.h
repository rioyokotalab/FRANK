#ifndef hicma_classes_initialization_helpers_matrix_initializer_file_h
#define hicma_classes_initialization_helpers_matrix_initializer_file_h

#include "hicma/classes/initialization_helpers/matrix_initializer.h"

#include <string>


namespace hicma
{

template<typename U = double>
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
      std::string filename, MatrixLayout ordering, double admis, int64_t rank
    );

    void fill_dense_representation(
      Matrix& A, const IndexRange& row_range, const IndexRange& col_range
    ) const override;
    
    template<typename T>
    void fill_dense_representation(
      Dense<T>& A, const IndexRange& row_range, const IndexRange& col_range
    ) const;
};

} // namespace hicma

#endif // hicma_classes_initialization_helpers_matrix_initializer_file_h
