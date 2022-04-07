#ifndef hicma_classes_initialization_helpers_matrix_initializer_file_h
#define hicma_classes_initialization_helpers_matrix_initializer_file_h

#include "hicma/classes/initialization_helpers/matrix_initializer.h"

#include <string>


namespace hicma
{

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
      std::string filename, MatrixLayout ordering, double admis=0, int64_t rank=0,
      int admis_type=POSITION_BASED_ADMIS, vec2d<double> params= vec2d<double>()
    );

    //template<typename P>
    void fill_dense_representation(
      Matrix& A, const IndexRange& row_range, const IndexRange& col_range
    ) const override;
};

} // namespace hicma

#endif // hicma_classes_initialization_helpers_matrix_initializer_file_h
