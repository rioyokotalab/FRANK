#ifndef hicma_classes_initialization_helpers_matrix_initializer_h
#define hicma_classes_initialization_helpers_matrix_initializer_h

#include <cstdint>
#include <vector>


namespace hicma
{

class ClusterTree;
class Dense;

class MatrixInitializer {
 private:
  void (*kernel)(
    Dense& A, const std::vector<double>& x, int64_t row_start, int64_t col_start
  ) = nullptr;
  const std::vector<double>& x;
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
    void (*kernel)(
      Dense& A, const std::vector<double>& x,
      int64_t row_start, int64_t col_start
    ),
    const std::vector<double>& x
  );

  // Utility methods
  virtual void fill_dense_representation(Dense& A, const ClusterTree& node) const;

  virtual Dense get_dense_representation(const ClusterTree& node) const;
};

} // namespace hicma


#endif // hicma_classes_initialization_helpers_matrix_initializer_h
