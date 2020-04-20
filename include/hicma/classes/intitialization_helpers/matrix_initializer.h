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
  std::vector<double>& x;
  void (*kernel)(
    Dense& A, std::vector<double>& x, int64_t row_start, int64_t col_start
  ) = nullptr;
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
    std::vector<double>& x,
    void (*kernel)(
      Dense& A, std::vector<double>& x, int64_t row_start, int64_t col_start)
  );

  // Utility methods
  virtual void fill_dense_representation(Dense& A, const ClusterTree& node) const;

  virtual Dense get_dense_representation(const ClusterTree& node) const;
};

} // namespace hicma


#endif // hicma_classes_initialization_helpers_matrix_initializer_h
