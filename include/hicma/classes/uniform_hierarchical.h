#ifndef hicma_classes_uniform_hierarchical_h
#define hicma_classes_uniform_hierarchical_h

#include "hicma/classes/hierarchical.h"

#include "yorel/yomm2/cute.hpp"

#include <cstdint>
#include <memory>
#include <vector>


namespace hicma
{

class Dense;
class IndexRange;
class LowRankShared;
class NodeProxy;

class UniformHierarchical : public Hierarchical {
 private:
  std::vector<std::shared_ptr<Dense>> col_basis, row_basis;

  Dense make_block_row(
    int64_t row, int64_t i_abs, int64_t j_abs,
    void (*func)(
      Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin
      ),
    std::vector<double>& x,
    int64_t admis,
    int64_t i_begin, int64_t j_begin
  );

  Dense make_block_col(
    int64_t col, int64_t i_abs, int64_t j_abs,
    void (*func)(
      Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin
    ),
    std::vector<double>& x,
    int64_t admis,
    int64_t i_begin, int64_t j_begin
  );

  LowRankShared construct_shared_block_id(
    int64_t i, int64_t j, int64_t i_abs, int64_t j_abs,
    std::vector<std::vector<int64_t>>& selected_rows,
    std::vector<std::vector<int64_t>>& selected_cols,
    void (*func)(
      Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin
    ),
    std::vector<double>& x,
    int64_t rank,
    int64_t admis,
    int64_t i_begin, int64_t j_begin
  );

  LowRankShared construct_shared_block_svd(
    int64_t i, int64_t j, int64_t i_abs, int64_t j_abs,
    void (*func)(
      Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin
    ),
    std::vector<double>& x,
    int64_t rank,
    int64_t admis,
    int64_t i_begin, int64_t j_begin
  );
 public:
  // Special member functions
  UniformHierarchical() = default;

  ~UniformHierarchical() = default;

  UniformHierarchical(const UniformHierarchical& A);

  UniformHierarchical& operator=(const UniformHierarchical& A) = default;

  UniformHierarchical(UniformHierarchical&& A) = default;

  UniformHierarchical& operator=(UniformHierarchical&& A) = default;

  // Overridden functions from Hierarchical
  std::unique_ptr<Node> clone() const override;

  std::unique_ptr<Node> move_clone() override;

  const char* type() const override;

  // Conversion constructors
  UniformHierarchical(NodeProxy&&);

  // Additional constructors
  UniformHierarchical(int64_t ni_level, int64_t nj_level);

  UniformHierarchical(
    IndexRange row_range, IndexRange col_range,
    void (*func)(
      Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin
    ),
    std::vector<double>& x,
    int64_t rank,
    int64_t nleaf,
    int64_t admis=1,
    int64_t ni_level=2, int64_t nj_level=2,
    bool use_svd=false,
    int64_t i_begin=0, int64_t j_begin=0,
    int64_t i_abs=0, int64_t j_abs=0
  );

  UniformHierarchical(
    void (*func)(
      Dense& A, std::vector<double>& x, int64_t i_begin, int64_t j_begin
    ),
    std::vector<double>& x,
    int64_t ni, int64_t nj,
    int64_t rank,
    int64_t nleaf,
    int64_t admis=1,
    int64_t ni_level=2, int64_t nj_level=2,
    bool use_svd=false,
    int64_t i_begin=0, int64_t j_begin=0,
    int64_t i_abs=0, int64_t j_abs=0
  );

  // Additional indexing methods
  Dense& get_row_basis(int64_t i);

  const Dense& get_row_basis(int64_t i) const;

  Dense& get_col_basis(int64_t j);

  const Dense& get_col_basis(int64_t j) const;

  // Utiliry methods
  void copy_col_basis(const UniformHierarchical& A);

  void copy_row_basis(const UniformHierarchical& A);

  void set_col_basis(int64_t i, int64_t j);

  void set_row_basis(int64_t i, int64_t j);
};

register_class(UniformHierarchical, Hierarchical)

} // namespace hicma

#endif // hicma_classes_uniform_hierarchical_h
