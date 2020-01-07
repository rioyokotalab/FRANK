#ifndef hicma_classes_uniform_hierarchical_h
#define hicma_classes_uniform_hierarchical_h

#include "hicma/classes/hierarchical.h"

#include <memory>
#include <vector>

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

class NodeProxy;
class Dense;

class UniformHierarchical : public Hierarchical {
private:
  std::vector<std::shared_ptr<Dense>> row_basis, col_basis;
public:
  MM_CLASS(UniformHierarchical, Hierarchical);

  // Special member functions
  UniformHierarchical();

  ~UniformHierarchical();

  UniformHierarchical(const UniformHierarchical& A);

  UniformHierarchical& operator=(const UniformHierarchical& A);

  UniformHierarchical(UniformHierarchical&& A);

  UniformHierarchical& operator=(UniformHierarchical&& A);

  // Overridden functions from Hierarchical
  std::unique_ptr<Node> clone() const override;

  std::unique_ptr<Node> move_clone() override;

  const char* type() const override;

  // Conversion constructors
  UniformHierarchical(NodeProxy&&);

  // Additional constructors
  UniformHierarchical(const Node& node, int ni_level, int nj_level);

  UniformHierarchical(
    const Node& node,
    void (*func)(
      std::vector<double>& data,
      std::vector<double>& x,
      int ni, int nj,
      int i_begin, int j_begin
    ),
    std::vector<double>& x,
    int rank,
    int nleaf,
    int admis=1,
    int ni_level=2, int nj_level=2,
    bool use_svd=false
  );

  UniformHierarchical(
    void (*func)(
      std::vector<double>& data,
      std::vector<double>& x,
      int ni, int nj,
      int i_begin, int j_begin
    ),
    std::vector<double>& x,
    int ni, int nj,
    int rank,
    int nleaf,
    int admis=1,
    int ni_level=2, int nj_level=2,
    bool use_svd=false,
    int i_begin=0, int j_begin=0,
    int i_abs=0, int j_abs=0,
    int level=0
  );

  // Additional indexing methods
  Dense& get_row_basis(int i);

  const Dense& get_row_basis(int i) const;

  Dense& get_col_basis(int j);

  const Dense& get_col_basis(int j) const;

  // Utiliry methods
  void copy_col_basis(const UniformHierarchical& A);

  void copy_row_basis(const UniformHierarchical& A);

  void set_col_basis(int i, int j);

  void set_row_basis(int i, int j);
};

} // namespace hicma

#endif // hicma_classes_uniform_hierarchical_h
