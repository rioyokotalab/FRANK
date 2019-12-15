#ifndef hicma_classes_uniform_hierarchical_h
#define hicma_classes_uniform_hierarchical_h

#include "hicma/classes/hierarchical.h"

#include <memory>
#include <vector>

#include "yorel/multi_methods.hpp"

namespace hicma
{

class NodeProxy;
class Dense;

class UniformHierarchical : public Hierarchical {
 public:
  MM_CLASS(UniformHierarchical, Hierarchical);
  std::vector<std::shared_ptr<Dense>> row_basis, col_basis;

  UniformHierarchical();

  UniformHierarchical(
    void (*func)(
      std::vector<double>& data,
      std::vector<double>& x,
      const int& ni,
      const int& nj,
      const int& i_begin,
      const int& j_begin
    ),
    std::vector<double>& x,
    const int ni,
    const int nj,
    const int rank,
    const int nleaf,
    const int admis=1,
    const int ni_level=2,
    const int nj_level=2,
    const int i_begin=0,
    const int j_begin=0,
    const int i_abs=0,
    const int j_abs=0,
    const int level=0
  );

  UniformHierarchical(const UniformHierarchical& A);

  UniformHierarchical(UniformHierarchical&& A);

  UniformHierarchical* clone() const override;

  UniformHierarchical* move_clone() override;

  friend void swap(UniformHierarchical& A, UniformHierarchical& B);

  const char* type() const override;

  const NodeProxy& operator[](const int i) const;

  NodeProxy& operator[](const int i);

  const NodeProxy& operator()(const int i, const int j) const;

  NodeProxy& operator()(const int i, const int j);
};

} // namespace hicma

#endif // hicma_classes_uniform_hierarchical_h
