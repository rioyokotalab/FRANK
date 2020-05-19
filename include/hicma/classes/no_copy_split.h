#ifndef hicma_classes_no_cop_split_h
#define hicma_classes_no_cop_split_h

#include "hicma/classes/hierarchical.h"

#include <cstdint>


namespace hicma
{

class Node;

class NoCopySplit : public Hierarchical {
 public:
  // Special member functions
  NoCopySplit() = default;

  ~NoCopySplit() = default;

  NoCopySplit(const NoCopySplit& A) = default;

  NoCopySplit& operator=(const NoCopySplit& A) = default;

  NoCopySplit(NoCopySplit&& A) = default;

  NoCopySplit& operator=(NoCopySplit&& A) = default;

  // Additional constructors
  NoCopySplit(Node& A, int64_t n_row_blocks, int64_t n_col_blocks);

  NoCopySplit(const Node& A, int64_t n_row_blocks, int64_t n_col_blocks);

  NoCopySplit(Node& A, const Hierarchical& like);

  NoCopySplit(const Node& A, const Hierarchical& like);
};

} // namespace hicma

#endif // hicma_classes_no_cop_split_h
