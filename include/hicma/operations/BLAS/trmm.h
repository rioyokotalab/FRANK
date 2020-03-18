#ifndef operations_trmm_h
#define operations_trmm_h

#include "hicma/classes/node.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;


namespace hicma
{

  void trmm(
    const Node& A, Node& B,
    const char& side, const char& uplo, const char& trans, const char& diag,
    const double& alpha
  );

  void trmm(
    const Node& A, Node& B,
    const char& side, const char& uplo,
    const double& alpha
  );

  declare_method(
    void, trmm_omm,
    (
      virtual_<const Node&>, virtual_<Node&>,
      const char&, const char&, const char&, const char&,
      const double&
    )
  );

} // namespace hicma

#endif
