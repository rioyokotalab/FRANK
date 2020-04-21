#ifndef operations_trmm_h
#define operations_trmm_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

namespace hicma
{

  class Node;

  void trmm(const Node& A, Node& B, const char& side, const char& uplo, const char& trans, const char& diag, const double& alpha);

  void trmm(const Node& A, Node& B, const char& side, const char& uplo, const double& alpha);

  MULTI_METHOD(
               trmm_omm, void,
               const virtual_<Node>& A,
               virtual_<Node>& B,
               const char& side,
               const char& uplo,
               const char& trans,
               const char& diag,
               const double& alpha
               );

} // namespace hicma

#endif
