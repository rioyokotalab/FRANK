#include "hicma/operations/misc.h"

#include "hicma/classes/dense.h"
#include "hicma/classes/hierarchical.h"
#include "hicma/classes/low_rank.h"
#include "hicma/classes/matrix_proxy.h"
#include "hicma/classes/nested_basis.h"
#include "hicma/operations/BLAS.h"
#include "hicma/operations/LAPACK.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <cassert>
#include <cstdint>
#include <tuple>
#include <vector>




#include "hicma/util/print.h"
#include "hicma/util/l2_error.h"

namespace hicma
{

declare_method(
  MatrixProxy, get_uncomp_col_basis,
  (virtual_<Matrix&>, std::vector<LowRank*>&, int64_t&)
)

define_method(
  MatrixProxy, get_uncomp_col_basis,
  (LowRank& A, std::vector<LowRank*>& lr_row, int64_t& desired_rank)
) {
  if (A.S.dim[0] > A.S.dim[1]) {
    if (desired_rank == 0) {
      desired_rank = A.S.dim[1];
    } else {
      assert(desired_rank == A.S.dim[1]);
    }
    lr_row.push_back(&A);
    return share_basis(A.U);
  } else {
    // Do nothing
    return MatrixProxy();
  }
}

define_method(
  MatrixProxy, get_uncomp_col_basis,
  (Matrix&, std::vector<LowRank*>&, int64_t&)
) {
  // Do nothing
  return MatrixProxy();
}

declare_method(
  MatrixProxy, get_uncomp_row_basis,
  (virtual_<Matrix&>, std::vector<LowRank*>&, int64_t&)
)

define_method(
  MatrixProxy, get_uncomp_row_basis,
  (LowRank& A, std::vector<LowRank*>& lr_col, int64_t& desired_rank)
) {
  if (A.S.dim[1] > A.S.dim[0]) {
    if (desired_rank == 0) {
      desired_rank = A.S.dim[0];
    } else {
      assert(desired_rank == A.S.dim[0]);
    }
    lr_col.push_back(&A);
    return share_basis(A.V);
  } else {
    // Do nothing
    return MatrixProxy();
  }
}

define_method(
  MatrixProxy, get_uncomp_row_basis,
  (Matrix&, std::vector<LowRank*>&, int64_t&)
) {
  // Do nothing
  return MatrixProxy();
}

void recompress(Hierarchical& A, int64_t start) {
  // TODO only works for a single layer atm (ie shared basis UBLR).
  // Col basis
  for (int64_t i=start; i<A.dim[0]; ++i) {
    std::vector<LowRank*> lr_row;
    MatrixProxy U;
    int64_t desired_rank = 0;
    for (int64_t j=start; j<A.dim[1]; ++j) {
      U = get_uncomp_col_basis(A(i, j), lr_row, desired_rank);
    }
    if (!lr_row.empty()) {
      // TODO inefficient due to copies
      Hierarchical lr_rowH(1, lr_row.size());
      for (uint64_t j=0; j<lr_row.size(); ++j) {
        lr_rowH[j] = lr_row[j]->S;
      }
      Dense US = gemm(U, lr_rowH);
      Dense new_UD, S, V;
      std::tie(new_UD, S, V) = svd(US);
      NestedBasis new_U(resize(new_UD, new_UD.dim[0], desired_rank), true);
      Dense new_S = gemm(
        resize(S, desired_rank, desired_rank), resize(V, desired_rank, V.dim[1])
      );
      // TODO Assumes all ranks are equal and as before!
      for (uint64_t j=0; j<lr_row.size(); ++j) {
        lr_row[j]->U = share_basis(new_U);
        lr_row[j]->S = Dense(
          new_S,
          new_S.dim[0], lr_row[j]->S.dim[1], 0, lr_row[j]->S.dim[1]*j,
          true
        );
      }
    }
  }
  // Row basis
  for (int64_t j=start; j<A.dim[1]; ++j) {
    std::vector<LowRank*> lr_col;
    MatrixProxy V;
    int64_t desired_rank = 0;
    for (int64_t i=start; i<A.dim[0]; ++i) {
      V = get_uncomp_row_basis(A(i, j), lr_col, desired_rank);
    }
    if (!lr_col.empty()) {
      // TODO inefficient due to copies
      Hierarchical lr_colH(lr_col.size(), 1);
      for (uint64_t i=0; i<lr_col.size(); ++i) {
        lr_colH[i] = lr_col[i]->S;
      }
      Dense SV = gemm(lr_colH, V);
      Dense U, S, new_VD;
      std::tie(U, S, new_VD) = svd(SV);
      NestedBasis new_V(resize(new_VD, desired_rank, new_VD.dim[1]), false);
      Dense new_S = gemm(
        resize(U, U.dim[0], desired_rank), resize(S, desired_rank, desired_rank)
      );
      // TODO Assumes all ranks are equal and as before!
      for (uint64_t i=0; i<lr_col.size(); ++i) {
        lr_col[i]->V = share_basis(new_V);
        lr_col[i]->S = Dense(
          new_S,
          lr_col[i]->S.dim[0], new_S.dim[1], lr_col[i]->S.dim[0]*i, 0,
          true
        );
      }
    }
  }
}

} // namespace hicma
