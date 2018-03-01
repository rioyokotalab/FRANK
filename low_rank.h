#ifndef low_rank_h
#define low_rank_h
#include <cassert>
#include <vector>

namespace hicma {
  class LowRank {
  public:
    Dense U, B, V;
    int dim[2];
    int rank;

    LowRank() {
      dim[0]=0; dim[1]=0; rank=0;
    }

    LowRank(int i, int j, int k) {
      dim[0]=i; dim[1]=j; rank=k;
      U.resize(dim[0],rank);
      B.resize(rank,rank);
      V.resize(rank,dim[1]);
    }

    LowRank(LowRank &LR) {
      dim[0]=LR.dim[0]; dim[1]=LR.dim[1]; rank=LR.rank;
      for (int i=0; i<dim[0]*rank; i++) U[i] = LR.U[i];
      for (int i=0; i<rank*rank; i++) B[i] = LR.B[i];
      for (int i=0; i<rank*dim[1]; i++) V[i] = LR.V[i];
    }

    Dense operator+(const Dense& D) const {
      return U * B * V + D;
    }

    LowRank operator*(const Dense& D) {
      LowRank LR = *this;
      LR.V = LR.V * D;
      return LR;
    }

  };
}
#endif
