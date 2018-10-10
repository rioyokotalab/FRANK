#include "rsvd.h"

namespace hicma {
  void transpose(const std::vector<double>&  mat, std::vector<double>& mat_t, int nrows, int ncols) {
    for (int i=0; i<nrows; i++) {
      for (int j=0; j<ncols; j++) {
        mat_t[i*ncols+j] = mat[j*nrows+i];
      }
    }
  }

  void rsvd(const Dense& A, int rank, Dense& U, Dense& S, Dense& V) {
    int nrows = A.dim[0];
    int ncols = A.dim[1];

    // RN = randn(n,k+p)
    Dense RN(ncols,rank);
    std::mt19937 generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (int i=0; i<ncols*rank; i++) {
      RN[i] = distribution(generator);
    }

    // Y = A * RN
    Dense Y(nrows,rank);
    Y.gemm(A, RN, 1, 0);

    // [Q, R] = qr(Y)
    Dense Q(nrows,rank);
    Dense R(rank,rank);
    Y.qr(Q, R);

    // B' = A' * Q
    Dense At(ncols,nrows);
    transpose(A.data, At.data, ncols, nrows);

    Dense Bt(ncols,rank);
    Bt.gemm(At, Q, 1, 0);

    // [Qb, Rb] = qr(B')
    Dense Qb(ncols,rank);
    Dense Rb(rank,rank);
    Bt.qr(Qb,Rb);

    // [Ur, S, Vr] = svd(Rb);
    Dense Ur(rank,rank);
    Dense Vr(rank,rank);
    Rb.svd(Ur,S,Vr);

    // Vr = Vr'
    Dense Vr_t(rank,rank);
    transpose(Vr.data, Vr_t.data, rank, rank);

    // U = Q * Vr'
    U.gemm(Q, Vr_t);

    // V' = Qb * Ur
    Dense V_t(ncols,rank);
    V_t.gemm(Qb, Ur);

    // V = V'
    transpose(V_t.data, V.data, rank, ncols);
  }
}
