#include "hicma/uniform_hierarchical.h"

#include "hicma/node.h"
#include "hicma/node_proxy.h"
#include "hicma/dense.h"
#include "hicma/low_rank.h"
#include "hicma/low_rank_shared.h"
#include "hicma/hierarchical.h"
#include "hicma/operations/gemm.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>

#ifndef USE_MKL
#include <cblas.h>
#include <lapacke.h>
#else
#include <mkl.h>
#endif
#include "yorel/multi_methods.hpp"


#include <cstdio>

namespace hicma
{

UniformHierarchical::UniformHierarchical() : Hierarchical() {
  MM_INIT();
}

UniformHierarchical::UniformHierarchical(
  void (*func)(
    std::vector<double>& data,
    std::vector<double>& x,
    const int& ni, const int& nj,
    const int& i_begin, const int& j_begin
  ),
  std::vector<double>& x,
  const int ni, const int nj,
  const int rank,
  const int nleaf,
  const int admis,
  const int ni_level, const int nj_level,
  const int i_begin, const int j_begin,
  const int i_abs, const int j_abs,
  const int level
) : Hierarchical(ni_level, nj_level, i_abs, j_abs, level) {
  MM_INIT();
  if (!level) {
    assert(int(x.size()) == std::max(ni,nj));
    std::sort(x.begin(),x.end());
  }
  col_basis.resize(dim[0]);
  row_basis.resize(dim[1]);
  for (int i=0; i<dim[0]; i++) {
    for (int j=0; j<dim[1]; j++) {
      int ni_child = ni/dim[0];
      if ( i == dim[0]-1 ) ni_child = ni - (ni/dim[0]) * (dim[0]-1);
      int nj_child = nj/dim[1];
      if ( j == dim[1]-1 ) nj_child = nj - (nj/dim[1]) * (dim[1]-1);
      int i_begin_child = i_begin + ni/dim[0] * i;
      int j_begin_child = j_begin + nj/dim[1] * j;
      int i_abs_child = i_abs * dim[0] + i;
      int j_abs_child = j_abs * dim[1] + j;
      if (
          std::abs(i_abs_child - j_abs_child) <= admis // Check regular admissibility
          || (nj == 1 || ni == 1) ) { // Check if vector, and if so do not use LowRank
        if ( ni_child/ni_level < nleaf && nj_child/nj_level < nleaf ) {
          (*this)(i,j) = Dense(
            func,
            x,
            ni_child, nj_child,
            i_begin_child, j_begin_child,
            i_abs_child, j_abs_child,
            level+1
          );
        }
        else {
          (*this)(i,j) = UniformHierarchical(
            func,
            x,
            ni_child, nj_child,
            rank,
            nleaf,
            admis,
            ni_level, nj_level,
            i_begin_child, j_begin_child,
            i_abs_child, j_abs_child,
            level+1
          );
        }
      }
      else {
        if (col_basis[i].get() == nullptr) {
          // Take entire row block of matrix (slim and wide)
          Dense bottom_row(
            func,
            x,
            ni_child, nj*(std::pow(nj_level, level)),
            i_begin + ni_child*i, 0
          );
          // Split into parts
          Hierarchical bottom_row_h(bottom_row, 1, std::pow(nj_level, level));
          // Set part covered by dense blocks to 0
          for (int j_b=0; j_b<std::pow(nj_level, level); ++j_b) {
            if (std::abs(j_b - i_abs_child) <= admis)
              static_cast<Dense&>(*bottom_row_h[j_b].ptr) = 0.0;
          }
          // Reconvert to dense and get U of top row
          // Likely not efficient either!
          col_basis[i] = std::make_shared<Dense>(
            LowRank(Dense(bottom_row_h), rank).U);
        }
        if (row_basis[j].get() == nullptr) {
          // Take entire column block of matrix (tall and slim)
          Dense left_col(
            func,
            x,
            ni*std::pow(ni_level, level), nj_child,
            0, j_begin + nj_child*j
          );
          // Split into parts
          Hierarchical left_col_h(left_col, std::pow(ni_level, level), 1);
          // Set part covered by (*this)(0, 0) to 0
          for (int i_b=0; i_b<std::pow(ni_level, level); ++i_b) {
            if (std::abs(i_b - j_abs_child) <= admis)
              static_cast<Dense&>(*left_col_h[i_b].ptr) = 0.0;
          }
          // Reconvert to dense and get V of right col
          // Likely not efficient either!
          row_basis[j] = std::make_shared<Dense>(
            LowRank(Dense(left_col_h), rank).V);
        }
        Dense D(
          func,
          x,
          ni_child, nj_child,
          i_begin_child, j_begin_child
        );
        Dense UtxD(rank, rank);
        gemm(*col_basis[i], D, UtxD, CblasTrans, CblasNoTrans, 1, 0);
        Dense S(rank, rank, i_abs_child, j_abs_child, level+1);
        gemm(UtxD, *row_basis[j], S, CblasNoTrans, CblasTrans, 1, 0);
        (*this)(i, j) = LowRankShared(
          S,
          col_basis[i], row_basis[j]
        );
      }
    }
  }
}

UniformHierarchical::UniformHierarchical(const UniformHierarchical& A)
  : Hierarchical(A) {
  MM_INIT();
}

UniformHierarchical::UniformHierarchical(UniformHierarchical&& A) {
  MM_INIT();
  swap(*this, A);
}

UniformHierarchical* UniformHierarchical::clone() const {
  return new UniformHierarchical(*this);
}

void swap(UniformHierarchical& A, UniformHierarchical& B) {
  using std::swap;
  swap(static_cast<Node&>(A), static_cast<Node&>(B));
  swap(A.data, B.data);
  swap(A.dim, B.dim);
}

const char* UniformHierarchical::type() const { return "UniformHierarchical"; }

const NodeProxy& UniformHierarchical::operator[](const int i) const {
  assert(i<dim[0]*dim[1]);
  return data[i];
}

NodeProxy& UniformHierarchical::operator[](const int i) {
  assert(i<dim[0]*dim[1]);
  return data[i];
}

const NodeProxy& UniformHierarchical::operator()(const int i, const int j) const {
  assert(i<dim[0] && j<dim[1]);
  return data[i*dim[1]+j];
}

NodeProxy& UniformHierarchical::operator()(const int i, const int j) {
  assert(i<dim[0] && j<dim[1]);
  return data[i*dim[1]+j];
}

} // namespace hicma
