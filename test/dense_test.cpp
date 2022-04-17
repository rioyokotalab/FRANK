#include "hicma/hicma.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>


TEST(DenseTest, ConstructorHierarchical) {
  hicma::initialize();
  // Check whether the Dense(Hierarchical) constructor works correctly.
  constexpr int64_t N = 128;
  constexpr int64_t nblocks = 4;
  constexpr int64_t nleaf = N / nblocks;
  // Construct single level all-dense hierarchical
  const hicma::Hierarchical H(hicma::random_uniform, {},
                              N, N, 0, nleaf, nblocks, nblocks, nblocks);
  const hicma::Dense D(H);
  // Check block-by-block and element-by-element if values match
  for (int64_t ib=0; ib<nblocks; ++ib) {
    for (int64_t jb=0; jb<nblocks; ++jb) {
      hicma::Dense D_compare = hicma::Dense(H(ib, jb));
      for (int64_t i=0; i<nleaf; ++i) {
        for (int64_t j=0; j<nleaf; ++j) {
          ASSERT_EQ(D(nleaf*ib+i, nleaf*jb+j), D_compare(i, j));
        }
      }
    }
  }
}

TEST(DenseTest, Split1DTest) {
  hicma::initialize();
  constexpr int64_t N = 128;
  constexpr int64_t nblocks = 2;
  constexpr int64_t nleaf = N / nblocks;
  const hicma::Dense D(hicma::random_uniform, {}, N, N);
  const hicma::Hierarchical DH = hicma::split(D, nblocks, nblocks);
  const hicma::Hierarchical DH_copy = hicma::split(D, nblocks, nblocks, true);
    
  for (int64_t ib=0; ib<nblocks; ++ib) {
    for (int64_t jb=0; jb<nblocks; ++jb) {
      hicma::Dense D_compare = DH(ib, jb) - DH_copy(ib, jb);
      for (int64_t i=0; i<nleaf; ++i) {
        for (int64_t j=0; j<nleaf; ++j) {
          ASSERT_EQ(0, D_compare(i, j));
        }
      }
    }
  }
}

TEST(DenseTest, SplitTest) {
  hicma::initialize();
  constexpr int64_t N = 128;
  constexpr int64_t nblocks = 4;
  constexpr int64_t nleaf = N / nblocks;
  const hicma::Dense col(hicma::random_normal, {}, N, nleaf);
  const hicma::Dense row(hicma::random_normal, {}, nleaf, N);
  hicma::Dense test1 = hicma::gemm(row, col);
  test1 *= 2;

  const hicma::Hierarchical colH = hicma::split(col, nblocks, 1);
  const hicma::Hierarchical rowH = hicma::split(row, 1, nblocks);
  hicma::Dense test2 = hicma::gemm(rowH, colH);
  test2 *= 2;
  for (int64_t i=0; i<nleaf; ++i) {
    for (int64_t j=0; j<nleaf; ++j) {
      ASSERT_FLOAT_EQ(test1(i, j), test2(i, j));
    }
  }
}

TEST(DenseTest, Resize) {
  hicma::initialize();
  constexpr int64_t N = 1024;
  const hicma::Dense D(hicma::random_normal, {}, N, N);
  const hicma::Dense D_resized = hicma::resize(D, N-N/8, N-N/8);
  for (int64_t i=0; i<D_resized.dim[0]; ++i) {
    for (int64_t j=0; j<D_resized.dim[1]; ++j) {
      ASSERT_EQ(D(i, j), D_resized(i, j));
    }
  }
}

TEST(DenseTest, Assign) {
  hicma::initialize();
  constexpr int64_t N = 24;
  hicma::Dense D(N, N);
  D = 8;
  for (int64_t i=0; i<N; ++i) {
    for (int64_t j=0; j<N; ++j) {
      ASSERT_EQ(D(i, j), 8);
    }
  }
}

TEST(DenseTest, Copy) {
  hicma::initialize();
  constexpr int64_t N = 42;
  const hicma::Dense D(hicma::random_normal, {}, N, N);
  const hicma::Dense A(D);
  hicma::Dense B(N, N);
  A.copy_to(B);
  for (int64_t i=0; i<N; ++i) {
    for (int64_t j=0; j<N; ++j) {
      ASSERT_EQ(D(i, j), A(i, j));
      ASSERT_EQ(D(i, j), B(i, j));
    }
  }
  hicma::Dense C(30, 30);
  const int offset = 12;
  D.copy_to(C, offset, offset);
  for (int64_t i=0; i<C.dim[0]; ++i) {
    for (int64_t j=0; j<C.dim[1]; ++j) {
      ASSERT_EQ(D(offset+i, offset+j), C(i, j));
      ASSERT_EQ(D(offset+i, offset+j), C(i, j));
    }
  }
}
