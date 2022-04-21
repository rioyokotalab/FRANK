#include "FRANK/FRANK.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>


TEST(DenseTest, ConstructorHierarchical) {
  FRANK::initialize();
  // Check whether the Dense(Hierarchical) constructor works correctly.
  constexpr int64_t N = 128;
  constexpr int64_t nblocks = 4;
  constexpr int64_t nleaf = N / nblocks;
  // Construct single level all-dense hierarchical
  const FRANK::Hierarchical H(FRANK::random_uniform, {},
                              N, N, 0, nleaf, nblocks, nblocks, nblocks);
  const FRANK::Dense D(H);
  // Check block-by-block and element-by-element if values match
  for (int64_t ib=0; ib<nblocks; ++ib) {
    for (int64_t jb=0; jb<nblocks; ++jb) {
      FRANK::Dense D_compare = FRANK::Dense(H(ib, jb));
      for (int64_t i=0; i<nleaf; ++i) {
        for (int64_t j=0; j<nleaf; ++j) {
          ASSERT_EQ(D(nleaf*ib+i, nleaf*jb+j), D_compare(i, j));
        }
      }
    }
  }
}

TEST(DenseTest, Split1DTest) {
  FRANK::initialize();
  constexpr int64_t N = 128;
  constexpr int64_t nblocks = 2;
  constexpr int64_t nleaf = N / nblocks;
  const FRANK::Dense D(FRANK::random_uniform, {}, N, N);
  const FRANK::Hierarchical DH = FRANK::split(D, nblocks, nblocks);
  const FRANK::Hierarchical DH_copy = FRANK::split(D, nblocks, nblocks, true);
    
  for (int64_t ib=0; ib<nblocks; ++ib) {
    for (int64_t jb=0; jb<nblocks; ++jb) {
      FRANK::Dense D_compare = DH(ib, jb) - DH_copy(ib, jb);
      for (int64_t i=0; i<nleaf; ++i) {
        for (int64_t j=0; j<nleaf; ++j) {
          ASSERT_EQ(0, D_compare(i, j));
        }
      }
    }
  }
}

TEST(DenseTest, SplitTest) {
  FRANK::initialize();
  constexpr int64_t N = 128;
  constexpr int64_t nblocks = 4;
  constexpr int64_t nleaf = N / nblocks;
  const FRANK::Dense col(FRANK::random_normal, {}, N, nleaf);
  const FRANK::Dense row(FRANK::random_normal, {}, nleaf, N);
  FRANK::Dense test1 = FRANK::gemm(row, col);
  test1 *= 2;

  const FRANK::Hierarchical colH = FRANK::split(col, nblocks, 1);
  const FRANK::Hierarchical rowH = FRANK::split(row, 1, nblocks);
  FRANK::Dense test2 = FRANK::gemm(rowH, colH);
  test2 *= 2;
  for (int64_t i=0; i<nleaf; ++i) {
    for (int64_t j=0; j<nleaf; ++j) {
      ASSERT_FLOAT_EQ(test1(i, j), test2(i, j));
    }
  }
}

TEST(DenseTest, Resize) {
  FRANK::initialize();
  constexpr int64_t N = 1024;
  const FRANK::Dense D(FRANK::random_normal, {}, N, N);
  const FRANK::Dense D_resized = FRANK::resize(D, N-N/8, N-N/8);
  for (int64_t i=0; i<D_resized.dim[0]; ++i) {
    for (int64_t j=0; j<D_resized.dim[1]; ++j) {
      ASSERT_EQ(D(i, j), D_resized(i, j));
    }
  }
}

TEST(DenseTest, Assign) {
  FRANK::initialize();
  constexpr int64_t N = 24;
  FRANK::Dense D(N, N);
  D = 8;
  for (int64_t i=0; i<N; ++i) {
    for (int64_t j=0; j<N; ++j) {
      ASSERT_EQ(D(i, j), 8);
    }
  }
}

TEST(DenseTest, Copy) {
  FRANK::initialize();
  constexpr int64_t N = 42;
  const FRANK::Dense D(FRANK::random_normal, {}, N, N);
  const FRANK::Dense A(D);
  FRANK::Dense B(N, N);
  A.copy_to(B);
  for (int64_t i=0; i<N; ++i) {
    for (int64_t j=0; j<N; ++j) {
      ASSERT_EQ(D(i, j), A(i, j));
      ASSERT_EQ(D(i, j), B(i, j));
    }
  }
  FRANK::Dense C(30, 30);
  const int offset = 12;
  D.copy_to(C, offset, offset);
  for (int64_t i=0; i<C.dim[0]; ++i) {
    for (int64_t j=0; j<C.dim[1]; ++j) {
      ASSERT_EQ(D(offset+i, offset+j), C(i, j));
      ASSERT_EQ(D(offset+i, offset+j), C(i, j));
    }
  }
}
