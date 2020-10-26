#include "hicma/hicma.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>


using namespace hicma;

TEST(DenseTest, ContructorHierarchical) {
  hicma::initialize();
  // Check whether the Dense(Hierarchical) constructor works correctly.
  int64_t N = 128;
  int64_t nblocks = 4;
  int64_t nleaf = N / nblocks;
  // Construct single level all-dense hierarchical
  Hierarchical H(
    random_uniform, std::vector<std::vector<double>>(),
    N, N, 0, nleaf, nblocks, nblocks, nblocks
  );
  Dense D(H);
  // Check block-by-block and element-by-element if values match
  for (int64_t ib=0; ib<nblocks; ++ib) {
    for (int64_t jb=0; jb<nblocks; ++jb) {
      Dense D_compare = Dense(H(ib, jb));
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
  int64_t N = 128;
  int64_t nblocks = 2;
  int64_t nleaf = N / nblocks;
  start_schedule();
  Dense D(random_uniform, std::vector<std::vector<double>>(), N, N);
  Hierarchical DH = split(D, nblocks, nblocks);
  Hierarchical DH_copy = split(D, nblocks, nblocks, true);
  execute_schedule();
  for (int64_t ib=0; ib<nblocks; ++ib) {
    for (int64_t jb=0; jb<nblocks; ++jb) {
      Dense D_compare = DH(ib, jb) - DH_copy(ib, jb);
      for (int64_t i=0; i<nleaf; ++i) {
        for (int64_t j=0; j<nleaf; ++j) {
          ASSERT_EQ(0, D_compare(i, j));
        }
      }
    }
  }

  start_schedule();
  Hierarchical H = split(D, nblocks, nblocks);
  Dense HD(H);
  Dense Q(HD.dim[0], HD.dim[1]);
  Dense R(HD.dim[1], HD.dim[1]);
  qr(HD, Q, R);
  Dense QR = gemm(Q, R);
  execute_schedule();
  for (int64_t i=0; i<N; ++i) {
    for (int64_t j=0; j<N; ++j) {
      ASSERT_FLOAT_EQ(D(i, j), QR(i, j));
    }
  }
}

TEST(DenseTest, SplitTest) {
  hicma::initialize();
  int64_t N = 128;
  int64_t nblocks = 4;
  int64_t nleaf = N / nblocks;
  Dense col(random_normal, std::vector<std::vector<double>>(), N, nleaf);
  Dense row(random_normal, std::vector<std::vector<double>>(), nleaf, N);
  // start_schedule();
  Dense test1 = gemm(row, col);
  test1 *= 2;
  // execute_schedule();

  start_schedule();
  Hierarchical colH = split(col, nblocks, 1);
  Hierarchical rowH = split(row, 1, nblocks);
  Dense test2 = gemm(rowH, colH);
  test2 *= 2;
  execute_schedule();
  for (int64_t i=0; i<nleaf; ++i) {
    for (int64_t j=0; j<nleaf; ++j) {
      ASSERT_FLOAT_EQ(test1(i, j), test2(i, j));
    }
  }
}

TEST(DenseTest, Resize) {
  timing::start("Init");
  hicma::initialize();
  int64_t N = 1024;
  Dense D(random_normal, std::vector<std::vector<double>>(), N, N);
  timing::stopAndPrint("Init");
  timing::start("Resize");
  Dense D_resized = resize(D, N-N/8, N-N/8);
  timing::stopAndPrint("Resize");
  timing::start("Check results");
  for (int64_t i=0; i<D_resized.dim[0]; ++i) {
    for (int64_t j=0; j<D_resized.dim[1]; ++j) {
      ASSERT_EQ(D(i, j), D_resized(i, j));
    }
  }
  timing::stopAndPrint("Check results");
}
