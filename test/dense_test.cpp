#include "hicma/hicma.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>


using namespace hicma;

TEST(DenseTest, StandardConstructors) {
  hicma::initialize();
  int64_t n = 4;

  // Standard Constructors
  Dense<float> Af;
  EXPECT_EQ(Af.dim[0], 0);
  EXPECT_EQ(Af.dim[1], 0);
  EXPECT_EQ(Af.stride, 0);
  Dense<double> A(n);
  EXPECT_EQ(A.dim[0], 4);
  EXPECT_EQ(A.dim[1], 1);
  EXPECT_EQ(A.stride, 1);
  for (int64_t i=0; i<n; ++i) {
    EXPECT_EQ(A[i], 0);
  }

  // Row/Column Constructor
  Dense<double> B(n, n);
  EXPECT_EQ(B.dim[0], 4);
  EXPECT_EQ(B.dim[1], 4);
  EXPECT_EQ(B.stride, 4);
  Dense<float> Cf(n, n);
  EXPECT_EQ(Cf.dim[0], 4);
  EXPECT_EQ(Cf.dim[1], 4);
  EXPECT_EQ(Cf.stride, 4);
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      B(i,j) = i*j;
      Cf(i,j) = -i*j;
    }
  }

  // Copy/MatrixProxy constructor, same datatype
  Dense<double> B_copy(B);
  Dense<float> Cf_copy(Cf);
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      EXPECT_EQ(B_copy(i,j), B(i,j));
      EXPECT_EQ(Cf_copy(i,j), Cf(i,j));
    }
  }

  // Copy constructor, different datatype
  Dense<float> Bf(B);
  Dense<double> C(Cf);
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      EXPECT_EQ(Bf(i,j), B(i,j));
      EXPECT_EQ(Cf(i,j), C(i,j));
    } 
  }

  // construct from MatrixProxy (works only for same datatype)
  Dense<double> D(std::move(MatrixProxy(B)));
  Dense<float> Df(std::move(MatrixProxy(Bf)));
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      EXPECT_EQ(D(i,j), B(i,j));
      EXPECT_EQ(Df(i,j), Bf(i,j));
    } 
  }
}


TEST(DenseTest, ConstructorKernel) {
  hicma::initialize();
  int64_t n = 10;
  int64_t offset = 5;

  // construct from Kernel
  Dense<float> Af(IdentityKernel<double>(), n, n);
  Dense<double> A(ArangeKernel<double>(), n, n, offset, offset);
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      EXPECT_EQ(Af(i,j), i!=j?0:1);
      EXPECT_EQ(A(i,j), (i+offset)*n + offset+j);
    } 
  }

  // construct from Function
  // TODO Legacy code, remove?
  Dense<double> B(
    arange, std::vector<std::vector<double>>(),
    n, n, offset, offset);
  Dense<float> Bf(
    identity, std::vector<std::vector<double>>(), n, n);
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      // note that the old arange function ignores the offset!
      EXPECT_EQ(B(i,j), i*n + j);
      EXPECT_EQ(Af(i,j), Bf(i,j));
    } 
  }
}


TEST(DenseTest, ConstructorFile) {
  hicma::initialize();
  int64_t n = 5;
  Dense<double> A("test_data/simple_matrix.txt", HICMA_ROW_MAJOR, n, n);
  Dense<float> Af("test_data/simple_matrix.txt", HICMA_COL_MAJOR, n, n);
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      EXPECT_EQ(A(i,j), i*n + j);
      EXPECT_EQ(A(i,j), Af(j,i));
    } 
  }
}


TEST(DenseTest, Assign) {
  hicma::initialize();
  int64_t n = 3;

  // Scalar
  Dense<float> Af(n, n);
  Dense<double> A(n, n);
  const double D_value = 2.5;
  A = D_value;
  Af = D_value;
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      EXPECT_EQ(A(i,j), D_value);
      EXPECT_EQ(Af(i,j), D_value);
    }
  }
  const float F_value = -0.3;
  A = F_value;
  Af = F_value;
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      EXPECT_EQ(A(i,j), F_value);
      EXPECT_EQ(Af(i,j), F_value);
    }
  }

  // Other matrix (works only for same datatype)
  Dense<double> B;
  Dense<float> Bf;
  B = A;
  Bf = Af;
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      EXPECT_EQ(A(i,j), B(i,j));
      EXPECT_EQ(Af(i,j), Bf(i,j));
    }
  }
}


TEST(DenseTEST, ConstructorLowRank) {
  hicma::initialize();
  int64_t m = 3;
  int64_t n = 2;

  Dense<double> U(ArangeKernel<double>(), m, n);
  Dense<double> V(ArangeKernel<double>(), n, m);
  Dense<double> S(IdentityKernel<double>(), n, n);
  LowRank<double> USV (U, S, V);

  Dense<double> check(m, m);
  check(0,0) = 3;
  check(0,1) = 4;
  check(0,2) = 5;
  check(1,0) = 9;
  check(1,1) = 14;
  check(1,2) = 19;
  check(2,0) = 15;
  check(2,1) = 24;
  check(2,2) = 33;

  // works only for the same datatype
  Dense<double> A(USV);
  for (int64_t i=0; i<m; ++i) {
    for (int64_t j=0; j<m; ++j) {
      EXPECT_EQ(A(i,j), check(i,j));
    }
  }
}


TEST(DenseTest, ContructorHierarchical) {
  hicma::initialize();
  int64_t n = 128;
  int64_t nblocks = 4;
  int64_t nleaf = n / nblocks;

  // Construct single level all-dense hierarchical
  Hierarchical H(RandomUniformKernel<double>(),
    n, n, 0, nleaf, nblocks, nblocks, nblocks
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


TEST(DenseTest, ElementAccess) {
  int64_t n = 8;
  Dense<float> Af(ArangeKernel<float>(), n);
  Dense<double> A(ArangeKernel<float>(), 1, n);
  for (int64_t i=0; i<n; ++i) {
      EXPECT_EQ(Af(i,0), Af[i]);
      EXPECT_EQ(A(0,i), A[i]);
  }
}


TEST(DenseTest, Copying) {
  hicma::initialize();
  int64_t n = 42;

  Dense<double> A(RandomNormalKernel<double>(), n, n);
  Dense<float> Af(RandomNormalKernel<double>(3), n, n);
  Dense<double> B(n, n);
  Dense<float> Bf(n, n);
  
  // same datatype
  A.copy_to(B);
  Af.copy_to(Bf);
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      ASSERT_EQ(A(i, j), B(i, j));
      ASSERT_EQ(Af(i, j), Bf(i, j));
    }
  }

  // different datatype
  A.copy_to(Bf);
  Af.copy_to(B);
  for (int64_t i=0; i<n; ++i) {
    for (int64_t j=0; j<n; ++j) {
      ASSERT_FLOAT_EQ(A(i, j), Bf(i, j));
      ASSERT_EQ(Af(i, j), B(i, j));
    }
  }

  // copy with offset
  Dense<double> C(30, 30);
  int64_t offset = 12;
  A.copy_to(C, offset, offset);
  for (int64_t i=0; i<C.dim[0]; ++i) {
    for (int64_t j=0; j<C.dim[1]; ++j) {
      ASSERT_EQ(A(offset+i, offset+j), C(i, j));
    }
  }

  // check for non-shallowness
  A.copy_to(B);
  double value = 23;
  B(0,0) = value;
  ASSERT_NE(A(0,0), value);

  // shallow_copy
  Dense<float> Cf = Bf.shallow_copy();
  for (int64_t i=0; i<C.dim[0]; ++i) {
    for (int64_t j=0; j<C.dim[1]; ++j) {
      ASSERT_EQ(Bf(i, j), Cf(i, j));
    }
  }
  Cf(0,0) = value;
  ASSERT_EQ(Bf(0,0), value);
}


TEST(DenseTest, Split) {
  int64_t n = 40;
  float value = -7;

  // no copy
  Dense<float> Af(RandomUniformKernel<double>(), n, n);
  std::vector<Dense<float>> submatrices_f = Af.split(3, 3);
  for (size_t k=0; k<submatrices_f.size(); ++k) {
    EXPECT_EQ(submatrices_f[k].dim[0], k<6?14:12);
    EXPECT_EQ(submatrices_f[k].dim[1], (k+1)%3?14:12);
    for (int64_t i=0; i<submatrices_f[k].dim[0]; ++i) {
      for (int64_t j=0; j<submatrices_f[k].dim[1]; ++j) {
        EXPECT_EQ(submatrices_f[k](i,j), Af((k/3)*14 + i, (k%3)*14 + j));
      }
    }
  }
  submatrices_f[0](0,0) = value;
  EXPECT_EQ(Af(0,0), value);

  // copy
  n = 30;
  Dense<double> A(RandomUniformKernel<double>(), n, n);
  std::vector<Dense<double>> submatrices = A.split(3, 3, true);
  for (size_t k=0; k<submatrices.size(); ++k) {
    EXPECT_EQ(submatrices[k].dim[0], 10);
    EXPECT_EQ(submatrices[k].dim[1], 10);
    for (int64_t i=0; i<submatrices[k].dim[0]; ++i) {
      for (int64_t j=0; j<submatrices[k].dim[1]; ++j) {
        EXPECT_EQ(submatrices[k](i,j), A((k/3)*10 + i, (k%3)*10 + j));
      }
    }
  }
  submatrices[0](0,0) = value;
  EXPECT_NE(A(0,0), value);
}


TEST(DenseTest, IsSubmatrix) {
  int64_t n = 64;
  Dense<double> A(n, n);
  std::vector<Dense<double>> submatrices = A.split(2, 2);
  EXPECT_FALSE(A.is_submatrix());
  for (size_t i=0; i<submatrices.size(); ++i)
    EXPECT_TRUE(submatrices[i].is_submatrix());
}

// TODO remove
/*
TEST(DenseTest, Id) {
  Dense<double> A;
  Dense<double> B;
  EXPECT_NE(A.id(), B.id());
  Dense<double> C = A.shallow_copy();
  EXPECT_EQ(A.id(), C.id());
}
*/


// TODO move
// this tests the dense split in the broader context of the general split function
// would prefer a more specific test here
TEST(DenseTest, Split1DTest) {
  hicma::initialize();
  int64_t N = 128;
  int64_t nblocks = 2;
  int64_t nleaf = N / nblocks;
  Dense D(random_uniform, std::vector<std::vector<double>>(), N, N);
  Hierarchical DH = split<double>(D, nblocks, nblocks);
  Hierarchical DH_copy = split<double>(D, nblocks, nblocks, true);
    
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

  Hierarchical H = split<double>(D, nblocks, nblocks);
  Dense HD(H);
  Dense Q(HD.dim[0], HD.dim[1]);
  Dense R(HD.dim[1], HD.dim[1]);
  qr(HD, Q, R);
  Dense QR = gemm(Q, R);
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
  Dense test1 = gemm(row, col);
  test1 *= 2;

  Hierarchical colH = split<double>(col, nblocks, 1);
  Hierarchical rowH = split<double>(row, 1, nblocks);
  Dense test2 = gemm(rowH, colH);
  test2 *= 2;
  for (int64_t i=0; i<nleaf; ++i) {
    for (int64_t j=0; j<nleaf; ++j) {
      ASSERT_FLOAT_EQ(test1(i, j), test2(i, j));
    }
  }
}
// TODO move, this is not a function of dense
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
