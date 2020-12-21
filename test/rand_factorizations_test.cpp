#include "hicma/hicma.h"

#include "gtest/gtest.h"

//#include <cstdint>
#include <vector>


using namespace hicma;

class RandomizedFactorizationsTest : public ::testing::Test{
    protected:
    void SetUp() override{
        hicma::initialize();
        int64_t n = 100;
        std::vector<std::vector<double>> randx{get_sorted_random_vector(2*n, true, 0)};
        Dense D(laplacend, randx, n, n, 0, n);
        A=D;
    }

    Dense A;
};

TEST_F(RandomizedFactorizationsTest, RSVD) {
    ASSERT_FLOAT_EQ(0, l2_error(A,A));
    Dense U, S, V;
    //TODO ensure deterministic RSVD
    std::tie(U, S, V) = rsvd(A, 16);
    double error_r16 = l2_error(A, gemm(gemm(U,S),V));
    EXPECT_LE(error_r16, 1e-12);
    std::tie(U, S, V) = rsvd(A, 17);
    double error_r17 = l2_error(A, gemm(gemm(U,S),V));
    EXPECT_LT(error_r17, error_r16);
    std::tie(U, S, V) = rsvd(A, 15);
    double error_r15 = l2_error(A, gemm(gemm(U,S),V));
    EXPECT_LT(error_r16, error_r15);
}


TEST_F(RandomizedFactorizationsTest, RID) {
    Dense U, S, V;
    //TODO ensure deterministic RID
    std::tie(U, S, V) = rid(A, 21, 16);
    double error_r16 = l2_error(A, gemm(gemm(U,S),V));
    EXPECT_LE(error_r16, 1e-13);
    std::tie(U, S, V) = rid(A, 22, 17);
    double error_r17 = l2_error(A, gemm(gemm(U,S),V));
    EXPECT_LT(error_r17, error_r16);
    std::tie(U, S, V) = rid(A, 20, 15);
    double error_r15 = l2_error(A, gemm(gemm(U,S),V));
    EXPECT_LT(error_r16, error_r15);
}

TEST_F(RandomizedFactorizationsTest, column_RID) {
    Dense V, Ap;
    std::vector<int64_t> p;
    //TODO ensure deterministic one-sided RID
    std::tie(V, p) = one_sided_rid(A, 21, 16, false);
    Ap = get_cols(A, p);
    double error_r16 = l2_error(A, gemm(Ap, V));
    EXPECT_LE(error_r16, 1e-13);
    std::tie(V, p) = one_sided_rid(A, 22, 17, false);
    Ap = get_cols(A, p);
    double error_r17 = l2_error(A, gemm(Ap, V));
    EXPECT_LT(error_r17, error_r16);
    std::tie(V, p) = one_sided_rid(A, 20, 15, false);
    Ap = get_cols(A, p);
    double error_r15 = l2_error(A, gemm(Ap, V));
    EXPECT_LT(error_r16, error_r15);
}

TEST_F(RandomizedFactorizationsTest, row_RID) {
    Dense V, Ap;
    std::vector<int64_t> p;
    //TODO ensure deterministic one-sided RID
    std::tie(V, p) = one_sided_rid(A, 21, 16, true);
    Ap = get_rows(A, p);
    double error_r16 = l2_error(A, gemm(V, Ap, 1, true, false));
    EXPECT_LE(error_r16, 1e-13);
    std::tie(V, p) = one_sided_rid(A, 22, 17, true);
    Ap = get_rows(A, p);
    double error_r17 = l2_error(A, gemm(V, Ap, 1, true, false));
    EXPECT_LT(error_r17, error_r16);
    std::tie(V, p) = one_sided_rid(A, 20, 15, true);
    Ap = get_rows(A, p);
    double error_r15 = l2_error(A, gemm(V, Ap, 1, true, false));
    EXPECT_LT(error_r16, error_r15);
}