#include <cstdint>
#include <tuple>
#include <string>
#include <vector>

#include "hicma/hicma.h"
#include "gtest/gtest.h"


using namespace hicma;

class RSVDTests : public testing::TestWithParam<std::tuple<int64_t, int64_t>> {};

TEST_P(RSVDTests, randomizedSVD) {
  int64_t n, rank;
  std::tie(n, rank) = GetParam();

  hicma::initialize();
  std::vector<std::vector<double>> randx_A{hicma::get_sorted_random_vector(2*n)};
  hicma::Dense A(hicma::laplacend, randx_A, n, n, 0, n);

  LowRank LR(A, rank);
  double error = l2_error(A, LR);
  EXPECT_LT(error, 1e-8);
}

INSTANTIATE_TEST_SUITE_P(
    Low_Rank, RSVDTests,
    testing::Values(std::make_tuple(2048, 16)),    
    [](const testing::TestParamInfo<RSVDTests::ParamType>& info) {
      std::string name = ("n" + std::to_string(std::get<0>(info.param)) + "rank" +
                          std::to_string(std::get<1>(info.param)));
      return name;
    });
/*
int main() {
  hicma::initialize();
  int64_t N = 2048;
  int64_t rank = 16;

  timing::start("Init matrix");
  std::vector<std::vector<double>> randx{get_sorted_random_vector(2*N)};
  Dense D(laplacend, randx, N, N, 0, N);
  timing::stopAndPrint("Init matrix");

  print("RSVD");
  timing::start("Randomized SVD");
  LowRank LR(D, rank);
  timing::stopAndPrint("Randomized SVD", 2);
  print("Rel. L2 Error", l2_error(D, LR), false);

  print("ID");
  Dense U, S, V;
  timing::start("ID");
  Dense Dwork(D);
  std::tie(U, S, V) = id(Dwork, rank);
  timing::stopAndPrint("ID", 2);
  Dense test = gemm(gemm(U, S), V);
  print("Rel. L2 Error", l2_error(D, test), false);

  print("RID");
  timing::start("Randomized ID");
  std::tie(U, S, V) = rid(D, rank+5, rank);
  timing::stopAndPrint("Randomized ID", 2);
  test = gemm(gemm(U, S), V);
  print("Rel. L2 Error", l2_error(D, test), false);
  return 0;
}*/
