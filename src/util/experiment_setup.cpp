#include "hicma/util/experiment_setup.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>
#include <time.h>


namespace hicma
{

std::vector<double> get_sorted_random_vector(int64_t N, bool deterministic, unsigned int seed) {
  std::vector<double> randx(N);
  std::random_device rd;
  std::mt19937 gen(rd());
  // TODO Remove random seed when experiments end
  if (deterministic)
    gen.seed(seed);
  else
    gen.seed(time(NULL));
  
  // uniform distribution between 0 and 1
  std::uniform_real_distribution<> dis(0, 1);
  for (double& x : randx) x = dis(gen);
  std::sort(randx.begin(), randx.end());
  return randx;
}

std::vector<double> get_non_negative_vector(int64_t N) {
  std::vector<double> randx(N);
  for (int i = 0; i < N; ++i) {
    randx[i] = i;
  }
  return randx;
}

} // namespace hicma
