#include "hicma/util/experiment_setup.h"

#include <algorithm>
#include <random>
#include <vector>

namespace hicma
{

std::vector<double> get_sorted_random_vector(unsigned int N) {
  std::vector<double> randx(N);
  std::random_device rd;
  std::mt19937 gen(rd());
  // TODO Remove random seed when experiments end
  gen.seed(0);
  // uniform distribution between 0 and 1
  std::uniform_real_distribution<> dis(0, 1);
  for (double& x : randx) x = dis(gen);
  std::sort(randx.begin(), randx.end());
  return randx;
}

} // namespace hicma
