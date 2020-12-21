#ifndef hicma_util_experiment_setup_h
#define hicma_util_experiment_setup_h

#include <cstdint>
#include <vector>


namespace hicma
{

std::vector<double> get_non_negative_vector(int64_t N);

std::vector<double> get_sorted_random_vector(int64_t N, bool deterministic=true, unsigned int seed=0);

} // namespace hicma

#endif // hicma_util_experiment_setup_h
