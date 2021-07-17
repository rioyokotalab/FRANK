#ifndef hicma_util_experiment_setup_h
#define hicma_util_experiment_setup_h

#include <cstdint>
#include <vector>


namespace hicma
{

std::vector<float> get_sorted_random_vector(int64_t N);
std::vector<float> get_non_negative_vector(int64_t N);
 
} // namespace hicma

#endif // hicma_util_experiment_setup_h
