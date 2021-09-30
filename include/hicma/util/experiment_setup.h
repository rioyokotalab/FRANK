#ifndef hicma_util_experiment_setup_h
#define hicma_util_experiment_setup_h

#include <cstdint>
#include <vector>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

std::vector<double> get_sorted_random_vector(int64_t N);
std::vector<double> get_non_negative_vector(int64_t N);
 
} // namespace hicma

#endif // hicma_util_experiment_setup_h
