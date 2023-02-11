#ifndef hicma_util_experiment_setup_h
#define hicma_util_experiment_setup_h

#include <cstdint>
#include <vector>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

template<typename U = double>
std::vector<U> get_sorted_random_vector(int64_t N, int seed=0);

template<typename U = double>
std::vector<U> get_non_negative_vector(int64_t N);

template<typename U = double>
std::vector<std::vector<U>> get_circular_coords(int64_t N, int ndim=2);

template<typename U = double>
std::vector<std::vector<U>> get_rectangular_coords(int64_t N, int ndim=2);

template<typename U = double>
std::vector<std::vector<U>> get_rectangular_coords_rand(int64_t N, int ndim=2);

} // namespace hicma

#endif // hicma_util_experiment_setup_h
