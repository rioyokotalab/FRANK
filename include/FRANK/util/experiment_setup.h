#ifndef FRANK_util_experiment_setup_h
#define FRANK_util_experiment_setup_h

#include <cstdint>
#include <vector>


/**
 * @brief General namespace of the FRANK library
 */
namespace FRANK
{

/**
 * @brief Create a sorted vector of random numbers
 * 
 * Double values are generated from an unifrom distribution (from 0 to 1)
 * and sorted in ascending order.
 * 
 * @param N size of the resulting vector (i.e. number of values)
 * @return std::vector<double> sorted vector of random numbers
 */
std::vector<double> get_sorted_random_vector(const int64_t N);

/**
 * @brief Create a vector of non-negative numbers
 * 
 * The vector is filled with the numbers from 0 to N.
 * 
 * @param N size of the resulting vector (i.e. number of values)
 * @return std::vector<double> vector of non-negative numbers
 */
std::vector<double> get_non_negative_vector(const int64_t N);
 
} // namespace FRANK

#endif // FRANK_util_experiment_setup_h
