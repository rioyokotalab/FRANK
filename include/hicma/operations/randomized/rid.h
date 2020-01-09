#ifndef hicma_operations_randomized_rsvh_h
#define hicma_operations_randomized_rsvh_h

#include "yorel/multi_methods.hpp"
using yorel::multi_methods::virtual_;

#include <tuple>

namespace hicma
{

class Dense;

std::tuple<Dense, Dense, Dense> rid(const Dense&, int sample_size, int rank);

} // namespace hicma

#endif // hicma_operations_randomized_rsvh_h
