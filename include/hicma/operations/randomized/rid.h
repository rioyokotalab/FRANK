#ifndef hicma_operations_randomized_rid_h
#define hicma_operations_randomized_rid_h

#include <tuple>
#include <vector>

namespace hicma
{

class Dense;

std::tuple<Dense, Dense, Dense> rid(const Dense&, int sample_size, int rank);

std::tuple<Dense, std::vector<int>> one_sided_rid(const Dense&, int sample_size, int rank);

} // namespace hicma

#endif // hicma_operations_randomized_rid_h
