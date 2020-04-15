#ifndef hicma_oprations_randomized_factorizations_h
#define hicma_oprations_randomized_factorizations_h

#include <cstdint>
#include <tuple>
#include <vector>


namespace hicma
{

class Dense;

std::tuple<Dense, std::vector<int64_t>> one_sided_rid(
  const Dense&, int64_t sample_size, int64_t rank, bool transA=false);

std::tuple<Dense, Dense, Dense> rid(
  const Dense&, int64_t sample_size, int64_t rank);

std::tuple<Dense, Dense, Dense> rsvd(const Dense&, int64_t sample_size);

std::tuple<Dense, Dense, Dense> old_rsvd(const Dense&, int64_t sample_size);

} // namespace hicma

#endif // hicma_oprations_randomized_factorizations_h
