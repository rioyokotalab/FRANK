#ifndef hicma_oprations_randomized_factorizations_h
#define hicma_oprations_randomized_factorizations_h

#include <cstdint>
#include <tuple>
#include <vector>


namespace hicma
{

class Dense;

std::tuple<Dense, Dense> aca(const Dense& A, int64_t rank);

std::tuple<Dense, Dense> aca_complete(const Dense& A, int64_t rank);

std::tuple<Dense, std::vector<int64_t>> one_sided_rid(
  const Dense&, int64_t sample_size, int64_t rank, bool transA=false);

std::tuple<Dense, std::vector<int64_t>> one_sided_rid_new(
  const Dense&, int64_t sample_size, int64_t rank, int64_t q=0, bool transA=false);

std::tuple<Dense, Dense, Dense> rid(
  const Dense&, int64_t sample_size, int64_t rank);

std::tuple<Dense, Dense, Dense> rid_new(
  const Dense&, int64_t sample_size, int64_t rank, int64_t q);

std::tuple<Dense, Dense, Dense> rsvd(const Dense&, int64_t sample_size);

std::tuple<Dense, Dense, Dense> old_rsvd(const Dense&, int64_t sample_size);

std::tuple<Dense, Dense, Dense> rsvd_pow(const Dense& A, int64_t sample_size, int64_t q);

std::tuple<Dense, Dense, Dense> rsvd_powOrtho(const Dense& A, int64_t sample_size, int64_t q);

std::tuple<Dense, Dense, Dense> rsvd_singlePass(const Dense& A, int64_t sample_size);

std::tuple<Dense, Dense, Dense> rrqr(const Dense&, int64_t rank);

std::tuple<Dense, Dense, Dense> rrlu(const Dense&, int64_t rank);

std::tuple<Dense, Dense, Dense> pqr_block(bool pivoting, const Dense& M, int64_t rank);

} // namespace hicma

#endif // hicma_oprations_randomized_factorizations_h
