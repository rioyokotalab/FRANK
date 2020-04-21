#ifndef hicma_operations_LAPACK_h
#define hicma_operations_LAPACK_h

#include <cstdint>
#include <tuple>
#include <vector>


namespace hicma
{

class Node;
class NodeProxy;
class Dense;

std::tuple<Dense, std::vector<int64_t>> geqp3(Node& A);

void geqrt(Node&, Node&);

void geqrt2(Dense&, Dense&);

std::tuple<NodeProxy, NodeProxy> getrf(Node&);

std::tuple<Dense, std::vector<int64_t>> one_sided_id(Node& A, int64_t k);

// TODO Does this need to be in the header?
Dense get_cols(const Dense& A, std::vector<int64_t> P);

std::tuple<Dense, Dense, Dense> id(Node& A, int64_t k);

void larfb(const Node&, const Node&, Node&, bool);

void latms(
  const char& dist,
  std::vector<int>& iseed,
  const char& sym,
  std::vector<double>& d,
  int mode,
  double cond,
  double dmax,
  int kl, int ku,
  const char& pack,
  Dense& A
);

void qr(Node&, Node&, Node&);

// TODO Does this need to be in the header?
bool need_split(const Node&);

// TODO Does this need to be in the header?
std::tuple<Dense, Dense> make_left_orthogonal(const Node&);

// TODO Does this need to be in the header?
void update_splitted_size(const Node&, int64_t&, int64_t&);

// TODO Does this need to be in the header?
NodeProxy split_by_column(const Node&, Node&, int64_t&);

// TODO Does this need to be in the header?
NodeProxy concat_columns(const Node&, const Node&, const Node&, int64_t&);

void zero_lowtri(Node&);

void zero_whole(Node&);

void rq(Node&, Node&, Node&);

std::tuple<Dense, Dense, Dense> svd(Dense& A);

std::tuple<Dense, Dense, Dense> sdd(Dense& A);

// TODO Does this need to be in the header?
Dense get_singular_values(Dense& A);

void tpmqrt(const Node&, const Node&, Node&, Node&, bool);

void tpqrt(Node&, Node&, Node&);

} // namespace hicma

#endif // hicma_operations_LAPACK_h
