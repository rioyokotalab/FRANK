#ifndef hicma_util_global_key_value_h
#define hicma_util_global_key_value_h

#include <string>

namespace hicma {

std::string getGlobalValue(std::string key);

void setGlobalValue(std::string key, std::string value);

void add_gemm_flops(int64_t m, int64_t n, int64_t k);
void add_trsm_flops(int64_t m, int64_t n, int lr);
void add_getrf_flops(int64_t m, int64_t n);
void add_geqrf_flops(int64_t m, int64_t n);
void add_orgqr_flops(int64_t m, int64_t n, int64_t k);
void add_gerqf_flops(int64_t m, int64_t n);
void add_orgrq_flops(int64_t m, int64_t n, int64_t k);
void add_svd_flops(int64_t m, int64_t n);
void add_plus_flops(int64_t m, int64_t n);

void reset_flops();

void print_flops();

int64_t get_flops();

} // namespace hicma

#endif // hicma_util_global_key_value_h
