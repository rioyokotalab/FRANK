#include "hicma/util/global_key_value.h"
#include "hicma/operations/BLAS.h"

#include <cstdlib>
#include <map>
#include <iostream>
#include <cfenv>

namespace hicma {

std::map<std::string, std::string> globalKeyValue;

int64_t gemm_flops_lr = 0;
int64_t gemm_flops = 0;
int64_t getrf_flops = 0;
int64_t trsm_flops = 0;
int64_t geqrf_flops = 0;
int64_t orgqr_flops = 0;
int64_t gerqf_flops = 0;
int64_t orgrq_flops = 0;
int64_t svd_flops = 0;
int64_t plus_flops = 0;

std::string getGlobalValue(std::string key) {
  if(globalKeyValue.find(key) == globalKeyValue.end()) {
    if(std::getenv(key.c_str()) != nullptr)
      return std::string(std::getenv(key.c_str())); //Fallback to ENV Variable
    else
      return ""; //Return empty string if not found as well
  }
  else return globalKeyValue[key];
}

void setGlobalValue(std::string key, std::string value) {
  globalKeyValue[key] = value;
}

void check_overflow(std::string op) {
  if ((bool)std::fetestexcept(FE_OVERFLOW)) {
      std::cout<<"Overflow of FLOPS count during "<<op<<" detected"<<std::endl;
  }
}

void reset_flops() {
  trsm_flops = 0;
  gemm_flops = 0;
  getrf_flops = 0;
  geqrf_flops = 0;
  orgqr_flops = 0;
  gerqf_flops = 0;
  orgrq_flops = 0;
  svd_flops = 0;
  plus_flops = 0;
  gemm_flops_lr = 0;
}

void add_plus_flops(int64_t m, int64_t n) {
  plus_flops += m + n;
}

void add_gemm_flops(int64_t m, int64_t n, int64_t k) {
  //if (m == n && m == 256)
    gemm_flops += 2 * m * n * k;
  //else
    //gemm_flops_lr += 2 * m * n * k;
  //check_overflow("GEMM");
}

void add_getrf_flops(int64_t m, int64_t n) {
  getrf_flops += m*n*n - 1./3.*n*n*n -1./2.*n*n + 5./6.*n;
  //check_overflow("GETRF");
}

// constand taken from https://www.netlib.org/lapack/lug/node71.html
void add_svd_flops(int64_t m, int64_t n) {
  int64_t k = m;
  if (m != n) {
    // do additional qr
    int64_t flops;
    if (m > n) {
      k = n;
      flops = 2*m*n*n - 2./3.*n*n*n + m*n + n*n + 14./3.*n; 
      flops += 4*m*n*n - 2*(m+n)*n*n + 4./3.*n*n*n + 3*n*n - m*n - n*n - 4./3.*n;
    } else {
      k = m;
      flops = 2*m*m*n - 2./3.*m*m*m + 3*m*n - m*m + 14./3.*m;
      flops += 4*m*n*m - 2*(m+n)*m*m + 4./3.*m*m*m + 3*n*m - m*m - m*m - 4./3.*m;
    }
  svd_flops += flops;
  }
  svd_flops += 6.67*k*k*k;
}

void add_trsm_flops(int64_t m, int64_t n, int lr) {
  int64_t flops;
  if (lr == TRSM_LEFT) {
    flops = n*m*m;
  } else {
    flops = m*n*n;
  }
  //check_overflow("TRSM");
  trsm_flops += flops;
  //check_overflow("TRSM");
}

void add_geqrf_flops(int64_t m, int64_t n) {
  int64_t flops;
  if (m > n) {
    flops = 2*m*n*n - 2./3.*n*n*n + m*n + n*n + 14./3.*n; 
  } else {
    flops = 2*m*m*n - 2./3.*m*m*m + 3*m*n - m*m + 14./3.*m;
  }
  //check_overflow("GEQRF");
  geqrf_flops += flops;
  //check_overflow("GEQRF");
}

void add_gerqf_flops(int64_t m, int64_t n) {
  int64_t flops;
  if (m > n) {
    flops = 2*m*n*n - 2./3.*n*n*n + 2*m*n + n*n + 17./3.*n; 
  } else {
    flops = 2*m*m*n - 2./3.*m*m*m + 2*m*n - m*m + 17./3.*m;
  }
  //check_overflow("GERQF");
  gerqf_flops += flops;
  //check_overflow("GERQF");
}

void add_orgqr_flops(int64_t m, int64_t n, int64_t k) {
  orgqr_flops += 4*m*n*k - 2*(m+n)*k*k + 4./3.*k*k*k + 3*n*k - m*k - k*k - 4./3.*k;
  //check_overflow("ORGQR");
}

void add_orgrq_flops(int64_t m, int64_t n, int64_t k) {
  orgrq_flops += 4*m*n*k - 2*(m+n)*k*k + 4./3.*k*k*k + 2*m*k - k*k - 1./3.*k;
  //check_overflow("ORGRQ");
}

void print_flops() {
  std::cout<<"GETRF: "<<getrf_flops<<std::endl;
  std::cout<<"TRSM: "<<trsm_flops<<std::endl;
  std::cout<<"GEMM: "<<gemm_flops<<std::endl;
  std::cout<<"GEQRF: "<<geqrf_flops<<std::endl;
  std::cout<<"ORGQR: "<<orgqr_flops<<std::endl;
  std::cout<<"GERQF: "<<gerqf_flops<<std::endl;
  std::cout<<"ORGRQ: "<<orgrq_flops<<std::endl;
  std::cout<<"GRSVD: "<<svd_flops<<std::endl;
  std::cout<<"ADD: "<<plus_flops<<std::endl;
  std::cout<<"TOTAL: "<<get_flops()<<std::endl;
  std::cout<<std::endl;
}

int64_t get_flops() {
  int64_t flops = getrf_flops + trsm_flops + gemm_flops
    + geqrf_flops + orgqr_flops + gerqf_flops + orgrq_flops
    + svd_flops + plus_flops;
  return flops;
}

} // namespace hicma
