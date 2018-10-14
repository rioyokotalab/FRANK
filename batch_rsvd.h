#ifndef batch_rsvd_h
#define batch_rsvd_h

#include "dense.h"

namespace hicma {

  extern bool useBatch;
  extern std::vector<int> h_m;
  extern std::vector<int> h_n;
  extern std::vector<Dense> vecA;
  extern std::vector<Any*> vecLR;

  void batch_rsvd();
}

#endif
