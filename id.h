#ifndef id_h
#define id_h

#include "dense.h"

#include <random>
#include <vector>
#include <cblas.h>
#include <lapacke.h>

namespace hicma {
  void rsvd(
            const Dense& A,
            int rank,
            Dense& U,
            Dense& S,
            Dense& V
            );
}
#endif
