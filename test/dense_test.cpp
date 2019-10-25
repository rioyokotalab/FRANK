#include "hicma/dense.h"
#include "hicma/hierarchical.h"
#include "hicma/functions.h"

#include "gtest/gtest.h"

#include "yorel/multi_methods.hpp"

using namespace hicma;

TEST(DenseTest, ContructorHierarchical) {
    yorel::multi_methods::initialize();
    // Check whether the Dense(Hierarchical) constructor works correctly.
    int N = 128;
    int nblocks = 4;
    int nleaf = N / nblocks;
    std::vector<double> randx(N);
    for (int i=0; i<N; i++) {
        randx[i] = drand48();
    }
    // Construct single level all-dense hierarchical
    Hierarchical H(random, randx, N, N, 0, nleaf, nblocks, nblocks, nblocks);
    Dense D(H);
    // Check block-by-block and element-by-element if values match
    for (int ib=0; ib<nblocks; ++ib) {
        for (int jb=0; jb<nblocks; ++jb) {
            Dense D_compare = Dense(H(ib, jb));
            for (int i=0; i<nleaf; ++i) {
                for (int j=0; j<nleaf; ++j) {
                    ASSERT_EQ(D(nleaf*ib+i, nleaf*jb+j), D_compare(i, j));
                }
            }
        }
    }
}
