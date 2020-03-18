#include "hicma/hicma.h"

#include "gtest/gtest.h"

#include "yorel/yomm2/cute.hpp"

using namespace hicma;

TEST(DenseTest, ContructorHierarchical) {
    yorel::yomm2::update_methods();
    // Check whether the Dense(Hierarchical) constructor works correctly.
    int N = 128;
    int nblocks = 4;
    int nleaf = N / nblocks;
    std::vector<double> randx = get_sorted_random_vector(N);
    // Construct single level all-dense hierarchical
    Hierarchical H(random_uniform, randx, N, N, 0, nleaf, nblocks, nblocks, nblocks);
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
