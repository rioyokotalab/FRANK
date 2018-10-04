Use Dense(LR) instead of LR.dense()

Clean randomized id.cpp
H^2-matrix LU (nested basis)
Recompress low_rank after add/sub (after H^2 mat)
ID with controllable precision
Pivots inside diagonal blocks
Non-square matrices
Write operator-=() using operator+=()

[Remove dependency on boost any]
Make a few realistic test cases in prototype (return types of add etc)
Get Hierarchical constructor working (commit to new branch if working)

[Features to implement]
MPI version
Asyncronous communication
Template over float/double

[Results]
Compare {FullRank, LowRank, Hierarchical}, {A, L, U} types with op. overloading
Compare {BLR, HSS, H-matrix, H2-matrix}, vary {rank, N, procs}
Check load balance, breakdown, vary {block size, partition size}
Asyncronous communication
Compare with other codes
