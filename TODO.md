[Immediate]
H gemm
typedef any_cast
Remove dense() in Hierarchical
Hierarchical matvec, assign scalar

[Development]
Tree construction
HODLR
H-matrix
H^2-matrix
MPI version
Asyncronous communication
ID with controllable precision
Template over float/double
Pivots inside diagonal blocks
Non-square matrices

[After SIAM]
Clean randomized id.cpp

[Results]
Compare {FullRank, LowRank, Hierarchical}, {A, L, U} types with op. overloading
Compare {BLR, HSS, H-matrix, H2-matrix}, vary {rank, N, procs}
Check load balance, breakdown, vary {block size, partition size}
Asyncronous communication
Compare with other codes