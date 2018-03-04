[Development]
Use BLAS, LAPACK in Dense class
MPI version
Asyncronous communication
ID with controllable precision
Template over float/double
Pivots inside diagonal blocks
Const correctness
Non-square matrices

[Results]
Compare {FullRank, LowRank, Hierarchical}, {A, L, U} types with op. overloading
Compare {BLR, HSS, H-matrix, H2-matrix}, vary {rank, N, procs}
Check load balance, breakdown, vary {block size, partition size}
Asyncronous communication
Compare with other codes