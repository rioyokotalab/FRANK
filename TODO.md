[Immediate]
Change dense ops to low rank in BLR
Tree construction
HODLR
H-matrix
H^2-matrix

[Development]
MPI version
Asyncronous communication
ID with controllable precision
Template over float/double
Pivots inside diagonal blocks
Non-square matrices

[Minor issues]
dim[2] to m, n
Const correctness

[Results]
Compare {FullRank, LowRank, Hierarchical}, {A, L, U} types with op. overloading
Compare {BLR, HSS, H-matrix, H2-matrix}, vary {rank, N, procs}
Check load balance, breakdown, vary {block size, partition size}
Asyncronous communication
Compare with other codes