[Immediate]
Assume ni == nj
Get timers to work across compile units

[Development]
H-matrix LU
MPI version
Asyncronous communication

[After SIAM]
Clean randomized id.cpp
typedef any_cast
H^2-matrix LU
ID with controllable precision
Template over float/double
Pivots inside diagonal blocks
Non-square matrices

[Results]
Compare {FullRank, LowRank, Hierarchical}, {A, L, U} types with op. overloading
Compare {BLR, HSS, H-matrix, H2-matrix}, vary {rank, N, procs}
Check load balance, breakdown, vary {block size, partition size}
Asyncronous communication
Compare with other codes