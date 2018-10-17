[Immediate]
Use Dense.svd in print
Always include headers that are used in the file

[Features to implement]
RSVD with controllable precision on CPU
GPU LU
H^2-matrix LU (nested basis)
MPI version
Asyncronous communication
Template over float/double
Wrappers that define +,-,*,/ operators
Pivots inside diagonal blocks
Non-square matrices

[Results to show]
Compare {FullRank, LowRank, Hierarchical}, {A, L, U} types with op. overloading
Compare {BLR, HSS, H-matrix, H2-matrix}, vary {rank, N, procs}
Check load balance, breakdown, vary {block size, partition size}
Asyncronous communication
Compare with other codes
