### Immediate

### Features to implement
* Regression testing
    * Clearly defined benchmarking suite
    * Separate repo with HiCMA as submodule
    * Execute on TSUBAME, ABCI etc
    * Automatic figure generation
    * Simple website for viewing? (Highchart, free for research projects)
* Testing
* Documentation
* RSVD with controllable precision on CPU
* Tile LU
* GPU LU
* H^2-matrix LU (nested basis)
* MPI version
* Asynchronous communication
* Template over float/double
* Wrappers that define +,-,*,/ operators
* Pivots inside diagonal blocks
* Non-square matrices
* QR decomposition
* Sparse matrix
* Proper assertions in all operations (catch errors early)

### Results to show
* Compare {FullRank, LowRank, Hierarchical}, {A, L, U} types with op. overloading
* Compare {BLR, HSS, H-matrix, H2-matrix}, vary {rank, N, procs}
* Check load balance, breakdown, vary {block size, partition size}
* Asynchronous communication
* Compare with other codes
