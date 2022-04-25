# FRANK (Factorization of RANK structured matrices)

“Any intelligent fool can make things bigger and more complex.
It takes a touch of genius - and a lot of courage to move in the
opposite direction.” - Albert Einstein

Library for hierarchical low-rank matrix factorizations.

# How to build and make
```
mkdir build
cd build
cmake -DUSE_MKL=ON ..
make
```
binaries will be compiled into bin. C++17 is required in order to build FRANK. The build process will try to detect or download the following dependencies:
- nlohmann_json
- YOMM2


# Build flags
```
-DUSE_MKL
```
If this is specified, FRANK will try to use the Intel MKL library on your system. Otherwise it will try to detect the default BLAS and LAPACK libraries installed on your system.

```
-DBUILD_DOCS
```
If this is set, the documentation files are created during the build process. Requires `Doxygen` installed on the system. Default to OFF.

```
-DBUILD_TESTS
```
If this is set, test files will be included in the build process. Default to ON.

```
-DBUILD_EXAMPLES
```
If this is set, examples files will be included in the build process. Default to ON.

