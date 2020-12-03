# HiCMA

Library for hierarchical low-rank matrix operations that calls kblas.

# Classes and their details
```c++
struct Node {
 int i_abs,j_abs,level;
}

struct Dense : public Node {
 int dim[2];
 std::vector<T> D(dim[0]*dim[1]);
 operator(i,j) {D[i*dim[1]+j];}
 resize(m,n) {dim[0]=m;dim[1]=n;D.resize(m*n);}
}

struct LowRank : public Node {
 int dim[2];
 int rank;
 Dense U; // (R for nested)
 Dense S;
 Dense V; // (W for nested)
}

struct Hierarchical : public Node {
  2 x 2 matrix of mixed data types
  Implementation discussed below in 1.2, 1.3
  For example, let D : Dense, L: LowRank, H: Hierarchical
  Weak admissibility
   --               --
   | H[0][0] L[1][0] |
   | L[0][1] H[1][1] |
   --               --
  Weak admissibility leaf
   --               --
   | D[0][0] L[1][0] |
   | L[0][1] D[1][1] |
   --               --
  Strong admissibility
   --               --
   | H[0][0] H[1][0] |
   | H[0][1] H[1][1] |
   --               --
  Strong admissibility diagonal leaf
   --               --
   | D[0][0] D[1][0] |
   | D[0][1] D[1][1] |
   --               --
  Strong admissibility off-diagonal leaf
   --               --
   | L[0][0] L[1][0] |
   | L[0][1] L[1][1] |
   --               --
}
```
# Design decisions for the Hierarchical class

## Idea 1: Use pointer to base class (doesn't work)
#### Implementation
Hierarchical has
```c++
std::vector<Node*> data[4];
Node* operator(i,j) {data[2*i+j]};
```
#### Example usage
```c++
Hierarchical H;
H(0,0) = new Dense;
H(0,1) = new LowRank;
H(1,0) = new LowRank;
H(1,1) = new Dense;
```
### Issues with this approach
Accessing `Dense*` and `LowRank*` elements using the above method requires all functions of
Dense and LowRank to be defined as virtual functions in the base class Node*
However, since we want to overload `operator*(Dense)` in all three classes we will need to define:
```c++
virtual Dense operator*(Dense)
virtual LowRank operator*(Dense)
virtual Hierarchical operator*(Dense)
```
in the base Node class.
This is not allowed since it will be operator overloading for the same input with different output

## Idea 2: Use struct/union (doesn't work)
#### Implementation
```c++
struct/union DLH {
  Dense D;
  LowRank L;
  Hierarchical H;
};
```
Hierarchical has
```c++
std::vector<DLH> data[4];`
DLH operator(i,j) {data[2*i+j]};
```
#### Example usage
```c++
Hierarchical H;
H(0,0).D = A(0,0);
H(0,1).L = LowRank(A(0,1));
H(1,0).L = LowRank(A(1,0));
H(1,1).D = A(1,1);
```
#### Issues with this approach
What we want to do during the hierarchical matrix operation is to call operations like
```c++
H(1,1) -= H(1,0) * H(0,1)
```
without knowing what type they are
This approach requires .D, .L, .H to be specified and will not work

## Idea 3: Use boost::any (works, but not elegant)
#### Implementation
```c++
std::vector<boost::any> data;
boost::any& operator(i,j) {data[2*i+j]};
```
#### Example usage
```c++
Dense A;
Hierarchical H;
H(0,0) = A(0,0);
H(0,1) = LowRank(A(0,1));
H(1,0) = LowRank(A(1,0));
H(1,1) = A(1,1);
```
#### Issues with this approach
What we want to do during the hierarchical matrix operation is to distinguish operations like
```c++
L(1,1)=gemm(H(1,0),H(0,1))
H(1,1)=gemm(H(1,0),H(0,1))
```
However, since the overloaded function gemm(Hierarchical, Hierarchical) does not distinguish between return types,
we cannot automatically call two different gemm functions with different return types.

## Idea 4: Create a wrapper class Any ourselves (current implementation)
#### Implementation
Any owns a `unique_ptr<Node>` and forwards the getrf, trsm and gemm calls to
the object pointed to (which may be `Dense`, `LowRank` or `Hierarchical`).
An advantage of this approach is that the indirection through `Any` allows us
to efficiently change what type of matrix is held at a certain index in
`Hierarchical`. Say for example `H(0, 1)` is a `Dense` object, but we want to
compress it to a `LowRank` object. We can call
```c++
H(0, 1) = LowRank(H(0, 1));
```
and the operation can happen without unnecessart data copies due to the c++11
move semantics.\
Another big advantage is the usage of `unique_ptr<Node>`, which allows for
clear ownership of objects and ensures leak-free code.
#### Issues with this approach
The classes `Any` and `Node` both fulfill the purpose of fascilitating runtime
polymorphism for `Dense`, `LowRank` and `Hierarchical`. Uniting them into one
class would be desirable, but is likely not possible.\
Another issue is that some functions may need an additional interface taking in
`Any` objects and forwarding to the `Node` they hold.

# Parallel paradigms
