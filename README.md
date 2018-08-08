# Hicma

Library for distributed LU decomposition of hierarchical matrices.

# Design decisions

## Parallel paradigms

## Classes and their details

1. Choice of data structure for H-matrices  
1.1 Basic data types  
struct Node {  
 int i,j,level;  
}  
  
struct Dense : public Node {  
 int dim[2];  
 std::vector<T> D(dim[0],dim[1]);  
 operator(i,j) {D[i*dim[1]+j];}  
 resize(m,n) {dim[0]=m;dim[1]=n;D.resize(m*n);}  
}  
  
struct LowRank : public Node {  
 int dim[2];  
 int rank;  
 Dense B;  
 Dense U; (R for nested)  
 Dense V; (W for nested)  
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
  
1.2 Idea 1: Use pointer to base class (doesn't work)  
[Implementation]  
 Hierarchical has std::vector<Node*> data[4];  
 Hierarchical operator(i,j) {data[2*i+j]};  
[Example usage]  
 Hierarchical H;  
 H(0,0) = new Dense;  
 H(0,1) = new LowRank;  
 H(1,0) = new LowRank;  
 H(1,1) = new Dense;  
[Issues with this approach]  
 Accessing Dense* and LowRank* elements using the above method requires all functions of  
 Dense and LowRank to be defined as virtual functions in the base class Node*  
 However, since we want to overload operator*(Dense) in all three classes we will need to define:  
 virtual Dense operator*(Dense)  
 virtual LowRank operator*(Dense)  
 virtual Hierarchical operator*(Dense)  
 in the base Node class  
 This is not allowed since it will be operator overloading for the same input with different output  
  
1.3 Idea 2: Use struct/union (doesn't work)  
[Implementation]  
 struct/union DLH {  
  Dense D;  
  LowRank L;  
  Hierarchical H;  
 };  
 Hierarchical has std::vector<DLH> data[4];  
 Hierarchical operator(i,j) {data[2*i+j]};  
[Example usage]  
 Hierarchical H;  
 H(0,0).D = A(0,0);  
 H(0,1).L = LowRank(A(0,1));  
 H(1,0).L = LowRank(A(1,0));  
 H(1,1).D = A(1,1);  
[Issues with this approach]  
 What we want to do during the hierarchical matrix operation is to call operations like  
 H(1,1) -= H(1,0) * H(0,1)
 without knowing what type they are
 This approach requires .D, .L, .H to be specified and will not work

1.4 Idea 3: Use boost::variant
[Implementation]
