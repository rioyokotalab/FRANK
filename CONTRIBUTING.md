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
We don't want our code to depend on boost.
No way of inferring the correct overloaded function for each type.
Which resulted in many if statements for each type.
For GEMM it has three nested if statement.
All arguments including output were passed as variables to all functions.

## Idea 4: Use a shared_ptr to Node
#### Implementation
```c++
std::vector<std::shared_ptr<Node>> data;
Node& operator(i,j) {data[2*i+j]};

enum{
  FRANK_NODE;
  FRANK_DENSE;
  FRANK_LOWRANK;
  FRANK_HIERARCHIAL;
}
Hierarchical::is(const int enum_id) {
  return enum_id == FRANK_HIERARCHICAL;
}
```
#### Features
This solved the dependency on boost.

Inferring the correct overloaded function for each type is now done through an is() function.

Now you could return Dense, LowRank, Hierarchical types from functions as shared_ptr<Node>.

#### Issues with this approach
We still have many if statements for each type (although it uses is() now).

We cannot operate on the return values as Dense, LowRank, Hierarchical types because they are shared_ptr<Node> type.

Following assignments no longer work:
```c++
Dense A;
Hierarchical H;
H(0,0) = A(0,0);
H(0,1) = LowRank(A(0,1));
H(1,0) = LowRank(A(1,0));
H(1,1) = A(1,1);
```

## Idea 5: Subclass the shared_ptr class as a BlockPtr class
#### Implementation
```c++
template<Typename T = Node>
class BlockPtr : public std::shared_ptr<T> {
  void getrf();
  void trsm(const Node&, const char&);
  void gemm(const Node&, const Node&);
}
typedef std::vector<BlockPtr> NodePtr;
NodePtr data;
NodePtr operator(i,j) {data[2*i+j]};

enum{
  FRANK_NODE;
  FRANK_DENSE;
  FRANK_LOWRANK;
  FRANK_HIERARCHIAL;
}
Hierarchical::is(const int enum_id) {
  return enum_id == FRANK_HIERARCHICAL;
}
```
#### Features
By subclassing the shared_ptr class we can define functions within the subclass BlockPtr,
which then allows us operate on the return values of Dense, LowRank, Hierarchical types.

Following assignments are possible again:
```c++
Dense A;
Hierarchical H;
H(0,0) = A(0,0);
H(0,1) = LowRank(A(0,1));
H(1,0) = LowRank(A(1,0));
H(1,1) = A(1,1);
```

#### Issues with this approach
Data ownership of data for the BlockPtr is not clear.

## Idea 6: Create a class Any (formerly Block) which contains a unique_ptr<Node>
```c++
class Any {
  std::unique_ptr<Node> ptr;
}
std::vector<Any> data;

const Node& operator(i,j) {data[2*i+j]};
Any& operator(i,j) {data[2*i+j]};

enum{
  FRANK_NODE;
  FRANK_DENSE;
  FRANK_LOWRANK;
  FRANK_HIERARCHIAL;
}
Hierarchical::is(const int enum_id) {
  return enum_id == FRANK_HIERARCHICAL;
}
```

#### Features
Has almost the same functionality as BlockPtr, but a little bit cleaner.

#### Issues with this approach
All functions {gemm, trsm, etc.} are defined in all classes {Dense, LowRank, etc.}

Using if checks in all functions {gemm, trsm, etc.} to determine the right combination of classes {Dense, LowRank, etc.}

## Idea 7: Add multimethods through YOMM. Define functions {gemm, trsm, etc.} outside of the classes {Dense, LowRank, etc.}.
```c++
class Any {
  std::unique_ptr<Node> ptr;
}
std::vector<Any> data;

const Node& operator(i,j) {data[2*i+j]};
Any& operator(i,j) {data[2*i+j]};

void getrf(Any& A) {
  getrf_omm(*A.ptr.get());
}
void getrf(Node& A) {
  getrf_omm(A);
}
```

#### Features
With multimethods we no longer need checks for classes {Dense, LowRank, etc.}.

#### Issues with this approach
*A.ptr.get() is ugly and pointer is exposed

## Minor fix
#### Name change
Any is now MatrixProxy
Node is now Matrix

#### Private pointers
```c++
class MatrixProxy {
private:
  std::unique_ptr<Matrix> ptr;
}
```

#### Conversion operators so that MatrixProxy is automatically converted to Matrix
```c++
operator const Matrix&() const;
operator Matrix&();
```
