#ifndef block_ptr
#define block_ptr
#include <iostream>
#include <memory>

namespace hicma {

  template <class T>
  struct return_type{ typedef T type; };

  // Specializations for _Dense
  class _Dense;
  template <>
  struct return_type<_Dense>{ typedef double type; };

  template <typename T>
  class BlockPtr : public std::shared_ptr<T> {
  public:
    template <typename>
    friend class BlockPtr;

    BlockPtr();

    BlockPtr(nullptr_t);

    BlockPtr(std::shared_ptr<T>);

    // Conversions from _Dense, LowRank, Hierarchical to Node.
    // Other way around DOES NOT WORK
    // template <typename U>
    // BlockPtr(const std::shared_ptr<U>& ptr) : std::shared_ptr<T>(ptr) {};

    template <typename U>
    BlockPtr(const BlockPtr<U>& ptr) : std::shared_ptr<T>(ptr) {};

    BlockPtr(T*);

    // This forwards the = operator to Node, _Dense, Hierarchical etc, isn't used
    // right now.
    template <typename U>
    const BlockPtr<T> operator=(const BlockPtr<U>& ptr) {
      *(this->get()) = *(ptr.get());
      return *this;
    };

    // Operators necessary for making U, S, V of LowRank a _DensePtr
    const BlockPtr<T>& operator=(int);
    const BlockPtr<T>& operator-() const;

    typename return_type<T>::type& operator()(int, int);
    const typename return_type<T>::type& operator()(int, int) const;

    typename return_type<T>::type& operator[](int);
    const typename return_type<T>::type& operator[](int) const;

    // Add constructor using arg list, forward to make_shared<T>
    // Might have to make template specialization for Node, _Dense etc...
    template <typename... Args>
    explicit BlockPtr(Args&&... args)
      : std::shared_ptr<T>(std::make_shared<T>(std::forward<Args>(args)...)) {};

    const bool is(const int) const;

    const char* is_string() const;

    void resize(int);

    void resize(int, int);

    double norm() const;

    void print() const;

    void getrf();

    template <typename U>
    void trsm(const BlockPtr<U>& A, const char& uplo) {
      return this->get()->trsm(A, uplo);
    }

    void gemm(const BlockPtr<T>&, const BlockPtr<T>&);
  };


  // Partial specializations for _Dense
  template <>
  double& BlockPtr<_Dense>::operator()(int, int);

  template<>
  const double& BlockPtr<_Dense>::operator()(int, int) const;

  template<>
  const BlockPtr<_Dense>& BlockPtr<_Dense>::operator-() const;
}

#endif
