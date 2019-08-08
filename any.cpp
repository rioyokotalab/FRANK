#include "any.h"
#include "low_rank.h"
#include "hierarchical.h"

#include <iostream>

namespace hicma {

  Any::Any() = default;

  Any::Any(const Any& A) : ptr(A.ptr->clone()) {}

  Any::Any(const Node& A) : ptr(A.clone()) {}

  Any::Any(Node&& A) {
    if (A.is(HICMA_DENSE)) {
      ptr = std::unique_ptr<Dense>(new Dense);
      swap(static_cast<Dense&>(*ptr), static_cast<Dense&>(A));
    } else if (A.is(HICMA_LOWRANK)) {
      ptr = std::unique_ptr<LowRank>(new LowRank);
      swap(static_cast<LowRank&>(*ptr), static_cast<LowRank&>(A));
    } else if (A.is(HICMA_HIERARCHICAL)) {
      ptr = std::unique_ptr<Hierarchical>(new Hierarchical);
      swap(static_cast<Hierarchical&>(*ptr), static_cast<Hierarchical&>(A));
    } else {
      std::cerr << "Node is of an undefined type." << std::endl;
      abort();
    }
  }

  Any::~Any() = default;

  const Any& Any::operator=(Any A) {
    this->ptr = std::move(A.ptr);
    return *this;
  }

  bool Any::is(const int i) const { return ptr->is(i); }

  const char* Any::type() const { return ptr->type(); }

  double Any::norm() const {
    return ptr->norm();
  }

  void Any::print() const {
    ptr->print();
  }

  void Any::transpose() {
    ptr->transpose();
  }

  void Any::getrf() {
    ptr->getrf();
  }

  void Any::trsm(const Any& A, const char& uplo) {
    if (A.is(HICMA_DENSE)) {
      ptr->trsm(static_cast<Dense&>(*A.ptr), uplo);
    } else if (A.is(HICMA_HIERARCHICAL)) {
      ptr->trsm(static_cast<Hierarchical&>(*A.ptr), uplo);
    } else {
      std::cerr << "Input matrix for trsm must be Hierarchical or Dense." << std::endl;
      abort();
    }
  }

  void Any::gemm(const Any& A, const Any& B, const double& alpha, const double& beta) {
    if (A.is(HICMA_DENSE)) {
      if (B.is(HICMA_DENSE)) {
        ptr->gemm(static_cast<Dense&>(*A.ptr), static_cast<Dense&>(*B.ptr), alpha, beta);
      } else if (B.is(HICMA_LOWRANK)) {
        ptr->gemm(static_cast<Dense&>(*A.ptr), static_cast<LowRank&>(*B.ptr), alpha, beta);
      } else if (B.is(HICMA_HIERARCHICAL)) {
        ptr->gemm(static_cast<Dense&>(*A.ptr), static_cast<Hierarchical&>(*B.ptr), alpha, beta);
      } else {
        std::cerr << "Input matrix for gemm must be Hierarchical, LowRank or Dense." << std::endl;
        abort();
      }
    } else if (A.is(HICMA_LOWRANK)) {
      if (B.is(HICMA_DENSE)) {
        ptr->gemm(static_cast<LowRank&>(*A.ptr), static_cast<Dense&>(*B.ptr), alpha, beta);
      } else if (B.is(HICMA_LOWRANK)) {
        ptr->gemm(static_cast<LowRank&>(*A.ptr), static_cast<LowRank&>(*B.ptr), alpha, beta);
      } else if (B.is(HICMA_HIERARCHICAL)) {
        ptr->gemm(static_cast<LowRank&>(*A.ptr), static_cast<Hierarchical&>(*B.ptr), alpha, beta);
      } else {
        std::cerr << "Input matrix for gemm must be Hierarchical, LowRank or Dense." << std::endl;
        abort();
      }
    } else if (A.is(HICMA_HIERARCHICAL)) {
      if (B.is(HICMA_DENSE)) {
        ptr->gemm(static_cast<Hierarchical&>(*A.ptr), static_cast<Dense&>(*B.ptr), alpha, beta);
      } else if (B.is(HICMA_LOWRANK)) {
        ptr->gemm(static_cast<Hierarchical&>(*A.ptr), static_cast<LowRank&>(*B.ptr), alpha, beta);
      } else if (B.is(HICMA_HIERARCHICAL)) {
        ptr->gemm(static_cast<Hierarchical&>(*A.ptr), static_cast<Hierarchical&>(*B.ptr), alpha, beta);
      } else {
        std::cerr << "Input matrix for gemm must be Hierarchical, LowRank or Dense." << std::endl;
        abort();
      }
    } else {
      std::cerr << "Input matrix for gemm must be Hierarchical, LowRank or Dense." << std::endl;
      abort();
    }
  }

  void Any::geqrt(Any& T) {
    if(T.is(HICMA_DENSE)) {
      ptr->geqrt(static_cast<Dense&>(*T.ptr));
    }
    else if(T.is(HICMA_HIERARCHICAL)) {
      ptr->geqrt(static_cast<Hierarchical&>(*T.ptr));
    }
    else {
      std::cerr << "Input matrix for geqrt must be Hierarchical or Dense." << std::endl;
      abort();
    }
  }

  void Any::larfb(const Any& Y, const Any& T, const bool trans) {
    if(Y.is(HICMA_DENSE) && T.is(HICMA_DENSE)) {
      ptr->larfb(static_cast<Dense&>(*Y.ptr), static_cast<Dense&>(*T.ptr), trans);
    }
    else if(Y.is(HICMA_HIERARCHICAL) && T.is(HICMA_HIERARCHICAL)) {
      ptr->larfb(static_cast<Hierarchical&>(*Y.ptr), static_cast<Hierarchical&>(*T.ptr), trans);
    }
    else {
      std::cerr << "Input matrix for larfb must be (Dense,Dense) or (Hierarchical,Hierarchical)." << std::endl;
      abort();
    }
  }

  void Any::tpqrt(Any& A, Any& T) {
    if(A.is(HICMA_DENSE) && T.is(HICMA_DENSE)) {
      ptr->tpqrt(static_cast<Dense&>(*A.ptr), static_cast<Dense&>(*T.ptr));
    }
    else if(A.is(HICMA_HIERARCHICAL) && T.is(HICMA_DENSE)) {
      ptr->tpqrt(static_cast<Hierarchical&>(*A.ptr), static_cast<Dense&>(*T.ptr));
    }
    else if(A.is(HICMA_HIERARCHICAL) && T.is(HICMA_HIERARCHICAL)) {
      ptr->tpqrt(static_cast<Hierarchical&>(*A.ptr), static_cast<Hierarchical&>(*T.ptr));
    }
    else {
      std::cerr << "Input matrix for tpqrt must be (Dense,Dense), (Hierarchical, Dense) or (Hierarchical,Hierarchical)." << std::endl;
      abort();
    }
  }

  void Any::tpmqrt(Any& B, const Any& Y, const Any& T, const bool trans) {
    if(B.is(HICMA_DENSE)) {
      if(Y.is(HICMA_DENSE) && T.is(HICMA_DENSE)) {
        ptr->tpmqrt(static_cast<Dense&>(*B.ptr), static_cast<Dense&>(*Y.ptr), static_cast<Dense&>(*T.ptr), trans);
      }
      else if(Y.is(HICMA_LOWRANK) && T.is(HICMA_DENSE)) {
        ptr->tpmqrt(static_cast<Dense&>(*B.ptr), static_cast<LowRank&>(*Y.ptr), static_cast<Dense&>(*T.ptr), trans);
      }
      else if(Y.is(HICMA_HIERARCHICAL) && T.is(HICMA_HIERARCHICAL)) {
        ptr->tpmqrt(static_cast<Dense&>(*B.ptr), static_cast<Hierarchical&>(*Y.ptr), static_cast<Hierarchical&>(*T.ptr), trans);
      }
      else {
        std::cerr << "Invalid input for tpmqrt with Dense upper rectangular block" << std::endl;
        abort();
      }
    }
    else if(B.is(HICMA_LOWRANK)) {
      if(Y.is(HICMA_DENSE) && T.is(HICMA_DENSE)) {
        ptr->tpmqrt(static_cast<LowRank&>(*B.ptr), static_cast<Dense&>(*Y.ptr), static_cast<Dense&>(*T.ptr), trans);
      }
      else if(Y.is(HICMA_LOWRANK) && T.is(HICMA_DENSE)) {
        ptr->tpmqrt(static_cast<LowRank&>(*B.ptr), static_cast<LowRank&>(*Y.ptr), static_cast<Dense&>(*T.ptr), trans);
      }
      else {
        std::cerr << "Invalid input for tpmqrt with LowRank upper rectangular block" << std::endl;
        abort();
      }
    }
    else if(B.is(HICMA_HIERARCHICAL)) {
      if(Y.is(HICMA_DENSE) && T.is(HICMA_DENSE)) {
        ptr->tpmqrt(static_cast<Hierarchical&>(*B.ptr), static_cast<Dense&>(*Y.ptr), static_cast<Dense&>(*T.ptr), trans);
      }
      else if(Y.is(HICMA_HIERARCHICAL) && T.is(HICMA_HIERARCHICAL)) {
        ptr->tpmqrt(static_cast<Hierarchical&>(*B.ptr), static_cast<Hierarchical&>(*Y.ptr), static_cast<Hierarchical&>(*T.ptr), trans);
      }
      else {
        std::cerr << "Input input for tpmqrt with Hierarchical upper rectangular block" << std::endl;
        abort();
      }
    }
    else {
      std::cerr << "Input matrix for tpmqrt must be Dense or Hierarchical" << std::endl;
      abort();
    }
  }

}
