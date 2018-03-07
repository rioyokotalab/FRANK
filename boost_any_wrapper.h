#ifndef boost_any_wrapper_h
#define boost_any_wrapper_h
#include "dense.h"
#include "low_rank.h"
#include "hierarchical.h"

namespace hicma {
  std::vector<int> getrf(boost::any& A) {
    std::vector<int> ipiv;
    if (A.type() == typeid(Dense)) {
      ipiv = boost::any_cast<Dense&>(A).getrf();
    }
    else if (A.type() == typeid(Hierarchical)) {
      ipiv = boost::any_cast<Hierarchical&>(A).getrf();
    }
    else {
      fprintf(stderr,"Data type must be Dense or Hierarchical.\n"); abort();
    }
    return ipiv;
  }

  void trsm(boost::any& Aii, boost::any& Aij, const char& uplo) {
    if (Aii.type() == typeid(Dense)) {
      if (Aij.type() == typeid(Dense)) {
        boost::any_cast<Dense&>(Aij).trsm(boost::any_cast<Dense&>(Aii), uplo);
      }
      else if (Aij.type() == typeid(LowRank)) {
        boost::any_cast<LowRank&>(Aij).trsm(boost::any_cast<Dense&>(Aii), uplo);
      }
      else if (Aij.type() == typeid(Hierarchical)) {
        fprintf(stderr,"Operation undefined 0.\n"); abort();
      }
      else {
        fprintf(stderr,"Second value must be Dense, LowRank or Hierarchical.\n"); abort();
      }
    }
    else if (Aii.type() == typeid(Hierarchical)) {
      if (Aij.type() == typeid(Dense)) {
        fprintf(stderr,"Operation undefined 1.\n"); abort();
      }
      else if (Aij.type() == typeid(LowRank)) {
        fprintf(stderr,"Operation undefined 2.\n"); abort();
      }
      else if (Aij.type() == typeid(Hierarchical)) {
        fprintf(stderr,"Operation undefined 3.\n"); abort();
      }
      else {
        fprintf(stderr,"Second value must be Dense, LowRank or Hierarchical.\n"); abort();
      }
    }
    else {
      fprintf(stderr,"First value must be Dense or Hierarchical.\n"); abort();
    }
  }

  void gemv(boost::any& A, boost::any& b, boost::any& x) {
    if (A.type() == typeid(Dense)) {
      if (b.type() == typeid(Dense)) {
        boost::any_cast<Dense&>(x).gemv(boost::any_cast<Dense&>(A), boost::any_cast<Dense&>(b));
      }
      else if (b.type() == typeid(Hierarchical)) {
        fprintf(stderr,"Operation undefined 4.\n"); abort();
      }
      else {
        fprintf(stderr,"Second value must be Dense or Hierarchical.\n"); abort();
      }
    }
    else if (A.type() == typeid(LowRank)) {
      if (b.type() == typeid(Dense)) {
        boost::any_cast<Dense&>(x).gemv(boost::any_cast<LowRank&>(A), boost::any_cast<Dense&>(b));
      }
      else if (b.type() == typeid(Hierarchical)) {
        fprintf(stderr,"Operation undefined 5.\n"); abort();
      }
      else {
        fprintf(stderr,"Second value must be Dense or Hierarchical.\n"); abort();
      }
    }
    else if (A.type() == typeid(Hierarchical)) {
      if (b.type() == typeid(Dense)) {
        fprintf(stderr,"Operation undefined 6.\n"); abort();
      }
      else if (b.type() == typeid(Hierarchical)) {
        fprintf(stderr,"Operation undefined 7.\n"); abort();
      }
      else {
        fprintf(stderr,"Second value must be Dense or Hierarchical.\n"); abort();
      }
    }
    else {
      fprintf(stderr,"First value must be Dense, LowRank or Hierarchical.\n"); abort();
    }
  }

  void gemm(boost::any& A, boost::any& B, boost::any& C) {
    if (A.type() == typeid(Dense)) {
      if (B.type() == typeid(Dense)) {
        if (C.type() == typeid(Dense)) {
          boost::any_cast<Dense&>(C).gemm(boost::any_cast<Dense&>(A), boost::any_cast<Dense&>(B));
        }
        else if (C.type() == typeid(LowRank)) {
          fprintf(stderr,"Operation undefined 8.\n"); abort();
        }
        else if (C.type() == typeid(Hierarchical)) {
          fprintf(stderr,"Operation undefined 9.\n"); abort();
        }
        else {
          fprintf(stderr,"Third value must be Dense, LowRank or Hierarchical.\n"); abort();
        }
      }
      else if (B.type() == typeid(LowRank)) {
        if (C.type() == typeid(Dense)) {
          boost::any_cast<Dense&>(C).gemm(boost::any_cast<Dense&>(A), boost::any_cast<LowRank&>(B));
        }
        else if (C.type() == typeid(LowRank)) {
          boost::any_cast<LowRank&>(C).gemm(boost::any_cast<Dense&>(A), boost::any_cast<LowRank&>(B));
        }
        else if (C.type() == typeid(Hierarchical)) {
          fprintf(stderr,"Operation undefined 12.\n"); abort();
        }
        else {
          fprintf(stderr,"Third value must be Dense, LowRank or Hierarchical.\n"); abort();
        }
      }
      else if (B.type() == typeid(Hierarchical)) {
        if (C.type() == typeid(Dense)) {
          fprintf(stderr,"Operation undefined 13.\n"); abort();
        }
        else if (C.type() == typeid(LowRank)) {
          fprintf(stderr,"Operation undefined 14.\n"); abort();
        }
        else if (C.type() == typeid(Hierarchical)) {
          fprintf(stderr,"Operation undefined 15.\n"); abort();
        }
        else {
          fprintf(stderr,"Third value must be Dense, LowRank or Hierarchical.\n"); abort();
        }
      }
      else {
        fprintf(stderr,"Second value must be Dense, LowRank or Hierarchical.\n"); abort();
      }
    }
    else if (A.type() == typeid(LowRank)) {
      if (B.type() == typeid(Dense)) {
        if (C.type() == typeid(Dense)) {
          boost::any_cast<Dense&>(C).gemm(boost::any_cast<LowRank&>(A), boost::any_cast<Dense&>(B));
        }
        else if (C.type() == typeid(LowRank)) {
          boost::any_cast<LowRank&>(C).gemm(boost::any_cast<LowRank&>(A), boost::any_cast<Dense&>(B));
        }
        else if (C.type() == typeid(Hierarchical)) {
          fprintf(stderr,"Operation undefined 18.\n"); abort();
        }
        else {
          fprintf(stderr,"Third value must be Dense, LowRank or Hierarchical.\n"); abort();
        }
      }
      else if (B.type() == typeid(LowRank)) {
        if (C.type() == typeid(Dense)) {
          boost::any_cast<Dense&>(C).gemm(boost::any_cast<LowRank&>(A), boost::any_cast<LowRank&>(B));
        }
        else if (C.type() == typeid(LowRank)) {
          boost::any_cast<LowRank&>(C).gemm(boost::any_cast<LowRank&>(A), boost::any_cast<LowRank&>(B));
        }
        else if (C.type() == typeid(Hierarchical)) {
          fprintf(stderr,"Operation undefined 21.\n"); abort();
        }
        else {
          fprintf(stderr,"Third value must be Dense, LowRank or Hierarchical.\n"); abort();
        }
      }
      else if (B.type() == typeid(Hierarchical)) {
        if (C.type() == typeid(Dense)) {
          fprintf(stderr,"Operation undefined 22.\n"); abort();
        }
        else if (C.type() == typeid(LowRank)) {
          fprintf(stderr,"Operation undefined 23.\n"); abort();
        }
        else if (C.type() == typeid(Hierarchical)) {
          fprintf(stderr,"Operation undefined 24.\n"); abort();
        }
        else {
          fprintf(stderr,"Third value must be Dense, LowRank or Hierarchical.\n"); abort();
        }
      }
      else {
        fprintf(stderr,"Second value must be Dense, LowRank or Hierarchical.\n"); abort();
      }
    }
    else if (A.type() == typeid(Hierarchical)) {
      if (B.type() == typeid(Dense)) {
        if (C.type() == typeid(Dense)) {
          fprintf(stderr,"Operation undefined 25.\n"); abort();
        }
        else if (C.type() == typeid(LowRank)) {
          fprintf(stderr,"Operation undefined 26.\n"); abort();
        }
        else if (C.type() == typeid(Hierarchical)) {
          fprintf(stderr,"Operation undefined 27.\n"); abort();
        }
        else {
          fprintf(stderr,"Third value must be Dense, LowRank or Hierarchical.\n"); abort();
        }
      }
      else if (B.type() == typeid(LowRank)) {
        if (C.type() == typeid(Dense)) {
          fprintf(stderr,"Operation undefined 28.\n"); abort();
        }
        else if (C.type() == typeid(LowRank)) {
          fprintf(stderr,"Operation undefined 29.\n"); abort();
        }
        else if (C.type() == typeid(Hierarchical)) {
          fprintf(stderr,"Operation undefined 30.\n"); abort();
        }
        else {
          fprintf(stderr,"Third value must be Dense, LowRank or Hierarchical.\n"); abort();
        }
      }
      else if (B.type() == typeid(Hierarchical)) {
        if (C.type() == typeid(Dense)) {
          fprintf(stderr,"Operation undefined 31.\n"); abort();
        }
        else if (C.type() == typeid(LowRank)) {
          fprintf(stderr,"Operation undefined 32.\n"); abort();
        }
        else if (C.type() == typeid(Hierarchical)) {
          fprintf(stderr,"Operation undefined 33.\n"); abort();
        }
        else {
          fprintf(stderr,"Third value must be Dense, LowRank or Hierarchical.\n"); abort();
        }
      }
      else {
        fprintf(stderr,"Second value must be Dense, LowRank or Hierarchical.\n"); abort();
      }
    }
    else {
      fprintf(stderr,"First value must be Dense, LowRank or Hierarchical.\n"); abort();
    }
  }
}
#endif
