#ifndef boost_any_wrapper_h
#define boost_any_wrapper_h
#include "dense.h"
#include "low_rank.h"
#include "grid.h"

namespace hicma {
  std::vector<int> getrf(boost::any& A) {
    std::vector<int> ipiv;
    if (A.type() == typeid(Dense)) {
      ipiv = boost::any_cast<Dense&>(A).getrf();
    } else if (A.type() == typeid(Grid)) {
    } else {
      fprintf(stderr,"Data type must be Dense or Grid.\n");
    }
    return ipiv;
  }

  void trsm(boost::any& Aii, boost::any& Aij, const char& uplo) {
    if (Aii.type() == typeid(Dense)) {
      if (Aij.type() == typeid(Dense)) {
        boost::any_cast<Dense&>(Aij).trsm(boost::any_cast<Dense&>(Aii), uplo);
      } else if (Aij.type() == typeid(LowRank)) {
      } else if (Aij.type() == typeid(Grid)) {
      } else {
        fprintf(stderr,"Second value must be Dense, LowRank or Grid.\n");
      }
    } else if (Aii.type() == typeid(Grid)) {
    } else {
      fprintf(stderr,"First value must be Dense or Grid.\n");
    }
  }

  void gemv(boost::any& A, boost::any& b, boost::any& x) {
    if (A.type() == typeid(Dense)) {
      if (b.type() == typeid(Dense)) {
        boost::any_cast<Dense&>(x).gemv(boost::any_cast<Dense&>(A), boost::any_cast<Dense&>(b));        
      } else if (b.type() == typeid(Grid)) {
      } else {
        fprintf(stderr,"Second value must be Dense or Grid.\n");
      }
    } else if (A.type() == typeid(LowRank)) {
      if (b.type() == typeid(Dense)) {
      } else if (b.type() == typeid(Grid)) {
      } else {
        fprintf(stderr,"Second value must be Dense or Grid.\n");
      }
    } else if (A.type() == typeid(Grid)) {
      if (b.type() == typeid(Dense)) {
      } else if (b.type() == typeid(Grid)) {
      } else {
        fprintf(stderr,"Second value must be Dense or Grid.\n");
      }
    } else {
      fprintf(stderr,"First value must be Dense, LowRank or Grid.\n");
    }
  }
  
  void gemm(boost::any& A, boost::any& B, boost::any& C) {
    if (A.type() == typeid(Dense)) {
      if (B.type() == typeid(Dense)) {
        boost::any_cast<Dense&>(C).gemm(boost::any_cast<Dense&>(A), boost::any_cast<Dense&>(B));
      } else if (B.type() == typeid(LowRank)) {
      } else if (B.type() == typeid(Grid)) {
      } else {
        fprintf(stderr,"Second value must be Dense, LowRank or Grid.\n");
      }
    } else if (A.type() == typeid(LowRank)) {
      if (B.type() == typeid(Dense)) {
      } else if (B.type() == typeid(LowRank)) {
      } else if (B.type() == typeid(Grid)) {
      } else {
        fprintf(stderr,"Second value must be Dense, LowRank or Grid.\n");
      }
    } else if (A.type() == typeid(Grid)) {
      if (B.type() == typeid(Dense)) {
      } else if (B.type() == typeid(LowRank)) {
      } else if (B.type() == typeid(Grid)) {
      } else {
        fprintf(stderr,"Second value must be Dense, LowRank or Grid.\n");
      }
    } else {
      fprintf(stderr,"First value must be Dense, LowRank or Grid.\n");
    }
  }
}
#endif
