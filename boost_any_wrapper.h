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
    } else if (A.type() == typeid(Hierarchical)) {
      fprintf(stderr,"Operation undefined 0.\n");
    } else {
      fprintf(stderr,"Data type must be Dense or Hierarchical.\n");
    }
    return ipiv;
  }

  void trsm(boost::any& Aii, boost::any& Aij, const char& uplo) {
    if (Aii.type() == typeid(Dense)) {
      if (Aij.type() == typeid(Dense)) {
        boost::any_cast<Dense&>(Aij).trsm(boost::any_cast<Dense&>(Aii), uplo);
      } else if (Aij.type() == typeid(LowRank)) {
        LowRank A = boost::any_cast<LowRank&>(Aij);
        Dense AD = A.dense();
        AD.trsm(boost::any_cast<Dense&>(Aii), uplo);
        Aij = LowRank(AD, boost::any_cast<LowRank&>(Aij).rank);
        fprintf(stderr,"Revert to dense 1.\n");
        /*
        switch (uplo) {
        case 'l' :
          boost::any_cast<LowRank&>(Aij).U.trsm(boost::any_cast<Dense&>(Aii), uplo);
        case 'u' :
          boost::any_cast<LowRank&>(Aij).V.trsm(boost::any_cast<Dense&>(Aii), uplo);
        default :
          fprintf(stderr,"Third argument must be 'l' for lower, 'u' for upper.\n");
        }
        */
      } else if (Aij.type() == typeid(Hierarchical)) {
        fprintf(stderr,"Operation undefined 1.\n");
      } else {
        fprintf(stderr,"Second value must be Dense, LowRank or Hierarchical.\n");
      }
    } else if (Aii.type() == typeid(Hierarchical)) {
      if (Aij.type() == typeid(Dense)) {
        boost::any_cast<Dense&>(Aij).trsm(boost::any_cast<Dense&>(Aii), uplo);
      } else if (Aij.type() == typeid(LowRank)) {
        fprintf(stderr,"Operation undefined 2.\n");
      } else if (Aij.type() == typeid(Hierarchical)) {
        fprintf(stderr,"Operation undefined 3.\n");
      } else {
        fprintf(stderr,"Second value must be Dense, LowRank or Hierarchical.\n");
      }
    } else {
      fprintf(stderr,"First value must be Dense or Hierarchical.\n");
    }
  }

  void gemv(boost::any& A, boost::any& b, boost::any& x) {
    if (A.type() == typeid(Dense)) {
      if (b.type() == typeid(Dense)) {
        boost::any_cast<Dense&>(x).gemv(boost::any_cast<Dense&>(A), boost::any_cast<Dense&>(b));
      } else if (b.type() == typeid(Hierarchical)) {
        fprintf(stderr,"Operation undefined 4.\n");
      } else {
        fprintf(stderr,"Second value must be Dense or Hierarchical.\n");
      }
    } else if (A.type() == typeid(LowRank)) {
      if (b.type() == typeid(Dense)) {
        Dense AD = boost::any_cast<LowRank&>(A).dense();
        boost::any_cast<Dense&>(x).gemv(AD, boost::any_cast<Dense&>(b));
        fprintf(stderr,"Revert to dense 2.\n");
      } else if (b.type() == typeid(Hierarchical)) {
        fprintf(stderr,"Operation undefined 5.\n");
      } else {
        fprintf(stderr,"Second value must be Dense or Hierarchical.\n");
      }
    } else if (A.type() == typeid(Hierarchical)) {
      if (b.type() == typeid(Dense)) {
        fprintf(stderr,"Operation undefined 6.\n");
      } else if (b.type() == typeid(Hierarchical)) {
        fprintf(stderr,"Operation undefined 7.\n");
      } else {
        fprintf(stderr,"Second value must be Dense or Hierarchical.\n");
      }
    } else {
      fprintf(stderr,"First value must be Dense, LowRank or Hierarchical.\n");
    }
  }

  void gemm(boost::any& A, boost::any& B, boost::any& C) {
    if (A.type() == typeid(Dense)) {
      if (B.type() == typeid(Dense)) {
        if (C.type() == typeid(Dense)) {
          boost::any_cast<Dense&>(C).gemm(boost::any_cast<Dense&>(A), boost::any_cast<Dense&>(B));
        } else if (C.type() == typeid(LowRank)) {
          fprintf(stderr,"Operation undefined 8.\n");
        } else if (C.type() == typeid(Hierarchical)) {
          fprintf(stderr,"Operation undefined 9.\n");
        } else {
          fprintf(stderr,"Third value must be Dense, LowRank or Hierarchical.\n");
        }
      } else if (B.type() == typeid(LowRank)) {
        if (C.type() == typeid(Dense)) {
          boost::any_cast<Dense&>(C).gemm(boost::any_cast<Dense&>(A), boost::any_cast<LowRank&>(B));
        } else if (C.type() == typeid(LowRank)) {
          boost::any_cast<LowRank&>(C) = boost::any_cast<LowRank&>(C) - boost::any_cast<Dense&>(A) * boost::any_cast<LowRank&>(B);
        } else if (C.type() == typeid(Hierarchical)) {
          fprintf(stderr,"Operation undefined 12.\n");
        } else {
          fprintf(stderr,"Third value must be Dense, LowRank or Hierarchical.\n");
        }
      } else if (B.type() == typeid(Hierarchical)) {
        if (C.type() == typeid(Dense)) {
          fprintf(stderr,"Operation undefined 13.\n");
        } else if (C.type() == typeid(LowRank)) {
          fprintf(stderr,"Operation undefined 14.\n");
        } else if (C.type() == typeid(Hierarchical)) {
          fprintf(stderr,"Operation undefined 15.\n");
        } else {
          fprintf(stderr,"Third value must be Dense, LowRank or Hierarchical.\n");
        }
      } else {
        fprintf(stderr,"Second value must be Dense, LowRank or Hierarchical.\n");
      }
    } else if (A.type() == typeid(LowRank)) {
      if (B.type() == typeid(Dense)) {
        if (C.type() == typeid(Dense)) {
          boost::any_cast<Dense&>(C).gemm(boost::any_cast<LowRank&>(A), boost::any_cast<Dense&>(B));
        } else if (C.type() == typeid(LowRank)) {
          Dense CD = boost::any_cast<LowRank&>(C).dense();
          CD.gemm(boost::any_cast<LowRank&>(A), boost::any_cast<Dense&>(B));
          C = LowRank(CD, boost::any_cast<LowRank&>(C).rank);
        } else if (C.type() == typeid(Hierarchical)) {
          fprintf(stderr,"Operation undefined 18.\n");
        } else {
          fprintf(stderr,"Third value must be Dense, LowRank or Hierarchical.\n");
        }
      } else if (B.type() == typeid(LowRank)) {
        if (C.type() == typeid(Dense)) {
          Dense AD = boost::any_cast<LowRank&>(A).dense();
          Dense BD = boost::any_cast<LowRank&>(B).dense();
          boost::any_cast<Dense&>(C).gemm(AD, BD);
          fprintf(stderr,"Revert to dense 5.\n");
        } else if (C.type() == typeid(LowRank)) {
          Dense AD = boost::any_cast<LowRank&>(A).dense();
          Dense BD = boost::any_cast<LowRank&>(B).dense();
          Dense CD = boost::any_cast<LowRank&>(C).dense();
          CD.gemm(AD, BD);
          C = LowRank(CD, boost::any_cast<LowRank&>(C).rank);
          fprintf(stderr,"Revert to dense 6.\n");
        } else if (C.type() == typeid(Hierarchical)) {
          fprintf(stderr,"Operation undefined 21.\n");
        } else {
          fprintf(stderr,"Third value must be Dense, LowRank or Hierarchical.\n");
        }
      } else if (B.type() == typeid(Hierarchical)) {
        if (C.type() == typeid(Dense)) {
          fprintf(stderr,"Operation undefined 22.\n");
        } else if (C.type() == typeid(LowRank)) {
          fprintf(stderr,"Operation undefined 23.\n");
        } else if (C.type() == typeid(Hierarchical)) {
          fprintf(stderr,"Operation undefined 24.\n");
        } else {
          fprintf(stderr,"Third value must be Dense, LowRank or Hierarchical.\n");
        }
      } else {
        fprintf(stderr,"Second value must be Dense, LowRank or Hierarchical.\n");
      }
    } else if (A.type() == typeid(Hierarchical)) {
      if (B.type() == typeid(Dense)) {
        if (C.type() == typeid(Dense)) {
          fprintf(stderr,"Operation undefined 25.\n");
        } else if (C.type() == typeid(LowRank)) {
          fprintf(stderr,"Operation undefined 26.\n");
        } else if (C.type() == typeid(Hierarchical)) {
          fprintf(stderr,"Operation undefined 27.\n");
        } else {
          fprintf(stderr,"Third value must be Dense, LowRank or Hierarchical.\n");
        }
      } else if (B.type() == typeid(LowRank)) {
        if (C.type() == typeid(Dense)) {
          fprintf(stderr,"Operation undefined 28.\n");
        } else if (C.type() == typeid(LowRank)) {
          fprintf(stderr,"Operation undefined 29.\n");
        } else if (C.type() == typeid(Hierarchical)) {
          fprintf(stderr,"Operation undefined 30.\n");
        } else {
          fprintf(stderr,"Third value must be Dense, LowRank or Hierarchical.\n");
        }
      } else if (B.type() == typeid(Hierarchical)) {
        if (C.type() == typeid(Dense)) {
          fprintf(stderr,"Operation undefined 31.\n");
        } else if (C.type() == typeid(LowRank)) {
          fprintf(stderr,"Operation undefined 32.\n");
        } else if (C.type() == typeid(Hierarchical)) {
          fprintf(stderr,"Operation undefined 33.\n");
        } else {
          fprintf(stderr,"Third value must be Dense, LowRank or Hierarchical.\n");
        }
      } else {
        fprintf(stderr,"Second value must be Dense, LowRank or Hierarchical.\n");
      }
    } else {
      fprintf(stderr,"First value must be Dense, LowRank or Hierarchical.\n");
    }
  }

  double l2norm(boost::any A) {
    double l2 = 0;
    if (A.type() == typeid(Dense)) {
      l2 = boost::any_cast<Dense&>(A).norm();
    } else if (A.type() == typeid(LowRank)) {
      fprintf(stderr,"Operation undefined 34.\n");
    } else if (A.type() == typeid(Hierarchical)) {
      fprintf(stderr,"Operation undefined 35.\n");
    } else {
      fprintf(stderr,"Data type must be Dense, LowRank or Hierarchical.\n");
    }
    return l2;
  }
}
#endif
