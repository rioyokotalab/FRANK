#ifndef print_h
#define print_h
#include "print.h"
#include "mpi_utils.h"
#include "node.h"
#include "hierarchical.h"
#include "low_rank.h"
#include "dense.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace hicma {
  bool VERBOSE = true;                          //!< Print to screen
  static const int stringLength = 24;           //!< Length of formatted string
  static const int decimal = 7;                 //!< Decimal precision
  static const int wait = 100;                  //!< Waiting time between output of different ranks

  void fillXML(const Node& _A, boost::property_tree::ptree& tree) {
    namespace pt = boost::property_tree;
    if (_A.is(HICMA_HIERARCHICAL)) {
      const Hierarchical& A = static_cast<const Hierarchical&>(_A);
      for (int i = 0; i < A.dim[0]; ++i) {
        for (int j = 0; j < A.dim[1]; ++j) {
          pt::ptree el_subtree{};
          fillXML(A(i, j), el_subtree);
          std::string el_name = "i" + std::to_string(i) + "j" + std::to_string(j);
          tree.add_child(el_name, el_subtree);
          tree.put(el_name + ".<xmlattr>.type", A(i, j).type());
        }
      }
      tree.put("<xmlattr>.type", A.type());
      tree.put("<xmlattr>.dim0", A.dim[0]);
      tree.put("<xmlattr>.dim1", A.dim[1]);
    } else if (_A.is(HICMA_LOWRANK)) {
      const LowRank& A = static_cast<const LowRank&>(_A);
      tree.put("<xmlattr>.type", A.type());
      tree.put("<xmlattr>.dim0", A.dim[0]);
      tree.put("<xmlattr>.dim1", A.dim[1]);
      tree.put("<xmlattr>.rank", A.rank);
    } else if (_A.is(HICMA_DENSE)) {
      const Dense& A = static_cast<const Dense&>(_A);
      tree.put("<xmlattr>.type", A.type());
      tree.put("<xmlattr>.dim0", A.dim[0]);
      tree.put("<xmlattr>.dim1", A.dim[1]);
    } else {
      tree.add("Node", "test");
    }
  }

  void printXML(const Node& A) {
    namespace pt = boost::property_tree;
    pt::ptree tree;
    // Write any header info you want here, like a time stamp
    // And then pass pass A into printXML along with the basic ptree
    pt::ptree root_el;
    fillXML(A, root_el);
    tree.add_child("root", root_el);

    pt::xml_writer_settings<std::string> settings(' ', 4);
    write_xml("test.xml", tree, std::locale(), settings);
  }

  void print(std::string s) {
    if (!VERBOSE | (MPIRANK != 0)) return;
    s += " ";
    std::cout << "--- " << std::setw(stringLength) << std::left
              << std::setfill('-') << s << std::setw(decimal+1) << "-"
              << std::setfill(' ') << std::endl;
  }

  template<typename T>
  void print(std::string s, T v, bool fixed=true) {
    if (!VERBOSE | (MPIRANK != 0)) return;
    std::cout << std::setw(stringLength) << std::left << s << " : ";
    if(fixed)
      std::cout << std::setprecision(decimal) << std::fixed;
    else
      std::cout << std::setprecision(1) << std::scientific;
    std::cout << v << std::endl;
  }

  template void print<double>(std::string s, double v, bool fixed=true);

  template<typename T>
  void printMPI(T data) {
    if (!VERBOSE) return;
    int size = sizeof(data);
    std::vector<T> recv(MPISIZE);
    MPI_Gather(&data, size, MPI_BYTE, &recv[0], size, MPI_BYTE, 0, MPI_COMM_WORLD);
    if (MPIRANK == 0) {
      for (int irank=0; irank<MPISIZE; irank++ ) {
        std::cout << recv[irank] << " ";
      }
      std::cout << std::endl;
    }
  }

  template<typename T>
  void printMPI(T data, const int irank) {
    if (!VERBOSE) return;
    int size = sizeof(data);
    if (MPIRANK == irank) MPI_Send(&data, size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    if (MPIRANK == 0) {
      MPI_Recv(&data, size, MPI_BYTE, irank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::cout << data << std::endl;
    }
  }

  template<typename T>
  void printMPI(T * data, const int begin, const int end) {
    if (!VERBOSE) return;
    int range = end - begin;
    int size = sizeof(*data) * range;
    std::vector<T> recv(MPISIZE * range);
    MPI_Gather(&data[begin], size, MPI_BYTE, &recv[0], size, MPI_BYTE, 0, MPI_COMM_WORLD);
    if (MPIRANK == 0) {
      int ic = 0;
      for (int irank=0; irank<MPISIZE; irank++ ) {
        std::cout << irank << " : ";
        for (int i=0; i<range; i++, ic++) {
          std::cout << recv[ic] << " ";
        }
        std::cout << std::endl;
      }
    }
  }

  template<typename T>
  void printMPI(T * data, const int begin, const int end, const int irank) {
    if (!VERBOSE) return;
    int range = end - begin;
    int size = sizeof(*data) * range;
    std::vector<T> recv(range);
    if (MPIRANK == irank) MPI_Send(&data[begin], size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    if (MPIRANK == 0) {
      MPI_Recv(&recv[0], size, MPI_BYTE, irank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (int i=0; i<range; i++) {
        std::cout << recv[i] << " ";
      }
      std::cout << std::endl;
    }
  }
}
#endif
