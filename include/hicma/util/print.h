#ifndef hicma_util_print_h
#define hicma_util_print_h

#include "hicma/classes/node.h"

#include "yorel/yomm2/cute.hpp"
using yorel::yomm2::virtual_;

#include <string>


namespace hicma
{

extern bool VERBOSE;

void printXML(const Node& A, std::string filename = "matrix.xml");

void print(const Node&);

declare_method(void, print_omm, (virtual_<const Node&>))

void print(std::string s);

template<typename T>
void print(std::string s, T v, bool fixed=true);

}

#endif // hicma_util_print_h
