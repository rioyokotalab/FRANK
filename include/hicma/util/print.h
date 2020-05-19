#ifndef hicma_util_print_h
#define hicma_util_print_h

#include <string>


namespace hicma
{

class Node;

extern bool VERBOSE;

std::string type(const Node&);

void printXML(const Node& A, std::string filename = "matrix.xml");

void print(const Node&);

void print(std::string s);

template<typename T>
void print(std::string s, T v, bool fixed=true);

}

#endif // hicma_util_print_h
