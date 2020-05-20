#ifndef hicma_util_print_h
#define hicma_util_print_h

#include <string>


namespace hicma
{

class Matrix;

extern bool VERBOSE;

std::string type(const Matrix&);

void printXML(const Matrix& A, std::string filename = "matrix.xml");

void print(const Matrix&);

void print(std::string s);

template<typename T>
void print(std::string s, T v, bool fixed=true);

}

#endif // hicma_util_print_h
