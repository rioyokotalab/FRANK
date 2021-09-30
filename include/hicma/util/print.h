#ifndef hicma_util_print_h
#define hicma_util_print_h

#include <string>


/**
 * @brief General namespace of the HiCMA library
 */
namespace hicma
{

class Matrix;

extern bool VERBOSE;

std::string type(const Matrix&);

void write_JSON(const Matrix& A, std::string filename = "matrix.json");

void print(const Matrix&);

void print(std::string s);

template<typename T>
void print(std::string s, T v, bool fixed=true);

}

#endif // hicma_util_print_h
