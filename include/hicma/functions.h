#ifndef functions_h
#define functions_h
#include <vector>

namespace hicma {
  void zeros(
            std::vector<double>& data,
            std::vector<double>& x,
            const int& ni,
            const int& nj,
            const int& i_begin,
            const int& j_begin
            );

  void random(
            std::vector<double>& data,
            std::vector<double>& x,
            const int& ni,
            const int& nj,
            const int& i_begin,
            const int& j_begin
            );

  void latms(
             std::vector<double>& data,
             std::vector<double>& x,
             const int& ni,
             const int& nj,
             const int& i_begin,
             const int& j_begin
             );

  void arange(
            std::vector<double>& data,
            std::vector<double>& x,
            const int& ni,
            const int& nj,
            const int& i_begin,
            const int& j_begin
            );

  void laplace1d(
                 std::vector<double>& data,
                 std::vector<double>& x,
                 const int& ni,
                 const int& nj,
                 const int& i_begin,
                 const int& j_begin
                 );
}
#endif
