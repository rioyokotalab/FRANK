#ifndef timer_h
#define timer_h
#include <map>
#include "print.h"
#include <sys/time.h>

namespace hicma {
  extern std::map<std::string,timeval> tic;
  extern std::map<std::string,double> sumTime;

  void start(std::string event);

  void stop(std::string event, bool verbose=true);

  void print2(std::string event);
}
#endif
