#ifndef timer_h
#define timer_h

#include <map>
#include <string>
#include <sys/time.h>

namespace hicma {

  extern std::map<std::string,timeval> tic;
  extern std::map<std::string,double> sumTime;

  void start(std::string event);

  void stop(std::string event, bool verbose=true);

  void printTime(std::string event);
}
#endif
