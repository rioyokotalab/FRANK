#ifndef timer_h
#define timer_h
#include "hicma/util/print.h"
#include "hicma/util/timer.h"

#include <map>
#include <sys/time.h>

namespace hicma {

  std::map<std::string,timeval> tic;
  std::map<std::string,double> sumTime;

  void start(std::string event) {
    timeval t;
    gettimeofday(&t, NULL);
    tic[event] = t;
  }

  void stop(std::string event, bool verbose=true) {
    timeval toc;
    gettimeofday(&toc, NULL);
    sumTime[event] += toc.tv_sec - tic[event].tv_sec +
      (toc.tv_usec - tic[event].tv_usec) * 1e-6;
    if (verbose) print(event, sumTime[event]);
  }

  void printTime(std::string event) {
    print(event, sumTime[event]);
    sumTime[event] = 0;
  }
}
#endif
