#include "hicma/util/counter.h"

#include "hicma/util/print.h"

#include <map>
#include <string>


namespace hicma {

std::map<std::string,int> globalCounter;

int getCounter(std::string event) {
  if(globalCounter.find(event) == globalCounter.end()) return -1;
  else return globalCounter[event];
}

void resetCounter(std::string event) {
  globalCounter[event] = 0;
}

void updateCounter(std::string event, int d) {
  if(globalCounter.find(event) == globalCounter.end())
    globalCounter[event] = 0;
  globalCounter[event] += d;
}

void printCounter(std::string event) {
  print(event, globalCounter[event]);
}

} // namespace hicma
