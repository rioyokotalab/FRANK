#ifndef hicma_util_timer_h
#define hicma_util_timer_h

#include <string>

namespace hicma {
namespace timing {

  void start(std::string event);

  void stop(std::string event);

  void stopAndPrint(std::string event, int depth = 0);

  void printTime(std::string event, int depth = 0);

} // namespace timing
} // namespace hicma

#endif // hicma_util_timer_h
