#ifndef hicma_util_timer_h
#define hicma_util_timer_h

#include <chrono>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace hicma {
namespace timing {

// Functions to manipulate global timer state machine
void start(std::string event);

double stop(std::string event);

void clearTimers();

void stopAndPrint(std::string event, int depth = 0);

void printTime(std::string event, int depth = 0);

// Interface of the Timer class if user wants to create own timers
class Timer {
public:
  Timer();

  Timer(std::string name, Timer* parent=nullptr);

  void start();

  void start_subtimer(std::string event);

  double stop();

  void clear();

  std::string get_name() const;

  Timer* get_parent() const;

  std::vector<double> get_durations_list() const;

  size_t get_number_of_runs() const;

  double get_total_duration() const;

  const std::map<std::string, Timer>& get_subtimers() const;

  const Timer& operator[](std::string event) const;

  Timer& operator[](std::string event);

  void print_to_depth(std::string event, int depth) const;

private:
  using clock = std::chrono::high_resolution_clock;
  typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;
  typedef std::chrono::duration<double> seconds;
  std::string name;
  Timer* parent;
  bool running;
  time_point start_time;
  std::vector<seconds> durations;
  seconds total_duration;
  std::map<std::string, Timer> subtimers;

  void print_to_depth(int depth, int at_depth, std::string tag_pre = "") const;
};

} // namespace timing
} // namespace hicma

#endif // hicma_util_timer_h
