#include "hicma/util/print.h"
#include "hicma/util/timer.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <string>
#include <chrono>

namespace hicma {
namespace timing {

class TimerClass {
public:
  std::string name;
  TimerClass* parent;
  bool running;
  typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;
  time_point start_time;
  std::chrono::duration<double> duration;
  std::map<std::string, TimerClass> subtimers;

  TimerClass() {
    name = "";
    parent = nullptr;
    running = true;
    duration = std::chrono::duration<double>::zero();
    start_time = std::chrono::high_resolution_clock::now();
  }

  TimerClass(std::string name, TimerClass* parent)
  : name(name), parent(parent) {
    running = true;
    duration = std::chrono::duration<double>::zero();
    start_time = std::chrono::high_resolution_clock::now();
  }

  void start() {
    assert(!running);
    running = true;
    start_time = std::chrono::high_resolution_clock::now();
  }

  void start(std::string event) {
    if (subtimers.find(event) == subtimers.end()) {
      subtimers[event] = TimerClass(event, this);
    } else {
      subtimers[event].start();
    }
  }

  void stop(std::string event) {
    assert(name == event);
    assert(running);
    time_point end_time = std::chrono::high_resolution_clock::now();
    running = false;
    duration += end_time - start_time;
  }

  double get_duration(std::string event) const {
    if (event.empty()) {
      return duration.count();
    } else {
      assert(subtimers.find(event) != subtimers.end());
      return subtimers.at(event).duration.count();
    }
  }

  void print_to_depth(int depth, int at_depth) const {
    std::string tag = "";
    for (int i=0; i<at_depth-1; i++) tag += " ";
    if (at_depth != 0) tag += "|-";
    tag += name;
    print(tag, duration.count());
    if (depth > 0) {
      std::vector<const TimerClass*> duration_sorted;
      for (const auto& pair : subtimers) {
        duration_sorted.push_back(&pair.second);
      }
      std::sort(
        duration_sorted.begin(), duration_sorted.end(),
        [](const TimerClass* a, const TimerClass* b) {
          return a->duration > b->duration;
        }
      );
      for (const TimerClass* ptr : duration_sorted) {
        ptr->print_to_depth(depth-1, at_depth+1);
      }
    }
  }

  void print_to_depth(
    std::string event, int depth, int at_depth=0
  ) const {
    assert(subtimers.find(event) != subtimers.end());
    assert(!subtimers.at(event).running);
    subtimers.at(event).print_to_depth(depth, at_depth);
  }
};

// Global recursive state machine
TimerClass Timer;
// Timer into recursive state machine
TimerClass* current_timer = &Timer;

void start(std::string event) {
  current_timer->start(event);
  // TODO write getter
  current_timer = &(current_timer->subtimers[event]);
}

void stop(std::string event) {
  current_timer->stop(event);
  if (current_timer->parent != nullptr) {
    current_timer = current_timer->parent;
  }
}

void stopAndPrint(std::string event, int depth) {
  stop(event);
  printTime(event, depth);
}

void printTime(std::string event, int depth) {
  current_timer->print_to_depth(event, depth);
}

} // namespace timing
} // namespace hicma
