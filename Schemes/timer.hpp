#ifndef TIMER_HPP
#define TIMER_HPP

#include <iostream>
#include <chrono>

class Timer {
  public:
    Timer() : _start(std::chrono::steady_clock::now()), _stop(_start) {;}
    std::chrono::steady_clock::duration val() const {return _stop-_start;}
    void stop() {
      _stop = std::chrono::steady_clock::now();
    }
  private:
    const std::chrono::steady_clock::time_point _start;
    std::chrono::steady_clock::time_point _stop;
};

std::ostream& operator<<(std::ostream& os, const Timer& T) {
  using namespace std::chrono_literals;
  std::chrono::steady_clock::duration dt = T.val();
  if (dt > 1h) {
    os << dt/1h <<"h ";
    dt = dt%1h;
  }
  if (dt > 1min) {
    os << dt/1min <<"m ";
    dt = dt%1min;
  }
  os << dt/1s <<".";
  dt = dt%1s;
  os << dt/1ms <<"s";
  return os;
}

#endif
