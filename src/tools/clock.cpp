
#include "tools/clock.h"

clock_t Clock::begin_time = 0;

void Clock::tick(){
    begin_time = std::clock();
}

void Clock::toc(){
    if(begin_time == clock_t()) std::cout << "tick not called first\n";
    std::cout << "Time diff:" <<float( clock () - begin_time ) /  CLOCKS_PER_SEC << "\n";
}