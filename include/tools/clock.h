
/* 
 * File:   clock.h
 * Author: ryan
 *
 * Created on 29 April 2017, 23:13
 */

#ifndef CLOCK_H
#define CLOCK_H

#include <ctime>
#include <iostream>

class Clock{
    static clock_t begin_time;
public:
    static void tick();
    static void toc();
};

#endif /* CLOCK_H */

