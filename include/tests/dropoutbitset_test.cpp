
/* 
 * File:   dropoutbitset_test.cpp
 * Author: ryan
 *
 * Created on 26 May 2017, 20:09
 */

#include "training/dropout_bitset.h"

#include <stdlib.h>
#include <iostream>

#include "training/dropout_bitset.h"

/*
 * Simple C++ Test Suite
 */

void test1() {
    std::cout << "dropoutbitset_test test 1" << std::endl;
}

void test2() {
    std::cout << "dropoutbitset_test test 2" << std::endl;
    std::cout << "%TEST_FAILED% time=0 testname=test2 (dropoutbitset_test) message=error message sample" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "%SUITE_STARTING% dropoutbitset_test" << std::endl;
    std::cout << "%SUITE_STARTED%" << std::endl;

    std::cout << "%TEST_STARTED% test1 (dropoutbitset_test)" << std::endl;
    test1();
    std::cout << "%TEST_FINISHED% time=0 test1 (dropoutbitset_test)" << std::endl;

    std::cout << "%TEST_STARTED% test2 (dropoutbitset_test)\n" << std::endl;
    test2();
    std::cout << "%TEST_FINISHED% time=0 test2 (dropoutbitset_test)" << std::endl;

    std::cout << "%SUITE_FINISHED% time=0" << std::endl;

    return (EXIT_SUCCESS);
}

