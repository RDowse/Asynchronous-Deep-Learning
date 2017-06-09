/* 
 * File:   graph.h
 * Author: ryan
 *
 * Created on 19 March 2017, 20:27
 */

#ifndef GRAPH_SETTINGS_H
#define GRAPH_SETTINGS_H

#include <vector>

class GraphSettings{
public:
    double runTime = 0;
    bool enableVariableEdgeDelay = false;
    int maxDelay = 0;
    int (*delayInitialiserFnc)(int);
public:
    GraphSettings(){}
    virtual void setParameters(std::vector<int>& params){};
    virtual void incrementTime(){};
};

#endif /* GRAPH_H */

