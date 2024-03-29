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
    int stepTime = 0;
    double runTime = 0;
    bool enableVariableEdgeDelay = false;
    GraphSettings(){}
    virtual void setParameters(std::vector<int>& params){};
    void incrementTime(){stepTime++;};
    virtual int delayInitialiser(){}
};

#endif /* GRAPH_H */

