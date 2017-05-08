/* 
 * File:   graph.h
 * Author: ryan
 *
 * Created on 19 March 2017, 20:27
 */

#ifndef GRAPH_SETTINGS_H
#define GRAPH_SETTINGS_H

#include "states/state.h"

#include <vector>

// TODO: rename to NetworkContext

class GraphSettings{
public:
    State* state;
    
    GraphSettings(){}
    virtual void setParameters(std::vector<int>& params){};
};

#endif /* GRAPH_H */

