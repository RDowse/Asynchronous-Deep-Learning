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
    GraphSettings(){}
    virtual void setParameters(std::vector<int>& params){};
};

#endif /* GRAPH_H */

