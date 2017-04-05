/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   dnn_graph_settings.h
 * Author: ryan
 *
 * Created on 19 March 2017, 16:50
 */

#ifndef DNN_GRAPH_SETTINGS_H
#define DNN_GRAPH_SETTINGS_H

#include "graphs/graph_settings.h"

class DNNGraphSettings: public GraphSettings{
public:
    // Various global settings for all nodes in the graph.
    struct GraphType{ 
        
    };
    
    struct PropertiesType{
        
    };
    
    struct DeviceType{
        
    };
    
    unsigned operation; // 0: unset, 1: predict, 2: training
    
    DNNGraphSettings():operation(0){}
    
    void info(){}
};

#endif /* DNN_GRAPH_H */

