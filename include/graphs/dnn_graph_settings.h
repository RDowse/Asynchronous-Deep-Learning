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
    enum Command{
        predict, train
    };
    
    Command cmd;
    float lr = 0.1;         // learning rate
    int sample = 0;         // selected sample for predicting
    int maxEpoch = 30;      // maximum epochs for training
    float minError = 0.01;  // minimum error to stop training
    int batchSize = 100;
    
    DNNGraphSettings(){}
    
    void info(){}
};

#endif /* DNN_GRAPH_H */

