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
    enum Operation{
        forward, backward
    };
    
    Command cmd = Command::train;
    Operation op = Operation::forward;
    float lr = 100;         // learning rate
    int sample = 0;         // selected sample for predicting
    int maxEpoch = 10;      // maximum epochs for training
    float minError = 0.01;  // minimum error to stop training
    int batchSize = 20;    // batch size
    
    bool update = false;    // flag for updating weight. TODO: update with msgs
    
    DNNGraphSettings(){}
    
    void info(){}
};

#endif /* DNN_GRAPH_H */

