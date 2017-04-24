/* 
 * File:   dnn_graph_settings.h
 * Author: ryan
 *
 * Created on 19 March 2017, 16:50
 */

#ifndef DNN_GRAPH_SETTINGS_H
#define DNN_GRAPH_SETTINGS_H

#include "graphs/graph_settings.h"

#include "states/forward_train_state.h"

class DNNGraphSettings: public GraphSettings{
public:
    enum Command{
        predict, train
    };
    enum Operation{
        forward, backward
    };
    
    State* state = new ForwardTrainState();
    Command cmd = Command::train;
    Operation op = Operation::forward;
    float lr = 0.01;            // learning rate
    float alpha = 0.5;         // momentum
    int sample = 0;             // selected sample for predicting
    int maxEpoch = 1;       // maximum epochs for training
    float minError = 0.01;      // minimum error to stop training
//    int batchSize = 90;         // batch size
    int batchSize = 4;         // batch size    TODO: check batch update logic
    
    bool update = false;    // flag for updating weight. TODO: update with msgs
    
    DNNGraphSettings(){}
    
    void info(){}
};

#endif /* DNN_GRAPH_H */

