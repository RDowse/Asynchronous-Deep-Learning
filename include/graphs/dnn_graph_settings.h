/* 
 * File:   dnn_graph_settings.h
 * Author: ryan
 *
 * Created on 19 March 2017, 16:50
 */

#ifndef DNN_GRAPH_SETTINGS_H
#define DNN_GRAPH_SETTINGS_H

#include "graphs/graph_settings.h"
#include "training/training_strategy.h"
#include "states/state.h"

// Note: context design pattern, maintains and shared information for the system 
class DNNGraphSettings: public GraphSettings{
public:
    // Training method
    TrainingStrategy* trainingStrategy; // = NULL;
    
    // Training param
    float lr = 0.1;            // learning rate
    float alpha = 0.5;          // momentum
    int sample = 0;             // selected sample for predicting
    int maxEpoch = 100000;           // maximum epochs for training
    float minError = 0.01;      // minimum error to stop training
    //int batchSize = 90;         // batch size TODO: add after basic backprop complete
    
    // SampleFunc
    // SampleParam
    
    // Weight initialisation
    void (*initWeightsFnc)(vector<float>& ,int ,int) = NULL;
    
    // Activation function
    float (*activationFnc)(float) = NULL;
    
    // Differentiated Activation function
    float (*deltaActivationFnc)(float) = NULL;
    
    // Current network state
    State* state = NULL;
    
    // Flags
    bool update = false;        // flag for updating weight. TODO: update with msgs
    
    DNNGraphSettings(){
//        activationFnc = &math::activationTan;
//        deltaActivationFnc = &math::deltaActivationTan;
        activationFnc = &math::activationSig;
        //activationFnc = &math::activationFastSig;
        deltaActivationFnc = &math::deltaActivationSig;
        initWeightsFnc = &math::initWeights;
    }
};

#endif /* DNN_GRAPH_H */

