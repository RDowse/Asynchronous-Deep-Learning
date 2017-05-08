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
#include "training/stochastic_momentum_training.h"

// Note: context design pattern, maintains and shared information for the system 
class DNNGraphSettings: public GraphSettings{
public:
    
    // Training method
    TrainingStrategy* trainingStrategy;
    
    // Training param
    float lr = 0.1;            // learning rate
    float alpha = 0.5;          // momentum
    int sample = 0;             // selected sample for predicting
    int maxEpoch = 100000;           // maximum epochs for training
    float minError = 0.01;      // minimum error to stop training
    int miniBatchSize = 1;
    // Weight initialisation
    void (*initWeightsFnc)(vector<float>& ,int ,int);
    
    // Activation function
    float (*activationFnc)(float);
    
    // Differentiated Activation function
    float (*deltaActivationFnc)(float);
    
    // Flags
    bool update = false;        // flag for updating weight. TODO: update with msgs
    
    DNNGraphSettings(){
        trainingStrategy = new StochasticMomentumTraining();
        activationFnc = &math::activationSig;
        deltaActivationFnc = &math::deltaActivationSig;
        initWeightsFnc = &math::initWeights;
    }
    
    void setParameters(vector<int>& params) override {

    }
};

#endif /* DNN_GRAPH_H */

