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
    float lr = 0.1;             // learning rate (0.1)
    float alpha = 0.5;          // momentum (0.5)
    int sample = 0;             // selected sample for predicting
    int maxEpoch = 30;         // maximum epochs for training
    float minError = 0.01;      // minimum error to stop training
    int epoch = 0;
    int batchSize = 20;
    
    // Weight initialisation
    //void (*initWeightsFnc)(vector<float>& ,int ,int);
    void (*initWeightsFnc)(Eigen::VectorXf& ,int ,int);
    
    // Activation function
    float (*activationFnc)(float);
    
    // Differentiated Activation function
    float (*deltaActivationFnc)(float);
    
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

