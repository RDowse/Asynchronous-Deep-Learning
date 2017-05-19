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
#include "training/dropout.h"

// Note: context design pattern, maintains and shared information for the system 
class DNNGraphSettings: public GraphSettings{
public:
    
    // Debugging log level
    int logLevel;
    string command;
    
    // Path params
    string netPath;
    string netType;
    string datasetTrainingPath;
    string datasetTestingPath;
    string datasetType;

    // Graph structure and build params
    int nHLayers;
    int nHidden;
    int nInput;
    int nOutput;
    
    // Current epoch
    int epoch = 0;
    
    // Training method
    TrainingStrategy* trainingStrategy;
    
    // Dropout strategy
    Dropout* dropoutStrategy;
    
    // Training parameters
    float lr;             // learning rate (0.1)
    float alpha;          // momentum (0.5)
    int batchSize;          // training set must be divisible batch size 
    int maxEpoch;         // maximum epochs for training
    float minError;      // minimum error to stop training
    
    // Weight initialisation
    void (*initWeightsFnc)(Eigen::VectorXf& ,int ,int);
    
    // Activation function
    float (*activationFnc)(float);
    
    // Differentiated Activation function
    float (*deltaActivationFnc)(float);
    
    DNNGraphSettings(){
        trainingStrategy = new StochasticMomentumTraining();
        dropoutStrategy = new Dropout();
        activationFnc = &math::activationSig;
        deltaActivationFnc = &math::deltaActivationSig;
        initWeightsFnc = &math::initWeights;
    }
    
    void setParameters(vector<int>& params) override {
        assert(params.size() == 4);
        nHLayers = params[0];
        nHidden = params[1];
        nInput = params[2];
        nOutput = params[3];
    }
    
    void printParameters(){
        cout << "Hyperparameters\n"; 
        cout << "lr "    << lr;
        cout << " alpha " << alpha;
        cout << " batchsize " << batchSize;
        cout << " maxEpoch "  << maxEpoch;  
        cout << " minError "  << minError << endl;
    }
};

#endif /* DNN_GRAPH_H */

