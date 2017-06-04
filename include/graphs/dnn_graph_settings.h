/* 
 * File:   dnn_graph_settings.h
 * Author: ryan
 *
 * Created on 19 March 2017, 16:50
 */

#ifndef DNN_GRAPH_SETTINGS_H
#define DNN_GRAPH_SETTINGS_H

#include "tools/math.h"
#include "misc/weight_initialisers.h"

#include "graphs/graph_settings.h"

#include <iostream>

using namespace std;

// Note: context design pattern, maintains and shared information for the system 
class DNNGraphSettings: public GraphSettings{
public:
    
    // Debugging log level
    int logLevel;
    string command;
    int seed = 0;
    
    // Parallel data parameters
    int numModels = 1;
    int wait = 1;
    
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
    
    // Specify dropout type
    string dropout;
    
    // Training parameters
    float lr;             // learning rate (0.1)
    float alpha;          // momentum (0.5)
    int batchSize;        // training set must be divisible batch size 
    int maxEpoch;         // maximum epochs for training
    float minError;       // minimum error to stop training
    
    // Error values
    float accuracy;
    float training_error;
    
    // Activation values
    float actMax;
    float actMin;
    
    // Weight initialisation
    void (*initWeightsFnc)(Eigen::VectorXf& ,int ,int);
    
    // Activation function
    float (*activationFnc)(float);
    
    // Differentiated Activation function
    float (*deltaActivationFnc)(float);
    
    // Error saving
    Eigen::VectorXf accuracy_validation;
    Eigen::VectorXf accuracy_train;
    Eigen::VectorXf accuracy_testing;
    Eigen::VectorXf error_validation;
    Eigen::VectorXf error_training;
    Eigen::VectorXf error_testing;
    Eigen::MatrixXi confusion_matrix_test;
    
    DNNGraphSettings(){
        activationFnc = &math::activationTanH;
        deltaActivationFnc = &math::deltaActivationTanH;
        initWeightsFnc = &weight_init::initWeights;
        
        actMax = activationFnc(5);
        actMin = activationFnc(-5);
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
        cout << " minError "  << minError;
        cout << " numModels "  << numModels << endl;
    }
};

#endif /* DNN_GRAPH_H */

