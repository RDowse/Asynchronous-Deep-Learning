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
#include "tbb/atomic.h"

#include <iostream>
#include <random>

using namespace std;

// Note: context design pattern, maintains and shared information for the system 
class DNNGraphSettings: public GraphSettings{
public:
    // Test
    int testNumber = -1;
    
    // Async
    float forwardDropTolerance = 1;
    float backwardDropTolerance = 1;
    float waitTimeFactor = 1;
    int waitTime = 0;
    float mean = 1;
    float std = 0;
    std::normal_distribution<double> distribution;
    std::default_random_engine generator;
    
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
    
    // Error values
    float accuracy = 0;
    float training_error = 0;
    
    // Activation values
    float actMax;
    float actMin;
    
    // Weight initialisation
    void (*initWeightsFnc)(Eigen::VectorXf& ,int ,int);
    
    // Activation function
    float (*activationFnc)(float);
    
    // Differentiated Activation function
    float (*deltaActivationFnc)(float);
    
    Eigen::MatrixXf (*regularizationFnc)(Eigen::MatrixXf, float);
    float c = 3;
    
    /*
     * Output Data, TODO refactor into separate class
     */
    
    // Error saving
    Eigen::VectorXf accuracy_validation;
    Eigen::VectorXf accuracy_train;
    Eigen::VectorXf accuracy_testing;
    Eigen::VectorXf error_validation;
    Eigen::VectorXf error_training;
    Eigen::VectorXf error_testing;
    Eigen::MatrixXi confusion_matrix_test;
    
    int numForwardMessagesDropped = 0;
    int numForwardMessagesSent = 0;
    int numBackwardMessagesDropped = 0;
    int numBackwardMessagesSent = 0;
    
    // specifically for the sync node
    int numForwardMessagesDroppedSync = 0;
    int numForwardMessagesSentSync = 0;
    int numBackwardMessagesDroppedSync = 0;
    int numBackwardMessagesSentSync = 0;
    
    // activation histogram
    vector<tbb::atomic<int>> hist;
    
    void insertHist(const Eigen::VectorXf& v){
        for(int i = 0; i < v.size(); i++){
            int x = std::round(v(i)*100)+100;
            if(x>=0 && x<hist.size()){
                hist[x]++;
            } else {
                cout << x << endl;
            }
        }
    }
    
    // Should be in another class, but kept here for simplicity
    int delayInitialiser() override{
        int number = -1;
        while( (number < 1) || (number>100.0) )
            number = (int)distribution(generator);
        return number;
    }
    
public:
    DNNGraphSettings(){
        activationFnc = &math::activationTanH;
        deltaActivationFnc = &math::deltaActivationTanH;
        initWeightsFnc = &weight_init::initWeights;
        regularizationFnc = &math::maxNormRegularization;
        
        actMax = activationFnc(5);
        actMin = activationFnc(-5);
        
        hist = vector<tbb::atomic<int>>(201,tbb::atomic<int>(0));
    }
    
    void setParameters(vector<int>& params) override {
        assert(params.size() == 4);
        nHLayers = params[0];
        nHidden = params[1];
        nInput = params[2];
        nOutput = params[3];
    }
    
    void printParameters(){
        cout << "HIST SIZE: " << hist.size() << endl << endl;
        cout << "Hyperparameters\n"; 
        cout << "lr "    << lr;
        cout << " alpha " << alpha;
        cout << " batchsize " << batchSize;
        cout << " maxEpoch "  << maxEpoch;  
        cout << " numModels "  << numModels << endl;
    }
};

#endif /* DNN_GRAPH_H */

