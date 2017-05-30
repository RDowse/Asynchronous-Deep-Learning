
/* 
 * File:   block_neural_network_settings.h
 * Author: ryan
 *
 * Created on 07 May 2017, 18:30
 */

#ifndef BLOCK_NEURAL_NETWORK_SETTINGS_H
#define BLOCK_NEURAL_NETWORK_SETTINGS_H

#include "misc/weight_initialisers.h"
#include "graphs/graph_settings.h"
#include "states/state.h"

// Note: context design pattern, maintains and shared information for the system 
class BlockNeuralNetworkSettings: public GraphSettings{
public:
    // arangement of nodes (number of nodes in each layer)
    // eg. numInput, numHidden, numOutput
    vector< int > netTopology;    
    // arangement of blocks (number of nodes in each block)
    // eg. inputSizes -> hiddenSizes -> outputSizes
    vector< vector<int> > blockTopology;    
    
    // Training param
    float lr = 0.1;            // learning rate
    float alpha = 0.5;          // momentum
    int sample = 0;             // selected sample for predicting
    int maxEpoch = 100000;           // maximum epochs for training
    float minError = 0.01;      // minimum error to stop training
    int miniBatchSize = 1;
    
    // Block weight initilisation.
    void (*initWeightsFnc)(Eigen::MatrixXf& ,int ,int);  
    
    // Activation function
    float (*activationFnc)(float);
    
    // Differentiated Activation function
    //float (*deltaActivationFnc)(float);
    Eigen::MatrixXf (*deltaActivationFnc)(Eigen::MatrixXf);
    
    // Flags
    bool update = false;        // flag for updating weight. TODO: update with msgs
    
    BlockNeuralNetworkSettings(){
        activationFnc = &math::activationSig;
        deltaActivationFnc = &math::blockDeltaActivationSig; //&math::deltaActivationSig;
        initWeightsFnc = &weight_init::initBlockWeights;
    }
    
    void setParameters(vector<int>& params) override {
        assert(params.size() == 7);
        // Assertion checking
        if(!blockTopology.empty()) assert(0);
        if(!netTopology.empty()) assert(0);

        // Unpack parameters
        int numHiddenLayers = params[0];
        int numHidden = params[1];
        int numInput = params[2];
        int numOutput = params[3];
        int numBlockHidden = params[4];
        int numBlockInput = params[5];
        int numBlockOutput = params[6];
        int totalNumLayers = numHiddenLayers + 2;

        // Construct net topology
        for(int i = 0; i < totalNumLayers; ++i) {
            if(i==0) netTopology.push_back(numInput); // Input
            else if(i==totalNumLayers-1) netTopology.push_back(numOutput); // Output
            else netTopology.push_back(numHidden);    // Hidden
        }

        // Warning! issues may arise if the input or layer size is not evenly divisible
        assert(numInput%numBlockInput==0);
        assert(numOutput%numBlockOutput==0);
        assert(numHidden%numBlockHidden==0);

        // Construct block topology
        for(int i = 0; i < totalNumLayers; ++i) {
            if(i==0){ // Input
                blockTopology.push_back(vector<int>(numBlockInput,numInput/numBlockInput)); 
            } else if(i==totalNumLayers-1) { // Output
                blockTopology.push_back(vector<int>(numBlockOutput,numOutput/numBlockOutput)); 
            } else { // Hidden
                blockTopology.push_back(vector<int>(numBlockHidden,numHidden/numBlockHidden));
            }
        }
    };
};


#endif /* BLOCK_NEURAL_NETWORK_SETTINGS_H */

