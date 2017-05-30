
/* 
 * File:   dropout.h
 * Author: ryan
 *
 * Created on 16 May 2017, 17:54
 */

#ifndef DROPOUT_H
#define DROPOUT_H

#include "training/dropout_strategy.h"
#include "tools/math.h"

#include <random>
#include <vector>
#include <cassert>
#include <iostream>

using namespace std;

class Dropout: public DropoutStrategy{
private:
    std::mt19937 rng;
    
    // if the node is currently not dropped
    bool active = true;
    bool update = true;
    
    // whether the node is a: input (0), hidden (1), output (2)
    NodeType type;
    
    // Indicies
    int nodeIndex;
    int layerIndex;
    
    // topology
    int prevLayerSize;
    int hiddenLayerSize;
    int nextLayerSize;
    int numLayers;

    // last seen time step
    int lastTime = 0;
    
    // prev layer
    int pBias = -1;
    int pStep = -1;    
    
    // current layer
    int cBias = -1;
    int cStep = -1;
    
    // next layer
    int nBias = -1;
    int nStep = -1;
    
    bool lastLayer = false;
    bool firstLayer = false;
    
    float dropRate = 0;
    int numActive;
public:
    // for hidden and bias nodes
    Dropout(NodeType _type, int _seed, 
            int _prevLayerSize, int _nextLayerSize, int _hiddenLayerSize, 
            int _numLayers, int _layerIndex, int _nodeIndex=-1):
    DropoutStrategy(true),
    type(_type), 
    nodeIndex(_nodeIndex), 
    layerIndex(_layerIndex), 
    prevLayerSize(_prevLayerSize), hiddenLayerSize(_hiddenLayerSize), nextLayerSize(_nextLayerSize), 
    numLayers(_numLayers){
        assert(math::isPrime(hiddenLayerSize));
        assert(nodeIndex < hiddenLayerSize);
        assert(layerIndex <= numLayers);
        
        rng.seed(_seed);
        if(layerIndex==0) firstLayer = true;
        if(layerIndex==numLayers-1) lastLayer = true;
                
        // discard rng
        if(NodeType::bias==type && layerIndex != -1) rng.discard((layerIndex+1)*2);
        else if(!firstLayer && layerIndex != -1) rng.discard((layerIndex-1)*2);
        
        numActive = (float)hiddenLayerSize * (1-dropRate);
    }
    void nextStep(int currTime){    
        assert(currTime >= lastTime);
        int diff = currTime - lastTime;
        for(int i = 0; i < diff; ++i){
            int bitgenCount = 0; 
            if(!firstLayer && NodeType::input != type && NodeType::bias != type){
                setBiasStep(rng, pBias, pStep, hiddenLayerSize);
                bitgenCount++;
            }
            if(NodeType::input != type && NodeType::output != type && NodeType::bias != type){
                setBiasStep(rng, cBias, cStep, hiddenLayerSize);
                bitgenCount++;
            }
            if(!lastLayer && NodeType::output != type){
                setBiasStep(rng, nBias, nStep, hiddenLayerSize);
                bitgenCount++;
            }
            rng.discard((numLayers-bitgenCount)*2);
        }
        lastTime = currTime;
    }
    bool isActive(){
        if(type != NodeType::hidden) return true;
        assert(cBias!=-1 && cStep!=-1);
        return permutation(cBias, cStep*nodeIndex, hiddenLayerSize) < numActive  ? true : false;
    }
    bool isPrevLayerNodeActive(int i){
        if(firstLayer) return true; // input layer always active
        if(i >= hiddenLayerSize) // bias node
            return true;
        assert(pBias!=-1 && pStep!=-1);
        assert( i < hiddenLayerSize && i >= 0);
        return permutation(pBias, pStep*i, hiddenLayerSize) < numActive ? true : false;
    }
    bool isNextLayerNodeActive(int i){
        if(lastLayer) return true; // output layer always active
        assert(nBias!=-1 && nStep != -1);
        return Dropout::permutation(nBias, nStep*i, hiddenLayerSize) < numActive ? true : false;
    }
    bool readyToSendForward(int forwardSeenCount){
        if(type == NodeType::input || type == NodeType::bias) 
            return forwardSeenCount == 1; // sync node connection
        else if(firstLayer)
            return (forwardSeenCount == prevLayerSize + 1) && active; // number of inputs
        else{
            assert(forwardSeenCount <= numActive + 1 );
            return (forwardSeenCount == numActive + 1) && active;
        }
    }
    bool readyToSendBackward(int backwardSeenCount){
        if(type == NodeType::output) 
            return backwardSeenCount == 1; // sync node connection
        else if(lastLayer)
            return backwardSeenCount == nextLayerSize; // number of outputs
        else
            return (backwardSeenCount == numActive) && active;
    }
    static int permutation(int bias, int acc, int prime){
        assert(bias>=0 && bias < prime);
        return (bias + acc) % prime;
    }
    vector<int> getBiasTerms(){
        vector<int> biasTerms{pBias, cBias, nBias};
        return biasTerms;
    }
    vector<int> getStepTerms(){
        vector<int> stepTerms{pStep, cStep, nStep};
        return stepTerms;
    }
private:
    void setBiasStep(std::mt19937& r, int& bias, int& step, int prime){
        bias = r() % prime;
        step = r() % prime; 
        if(step==0) step++; // ideally should repick   
    }
};

#endif /* DROPOUT_H */

