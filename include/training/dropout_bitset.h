
/* 
 * File:   dropout_bitset.h
 * Author: ryan
 *
 * Created on 25 May 2017, 15:58
 */

#ifndef DROPOUT_BITSET_H
#define DROPOUT_BITSET_H

#include "training/dropout_strategy.h"

#include <random>
#include <bitset>
#include <cassert>
#include <iostream>

template<size_t bufferSize>
class DropoutBitset: public DropoutStrategy{
private:
    std::bitset<bufferSize> prevbitset;
    std::bitset<bufferSize> currbitset;
    std::bitset<bufferSize> nextbitset;
    
    std::mt19937 rng;
    
    NodeType type;
    
    int seed;
    float dropRate;
    
    // Topology
    int numLayers;
    int prevLayerSize;
    int nextLayerSize;
    int hiddenLayerSize;
    
    // Indicies
    int layerIndex;
    int nodeIndex;
    
    int lastTime = 0;
    
    bool firstLayer = false;
    bool lastLayer = false;
    
    int numActive;
public:
    DropoutBitset(NodeType _type, int _seed, float _dropRate, 
            int _prevLayerSize, int _nextLayerSize, int _hiddenLayerSize, 
            int _numLayers, int _layerIndex, int _nodeIndex):
        seed(_seed), type(_type), dropRate(_dropRate), numLayers(_numLayers), 
        prevLayerSize(_prevLayerSize), nextLayerSize(_nextLayerSize), hiddenLayerSize(_hiddenLayerSize),
        layerIndex(_layerIndex), nodeIndex(_nodeIndex) 
    {
        rng.seed(seed);
        if(layerIndex == 0) firstLayer = true;
        if(layerIndex == numLayers - 1) lastLayer = true;
        
        // discard rng
        if(NodeType::bias==type && layerIndex != -1) rng.discard((layerIndex+1)*(hiddenLayerSize-1));
        else if(!firstLayer && layerIndex != -1) rng.discard((layerIndex-1)*(hiddenLayerSize-1));
        
        numActive = (1-dropRate)*hiddenLayerSize;
        
        initBitset(prevbitset,numActive);
        initBitset(currbitset,numActive);
        initBitset(nextbitset,numActive);
    }
    void nextStep(int currTime){
        assert(currTime >= lastTime);
        int diff = currTime - lastTime;
        for(int i = 0; i < diff; ++i){
            int bitgenCount = 0; 
            
            if(!firstLayer && NodeType::input != type  && NodeType::bias != type){
                shuffle(rng,prevbitset,hiddenLayerSize);
                bitgenCount++;
            }
            if(NodeType::input != type && NodeType::output != type  && NodeType::bias != type){
                shuffle(rng,currbitset,hiddenLayerSize);
                bitgenCount++;
            }
            if(!lastLayer && NodeType::output != type){
                shuffle(rng,nextbitset,hiddenLayerSize);
                bitgenCount++;
            }
            
            assert((hiddenLayerSize-1)*(numLayers-bitgenCount)>=0);
            rng.discard((hiddenLayerSize-1)*(numLayers-bitgenCount));
        }
        lastTime = currTime;
    }
    bool isActive(){
        if(NodeType::hidden != type || !enabled) return true;
        return currbitset[nodeIndex];
    }
    bool isPrevLayerNodeActive(int i){
        if(firstLayer || i == prevLayerSize || !enabled) return true;
        return prevbitset[i];
    }
    bool isNextLayerNodeActive(int i){
        if(lastLayer || !enabled) return true;
        return nextbitset[i];
    }
    bool readyToSendForward(int forwardSeenCount){
        if(NodeType::bias == type|| NodeType::input == type) return forwardSeenCount == 1;
        else if(firstLayer || !enabled) return forwardSeenCount == prevLayerSize + 1;
        else return (forwardSeenCount == numActive + 1);
    }
    bool readyToSendBackward(int backwardSeenCount){
        if(NodeType::output == type) return backwardSeenCount == 1;
        else if(lastLayer || !enabled) return backwardSeenCount == nextLayerSize;
        else return backwardSeenCount == numActive;
    }
    // Durstenfeld implementation of Fisher-Yates shuffle
    template<size_t size>
    void shuffle(std::mt19937& rng, std::bitset<size>& b, int n){
        for(int i = n-1; i > 0; i--){
            int index = rng() % i;
            if(b[i] != b[index]){
                b[index].flip();
                b[i].flip();
            }
        }
    }
    template<size_t size>
    void initBitset(std::bitset<size>& b, int n){   
        assert(n <= size);
        for(int i = 0; i < n; ++i) b[i].flip();
    }
};

#endif /* DROPOUT_BITSET_H */

