
/* 
 * File:   strategy_loader.h
 * Author: ryan
 *
 * Created on 21 May 2017, 23:29
 */

#ifndef STRATEGY_LOADER_H
#define STRATEGY_LOADER_H

#include "nodes/node.h"
#include "nodes/hidden_node.h"
#include "nodes/bias_node.h"
#include "nodes/output_node.h"
#include "nodes/input_node.h"

#include "training/dropout.h"
#include "training/dropout_bitset.h"

#include <vector>
#include <iostream>

using namespace std;

class StrategyLoader{
    typedef DropoutStrategy::NodeType NodeType;
    shared_ptr<DNNGraphSettings> context;
    vector<string> config;
    vector<Node*>* nodesPtr;
public: 
    StrategyLoader(shared_ptr<GraphSettings> c, vector<Node*>* n){
        context = std::static_pointer_cast<DNNGraphSettings>(c);
        nodesPtr = n;
    }
    void setConfig(const vector<string>& c){
        config = c;
    }
    void load(){
        for(auto str: config){
            if("dropout" == str){
                dropout();
            } else if("dropoutbitset" == str) {
                dropoutBitset();
            } else {
                cout << "nothing to load" << endl;
            }
        }
    }
private:
    void dropout(){
        vector<Node*> nodesRef = *nodesPtr;
        int nodeIndex = 0;
        int layerIndex = -1; // start at input layer
        int hiddenLayerSize = context->nHidden;
        int numLayers = context->nHLayers;
        int inputSize = context->nInput;
        int outputSize = context->nOutput;
        for(auto n: nodesRef){
            if(NeuralNode::HiddenNode* h = dynamic_cast<NeuralNode::HiddenNode*>(n)){
                int prevLayerSize=0,nextLayerSize=0;
                if(layerIndex == 0) prevLayerSize = inputSize;
                if(layerIndex == numLayers-1) nextLayerSize = outputSize;
                h->setDropoutStrategy(
                    new Dropout(NodeType::hidden,context->seed,prevLayerSize,
                        nextLayerSize,hiddenLayerSize,numLayers,layerIndex,nodeIndex++)
                );
            } else if(NeuralNode::InputNode* i = dynamic_cast<NeuralNode::InputNode*>(n)){
                i->setDropoutStrategy(
                    new Dropout(NodeType::input,context->seed,-1,hiddenLayerSize,hiddenLayerSize,numLayers,-1)
                );
            } else if(NeuralNode::OutputNode* o = dynamic_cast<NeuralNode::OutputNode*>(n)){
                o->setDropoutStrategy(
                    new Dropout(NodeType::output,context->seed,hiddenLayerSize,-1,hiddenLayerSize,numLayers,numLayers)
                );
            } else if(BiasNode* b = dynamic_cast<BiasNode*>(n)){
                int nextLayerSize = (layerIndex == numLayers-1) ? outputSize : hiddenLayerSize;
                b->setDropoutStrategy(
                    new Dropout(NodeType::bias,context->seed,-1,nextLayerSize,hiddenLayerSize,numLayers,layerIndex)
                );
                layerIndex++;
                nodeIndex = 0;
            }
        }
    }
    
    void dropoutBitset(){
        vector<Node*> nodesRef = *nodesPtr;
        
        const int bufferSize = 500; 
        
        float dropRate = 0.5;
        
        // index
        int nodeIndex = 0;
        int layerIndex = -1; // start at input layer
        
        // topology
        int hiddenLayerSize = context->nHidden;
        int numLayers = context->nHLayers;
        int inputSize = context->nInput;
        int outputSize = context->nOutput;
        
        for(auto n: nodesRef){
            if(NeuralNode::HiddenNode* h = dynamic_cast<NeuralNode::HiddenNode*>(n)){
                int prevLayerSize=0,nextLayerSize=0;
                if(layerIndex == 0) prevLayerSize = inputSize;
                if(layerIndex == numLayers-1) nextLayerSize = outputSize;
                h->setDropoutStrategy(
                    new DropoutBitset<bufferSize>(
                        NodeType::hidden,context->seed,dropRate,
                        prevLayerSize, nextLayerSize, hiddenLayerSize,
                        numLayers,layerIndex,nodeIndex++
                    )
                );
            } else if(NeuralNode::InputNode* i = dynamic_cast<NeuralNode::InputNode*>(n)){
                i->setDropoutStrategy(
                    new DropoutBitset<bufferSize>(
                        NodeType::input,context->seed,dropRate,
                        -1,hiddenLayerSize,hiddenLayerSize,
                        numLayers,-1,-1
                    )
                );
            } else if(NeuralNode::OutputNode* o = dynamic_cast<NeuralNode::OutputNode*>(n)){
                o->setDropoutStrategy(
                    new DropoutBitset<bufferSize>(
                        NodeType::output,context->seed,dropRate,
                        hiddenLayerSize,-1,hiddenLayerSize,
                        numLayers,-1,-1
                    )
                );
            } else if(BiasNode* b = dynamic_cast<BiasNode*>(n)){
                int nextLayerSize = (layerIndex == numLayers-1) ? outputSize : hiddenLayerSize;
                b->setDropoutStrategy(
                    new DropoutBitset<bufferSize>(
                        NodeType::bias,context->seed,dropRate,
                        -1,nextLayerSize,hiddenLayerSize,
                        numLayers,layerIndex,-1
                    )
                );
                layerIndex++;
                nodeIndex = 0;
            }
        }
    }
};

#endif /* STRATEGY_LOADER_H */

