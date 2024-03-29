
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

#include "nodes/pardata_nodes/parallel_data_neural_node.h"
#include "nodes/pardata_nodes/parallel_data_bias_node.h"
#include "nodes/pardata_nodes/parallel_data_hidden_node.h"
#include "nodes/pardata_nodes/parallel_data_output_node.h"
#include "nodes/pardata_nodes/parallel_data_input_node.h"

#include "nodes/async_nodes/async_neural_node.h"
#include "nodes/async_nodes/async_bias_node.h"
#include "nodes/async_nodes/async_hidden_node.h"
#include "nodes/async_nodes/async_input_node.h"
#include "nodes/async_nodes/async_output_node.h"
#include "nodes/async_nodes/async_sync_node.h"

#include "training/dropout.h"
#include "training/dropout_bitset.h"

#include <vector>
#include <iostream>

using namespace std;

template<typename TNode>
class StrategyLoader{
    typedef typename TNode::InputNode InputNode;
    typedef typename TNode::OutputNode OutputNode;
    typedef typename TNode::HiddenNode HiddenNode;
    typedef typename TNode::BiasNode BiasNode;
    typedef typename TNode::SyncNode SyncNode;
    
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
                cout << "No dropout strategy to load" << endl;
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
            if(HiddenNode* h = dynamic_cast<HiddenNode*>(n)){
                int prevLayerSize=hiddenLayerSize,nextLayerSize=hiddenLayerSize;
                if(layerIndex == 0) prevLayerSize = inputSize;
                if(layerIndex == numLayers-1) nextLayerSize = outputSize;
                h->setDropoutStrategy(
                    new Dropout(NodeType::hidden,context->seed,prevLayerSize,
                        nextLayerSize,hiddenLayerSize,numLayers,layerIndex,nodeIndex++)
                );
            } else if(InputNode* i = dynamic_cast<InputNode*>(n)){
                i->setDropoutStrategy(
                    new Dropout(NodeType::input,context->seed,-1,hiddenLayerSize,hiddenLayerSize,numLayers,-1)
                );
            } else if(OutputNode* o = dynamic_cast<OutputNode*>(n)){
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
            if(HiddenNode* h = dynamic_cast<HiddenNode*>(n)){
                int prevLayerSize=hiddenLayerSize,nextLayerSize=hiddenLayerSize;
                if(layerIndex == 0) prevLayerSize = inputSize;
                if(layerIndex == numLayers-1) nextLayerSize = outputSize;
                h->setDropoutStrategy(
                    new DropoutBitset<bufferSize>(
                        NodeType::hidden,context->seed,dropRate,
                        prevLayerSize, nextLayerSize, hiddenLayerSize,
                        numLayers,layerIndex,nodeIndex++
                    )
                );
            } else if(InputNode* i = dynamic_cast<InputNode*>(n)){
                i->setDropoutStrategy(
                    new DropoutBitset<bufferSize>(
                        NodeType::input,context->seed,dropRate,
                        -1,hiddenLayerSize,hiddenLayerSize,
                        numLayers,-1,-1
                    )
                );
            } else if(OutputNode* o = dynamic_cast<OutputNode*>(n)){
                o->setDropoutStrategy(
                    new DropoutBitset<bufferSize>(
                        NodeType::output,context->seed,dropRate,
                        hiddenLayerSize,-1,hiddenLayerSize,
                        numLayers,numLayers,-1
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

