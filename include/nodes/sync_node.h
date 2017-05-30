/* 
 * File:   sync_node.h
 * Author: ryan
 *
 * Created on 13 April 2017, 00:47
 */

#ifndef SYNC_NODE_H
#define SYNC_NODE_H

#include "nodes/neural_node.h"
#include "graphs/graph_settings.h"
#include "misc/node_factory.h"
#include "graphs/dnn_graph_settings.h"

#include "misc/data_wrapper.h"

#include <unordered_map>
#include <stack>
#include <string>
#include <cassert>
#include <cstdio>
#include <exception>
#include <memory>
#include <typeinfo>
#include <limits>

using namespace std;

class NeuralNode::SyncNode: public NeuralNode{
    static NodeRegister<SyncNode> m_reg;
    
    DataWrapper* dataset;
    
    State* lastState;           // tmp fix for deleting states
    
    bool tick = true;           // trigger initial message propagation
    bool validating = false;    // flag for propagating validation set
  
    // sampling
    int sampleIndex = 0;
    
    int batchSize = 0;
    
    int map_index = 0;
    unordered_map<int,int> dstOutputIndex;        // map backprop index to output
    
    // Error calculations
    Eigen::MatrixXf receivedOutput;
    float training_error = 0;
    float accuracy = 0;
    vector<float> min_error; 
    vector<float> error;
public:
    static std::string m_type;
    SyncNode(shared_ptr<GraphSettings> context): NeuralNode(context){}
    virtual ~SyncNode(){
        if(lastState) delete lastState;
    }
    string getType() override {return SyncNode::m_type;}
    void setDataSet(DataWrapper* ds ){
        dataset = ds;
        min_error = vector<float>(dataset->training_labels.size(),std::numeric_limits<float>::max());
        error = vector<float>(dataset->training_labels.size(),std::numeric_limits<float>::max());
    }
    void addEdge(Edge* e) override;    
    
    bool readyToSendForward() override;
    bool readyToSendBackward() override;
    
    void onRecv(ForwardPropagationMessage* msg) override;
    void onRecv(BackwardPropagationMessage* msg) override;
    
    bool sendBackwardMsgs(vector<Message*>& msgs);
    bool sendForwardMsgs(vector<Message*>& msgs);
    
private:
    template<typename TState>
    void swapState(){
        if(!lastState){
            context->state = new TState();
        } else {
            State* tmpState = lastState;
            lastState = context->state;
            context->state = tmpState;
        }
    }
};

#endif /* SYNC_NODE_H */