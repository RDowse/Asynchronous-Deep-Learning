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
    DataWrapper* dataset;
    
    bool tick = true;           // trigger initial message propagation
    bool validating = false;    // flag for propagating validation set
  
    DataSetType dataSetType = DataSetType::training;
    
    // sampling
    int sampleIndex = 0;
    
    int currBatchSize = 0;
    
    int map_index = 0;
    unordered_map<int,int> dstOutputIndex;        // map backprop index to output
    
    // backpropagation
    Eigen::MatrixXf receivedOutput;
    
    // Error calculations
    float error = 0;
    float accuracy = 0;
public:
    static std::string m_type;
    SyncNode(shared_ptr<GraphSettings> context): NeuralNode(context){}
    string getType() override {return SyncNode::m_type;}
    void setDataSet(DataWrapper* ds ){
        dataset = ds;
    }
    void addEdge(Edge* e) override;    
    
    bool readyToSendForward() override;
    bool readyToSendBackward() override;
    
    void onRecv(ForwardPropagationMessage* msg) override;
    void onRecv(BackwardPropagationMessage* msg) override;
    
    bool sendBackwardMsgs(vector<Message*>& msgs);
    bool sendForwardMsgs(vector<Message*>& msgs);
};

#endif /* SYNC_NODE_H */