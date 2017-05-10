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
    
    bool tick = true;           // trigger initial message propagation
    bool validating = false;    // flag for propagating validation set
  
    int sampleIndex = 0;
    int epochCount = 0;
    
    vector<float> min_error; 
    vector<float> error;
    
    vector<int> trainingIndices;
    
    map<int,int> dstOutputIndex;        // map backprop index to output
    vector<float> out;
    
    float actMax = settings->activationFnc(5);
    float actMin = settings->activationFnc(-5);
    
    float training_error = 0;
    
    // random number generation
    std::default_random_engine engine = std::default_random_engine{};
public:
    static std::string m_type;
    SyncNode(shared_ptr<GraphSettings> settings): NeuralNode(settings){}
    virtual ~SyncNode(){}
    
    string getType() override {return SyncNode::m_type;}
    void setDataSet(DataWrapper* ds ){
        dataset = ds;
        trainingIndices.reserve(dataset->training_labels.size());
        for(int i = 0; i < dataset->training_labels.size(); ++i)
            trainingIndices.push_back(i);
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
    int map_index = 0;
};

#endif /* SYNC_NODE_H */