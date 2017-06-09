/* 
 * File:   input_node.h
 * Author: ryan
 *
 * Created on 23 March 2017, 23:36
 */

#ifndef ASYNC_INPUT_NODE_H
#define ASYNC_INPUT_NODE_H

#include "nodes/async_nodes/async_neural_node.h"
#include "states/backward_train_state.h"
#include "states/forward_train_state.h"
#include "tools/logging.h"
#include "tools/math.h"
#include "common.h"

#include <stack>
#include <string>
#include <cassert>
#include <cstdio>
#include <exception>
#include <memory>
#include <typeinfo>
#include <random>
#include <functional>

using namespace std;

class AsyncNeuralNode::InputNode: public AsyncNeuralNode{
    unordered_map<int,int> dstWeightIndex;        // map of weights associated to dst ids
    
    Eigen::MatrixXf partialReceivedDelta;
    Eigen::MatrixXf receivedDelta;
    Eigen::VectorXf deltaWeights;     // delta weights, for momentum
    Eigen::VectorXf weights;
public:
    static std::string m_type;
    InputNode(shared_ptr<GraphSettings> context): AsyncNeuralNode(context){}
    virtual ~InputNode(){}
    string getType() override {return InputNode::m_type;}

    void addEdge(Edge* e) override;
    
    bool readyToSendForward() override{
        return forwardSeenCount == incomingForwardEdges.size();
    }
    
    void setWeights(const vector<float>& w) override{  }
        
    void onRecv(ForwardPropagationMessage* msg) override;
    void onRecv(BackwardPropagationMessage* msg) override;

    bool sendForwardMsgs(vector<Message*>& msgs) override;
    bool sendBackwardMsgs(vector<Message*>& msgs) override; 

private:
    // for populating weights map
    int map_index = 0;
    void initWeights(){
        weights = Eigen::VectorXf::Zero(outgoingForwardEdges.size());
        context->initWeightsFnc(weights,outgoingForwardEdges.size(),incomingForwardEdges.size());  
        
        // init size of delta values
        deltaWeights = Eigen::VectorXf::Zero(weights.size());
    }
};

#endif /* INPUT_NODE_H */

