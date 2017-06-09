/* 
 * File:   dnn_node.h
 * Author: ryan
 *
 * Created on 20 March 2017, 23:50
 */

#ifndef ASYNC_HIDDEN_NODE_H
#define ASYNC_HIDDEN_NODE_H

#include "nodes/async_nodes/async_neural_node.h"
#include "states/backward_train_state.h"
#include "states/forward_train_state.h"
#include "graphs/dnn_graph_settings.h"
#include "graphs/graph_settings.h"

#include "tools/math.h"

#include <eigen3/Eigen/Dense>
#include <stack>
#include <string>
#include <cassert>
#include <cstdio>
#include <exception>
#include <memory>
#include <typeinfo>
#include <random>

using namespace std;

class AsyncNeuralNode::HiddenNode: public AsyncNeuralNode{
    // for populating weights map
    int map_index = 0;
    unordered_map<int,int> dstWeightIndex;        // map of weights associated to dst ids
    
    Eigen::MatrixXf receivedDelta;    // store received delta values
    
    Eigen::VectorXf deltaWeights;     // delta weights, for momentum
    Eigen::VectorXf weights;          // current weights
   
    Eigen::VectorXf input;
public:    
    static std::string m_type;
    HiddenNode(shared_ptr<GraphSettings> context): AsyncNeuralNode(context){}
    virtual ~HiddenNode(){}
    string getType() override {return HiddenNode::m_type;}
    
    void addEdge(Edge* e) override;
    
    void setWeights(const vector<float>& w) override{ }

    void onRecv(ForwardPropagationMessage* msg) override;
    void onRecv(BackwardPropagationMessage* msg) override;

    bool sendForwardMsgs(vector<Message*>& msgs) override;
    bool sendBackwardMsgs(vector<Message*>& msgs) override;
private:
    void initWeights(){
        weights = Eigen::VectorXf::Zero(outgoingForwardEdges.size());
        context->initWeightsFnc(weights,outgoingForwardEdges.size(),incomingForwardEdges.size());
        deltaWeights = Eigen::VectorXf::Zero(weights.size());
    }
};

#endif /* HIDDEN_NODE_H */

