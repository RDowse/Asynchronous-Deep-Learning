/* 
 * File:   dnn_node.h
 * Author: ryan
 *
 * Created on 20 March 2017, 23:50
 */

#ifndef HIDDEN_NODE_H
#define HIDDEN_NODE_H

#include "nodes/neural_node.h"
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
#include <functional>

using namespace std;

class NeuralNode::HiddenNode: public NeuralNode{
    // for populating weights map
    int map_index = 0;
    unordered_map<int,int> dstWeightIndex;        // map of weights associated to dst ids
    
    Eigen::MatrixXf receivedDelta;    // store received delta values
    Eigen::VectorXf deltaWeights;     // delta weights, for momentum
    Eigen::VectorXf weights;          // current weights
   
    Eigen::VectorXf input;
public:    
    static std::string m_type;
    HiddenNode(shared_ptr<GraphSettings> context): NeuralNode(context){}
    virtual ~HiddenNode(){}
    string getType() override {return HiddenNode::m_type;}
    
    void addEdge(Edge* e) override;
    
    void setWeights(const vector<float>& w) override{ assert(0); }

    void onRecv(ForwardPropagationMessage* msg) override;
    void onRecv(BackwardPropagationMessage* msg) override;

    bool sendForwardMsgs(vector<Message*>& msgs) override;
    bool sendBackwardMsgs(vector<Message*>& msgs) override;
private:
    void initWeights(){
        weights = Eigen::VectorXf::Zero(outgoingForwardEdges.size());
        context->initWeightsFnc(weights,outgoingForwardEdges.size(),incomingForwardEdges.size());
        
        // init size of delta values
        deltaWeights = Eigen::VectorXf::Zero(weights.size());
    }
};

#endif /* HIDDEN_NODE_H */

