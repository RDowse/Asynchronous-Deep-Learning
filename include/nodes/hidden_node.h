/* 
 * File:   dnn_node.h
 * Author: ryan
 *
 * Created on 20 March 2017, 23:50
 */

#ifndef HIDDEN_NODE_H
#define HIDDEN_NODE_H

#include "nodes/neural_node.h"
#include "graphs/graph_settings.h"
#include "misc/node_factory.h"
#include "graphs/dnn_graph_settings.h"
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
    static NodeRegister<HiddenNode> m_reg;
    static std::string m_type;
    
    unordered_map<int,int> dstWeightIndex;        // map of weights associated to dst ids
    
    Eigen::VectorXf deltas;           // store received delta values
    Eigen::VectorXf deltaWeights;     // delta weights, for momentum
    Eigen::VectorXf newWeights;       // new weights to update
    Eigen::VectorXf weights;
   
    Eigen::VectorXf value;
    Eigen::VectorXf error;
public:
    HiddenNode(shared_ptr<GraphSettings> context): NeuralNode(context){}
    virtual ~HiddenNode(){}
    string getType() override {return HiddenNode::m_type;}
    
    void addEdge(Edge* e) override;
    
    void setWeights(const vector<float>& w) override{
        assert(w.size() == outgoingForwardEdges.size());
        //weights = Eigen::Map<Eigen::VectorXf>(&w[0],w.size());
        newWeights = weights; 
        
        // init size of delta values
        deltas = Eigen::VectorXf(weights.size());
        deltaWeights = Eigen::VectorXf(weights.size());
    }

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
        newWeights = weights;    
        
        // init size of delta values
        deltas = Eigen::VectorXf::Zero(weights.size());
        deltaWeights = Eigen::VectorXf::Zero(weights.size());
    }
};

#endif /* HIDDEN_NODE_H */

