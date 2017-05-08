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
    
    map<int,int> dstWeightIndex;        // map backprop index to the relevant weight
    
    vector<float> deltas;           // store received delta values
    vector<float> deltaWeights;     // delta weights, for momentum
    vector<float> newWeights;       // new weights to update
    vector<float> weights;
   
    float value = 0;
    float error = 0;
public:
    HiddenNode(shared_ptr<GraphSettings> context): NeuralNode(context){}
    virtual ~HiddenNode(){}
    string getType() override {return HiddenNode::m_type;}
    
    void addEdge(Edge* e) override;
    
    void setWeights(const vector<float>& w) override{
        assert(w.size() == outgoingForwardEdges.size());
        weights = w;
        newWeights = w;
        
        // init size of delta values
        deltas = vector<float>(weights.size());
        deltaWeights = vector<float>(weights.size());
    }

    void onRecv(ForwardPropagationMessage* msg) override;
    void onRecv(BackwardPropagationMessage* msg) override;

    bool sendForwardMsgs(vector<Message*>& msgs) override;
    bool sendBackwardMsgs(vector<Message*>& msgs) override;
private:
    // for populating weights map
    int map_index = 0;
    void initWeights(){
        weights = vector<float>(outgoingForwardEdges.size());
        settings->initWeightsFnc(weights,outgoingForwardEdges.size(),incomingForwardEdges.size());
        newWeights = weights;
        
        // init size of delta values
        deltas = vector<float>(weights.size());
        deltaWeights = vector<float>(weights.size());
    }
};

#endif /* HIDDEN_NODE_H */

