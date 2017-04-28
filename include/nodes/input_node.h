/* 
 * File:   input_node.h
 * Author: ryan
 *
 * Created on 23 March 2017, 23:36
 */

#ifndef INPUT_NODE_H
#define INPUT_NODE_H

#include "nodes/neural_node.h"
#include "graphs/graph_settings.h"
#include "misc/node_factory.h"
#include "graphs/dnn_graph_settings.h"
#include "tools/logging.h"
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

class InputNode: public NeuralNode{
    static NodeRegister<InputNode> m_reg;
    static std::string m_type;
    
    map<int,int> dstWeightIndex;        // map of weights associated to dst ids
    
    vector<float> deltas;
    vector<float> deltaWeights;     // delta weights, for momentum
    vector<float> newWeights;       // intermediate updated weights
    vector<float> weights;
public:
    InputNode(shared_ptr<GraphSettings> graphSettings): NeuralNode(graphSettings){}
    virtual ~InputNode(){}
    string getType() override {return InputNode::m_type;}

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

#endif /* INPUT_NODE_H */

