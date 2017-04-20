/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   input_node.h
 * Author: ryan
 *
 * Created on 23 March 2017, 23:36
 */

#ifndef INPUT_NODE_H
#define INPUT_NODE_H

#include "nodes/node.h"
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

class InputNode: public Node{
    static NodeRegister<InputNode> m_reg;
    static std::string m_type;
    shared_ptr<DNNGraphSettings> m_graph;
    
    // backpropagation vars
    stack<pair<int,float>> deltas;  // delta step
    map<int,int> idIndexMap;        // map of weights associated to dst ids
    vector<float> newWeights;       // intermediate updated weights
    
    // sorted edges
    shared_ptr<Edge> forwardSyncEdge;
    shared_ptr<Edge> backwardSyncEdge;
    vector<shared_ptr<Edge>> forwardEdges;
    vector<shared_ptr<Edge>> backwardEdges;
    
    // seen counts
    int forwardSeenCount = 0;
    int backwardSeenCount = 0;
    
    // node values
    float input = 0;
public:
    vector<float> weights;
    InputNode(shared_ptr<GraphSettings> graphSettings): Node(graphSettings){
        try{
            m_graph = std::static_pointer_cast<DNNGraphSettings>(graphSettings);
        } catch (const std::bad_cast& e) {
            std::cout << e.what() << std::endl;
        }
    }
    virtual ~InputNode(){}
    string getType() override {return InputNode::m_type;}
    bool readyToSend() override {
        if(m_graph->cmd == DNNGraphSettings::Command::predict){
            return (forwardSeenCount == 1); 
        }
        else if(m_graph->cmd == DNNGraphSettings::Command::train){
            return (forwardSeenCount == 1) || (backwardSeenCount == backwardEdges.size()-1);
        }
        return false;
    }

    void setup() override{
        // sort edges
        for(auto e: outgoingEdges){
            if(e->dst->getType() == "Sync"){
                backwardSyncEdge = e;
            } else {
                forwardEdges.push_back(e);
            }
        }
        for(auto e: incomingEdges){
            if(e->dst->getType() == "Sync"){
                forwardSyncEdge = e;
            } else {
                backwardEdges.push_back(e);
            }
        }
        // init weights
        weights = vector<float>(forwardEdges.size(),0);
        for(auto& w: weights) w = math::randomFloat(0.0,1.0);
        newWeights = weights;
        // map weight index to the corresponding dst edge/node
        for(int i = 0; i < forwardEdges.size(); ++i){
            idIndexMap[forwardEdges[i]->dst->getId()] = i;
        }
    }
    
    void onRecv(shared_ptr<ForwardPropagationMessage> msg) override;
    void onRecv(shared_ptr<BackwardPropagationMessage> msg) override;
    
    bool dispatchMsgs() override{
        if(DNNGraphSettings::Operation::forward == m_graph->op){
            dispatchForwardMsgs();
        } else if(DNNGraphSettings::Operation::backward == m_graph->op){
            dispatchBackwardMsgs();
        }
    }
    bool dispatchBackwardMsgs();
    bool dispatchForwardMsgs();
};

#endif /* INPUT_NODE_H */

