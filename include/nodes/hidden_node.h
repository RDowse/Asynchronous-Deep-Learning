/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   dnn_node.h
 * Author: ryan
 *
 * Created on 20 March 2017, 23:50
 */

#ifndef HIDDEN_NODE_H
#define HIDDEN_NODE_H

#include "nodes/node.h"
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

class HiddenNode: public Node{
    static NodeRegister<HiddenNode> m_reg;
    static std::string m_type;
    shared_ptr<DNNGraphSettings> m_graph;
    
    // edge seen count
    int forwardSeenCount = 0;
    int backwardSeenCount = 0;
    
    // backprop
    stack<pair<int,float>> deltas;  // store received delta values
    map<int,int> idIndexMap;        // map backprop index to the relevant weight
    vector<float> deltaWeights;     // delta weights, for momentum
    vector<float> newWeights;       // new weights to update
   
    // fwdprop
    float error = 0;
    float value = 0;
    float output = 0;
    
    vector<shared_ptr<Edge>> forwardEdges;
    vector<shared_ptr<Edge>> backwardEdges;
public:
    vector<float> weights;
    HiddenNode(shared_ptr<GraphSettings> graphSettings): Node(graphSettings){
        try{
            m_graph = std::static_pointer_cast<DNNGraphSettings>(graphSettings);
        } catch (const std::bad_cast& e) {
            std::cout << e.what() << std::endl;
        }
    }
    virtual ~HiddenNode(){}
    string getType() override {return HiddenNode::m_type;}
    bool readyToSend() override {
        if(m_graph->cmd == DNNGraphSettings::Command::predict){
            return (forwardSeenCount==(incomingEdges.size()-forwardEdges.size())); 
        }else if(m_graph->cmd == DNNGraphSettings::Command::train) {
            // refactor, far too confusing
            return (forwardSeenCount==(incomingEdges.size()-forwardEdges.size())) 
                    || (backwardSeenCount==forwardEdges.size());
        }
        return false;
    }
    
    void setup() override {
        int i = 0;
        for(auto& e: outgoingEdges){
            if(e->dst->getId() > m_id){ // change based on type of edge.
                forwardEdges.push_back(e);
                idIndexMap[e->dst->getId()] = i++;
            } else {
                backwardEdges.push_back(e);
            }
        }
        weights = vector<float>(forwardEdges.size(),0);
        float maxW = 1/sqrt(backwardEdges.size());
        for(auto& w: weights) w = math::randomFloat(-maxW,maxW);
        newWeights = weights;
        deltaWeights = vector<float>(weights.size(),0);
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

#endif /* HIDDEN_NODE_H */

