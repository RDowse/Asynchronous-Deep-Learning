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

#include <stack>
#include <string>
#include <cassert>
#include <cstdio>
#include <exception>
#include <memory>
#include <typeinfo>

using namespace std;

class InputNode: public Node{
    static NodeRegister<InputNode> m_reg;
    static std::string m_type;
    shared_ptr<DNNGraphSettings> m_graph;
    
    // backprop
    stack<pair<int,float>> deltas;
    map<int,int> idIndexMap;
    vector<float> newWeights;
    
    // Edge back to the sync node
    shared_ptr<Edge> syncEdge;
    vector<shared_ptr<Edge>> forwardEdges;
public:
    int seenCount = 0;
    int forwardSeenCount = 0;
    float input = 0;
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
        bool ready = false;
        if(m_graph->cmd == DNNGraphSettings::Command::predict
                && forwardSeenCount==1){
            ready = true; 
        }
        else if(m_graph->cmd == DNNGraphSettings::Command::train){
            ready = (forwardSeenCount==1) || (seenCount == incomingEdges.size());
        }
        return ready;
    }

    void setup() override{
        // sort edges
        for(auto e: outgoingEdges){
            if(e->dst->getType() == "Sync"){
                syncEdge = e;
            } else {
                forwardEdges.push_back(e);
            }
        }
        // init weights
        weights = vector<float>(forwardEdges.size(),1);
        newWeights = vector<float>(forwardEdges.size(),0);
        for(int i = 0; i < forwardEdges.size(); ++i){
            idIndexMap[forwardEdges[i]->dst->getId()] = i;
        }
    }
    
    bool onSend(shared_ptr<ForwardPropagationMessage> msg) override;
    bool onSend(shared_ptr<BackwardPropagationMessage> msg) override;
    
    void onRecv(shared_ptr<ForwardPropagationMessage> msg) override;
    void onRecv(shared_ptr<BackwardPropagationMessage> msg) override;
};

#endif /* INPUT_NODE_H */

