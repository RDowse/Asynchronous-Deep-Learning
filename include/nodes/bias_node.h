/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   bias_node.h
 * Author: ryan
 *
 * Created on 04 April 2017, 22:54
 */

#ifndef BIAS_NODE_H
#define BIAS_NODE_H

#include "nodes/node.h"
#include "misc/node_factory.h"
#include "graphs/graph_settings.h"
#include "graphs/dnn_graph_settings.h"

#include <cassert>

using namespace std;

class BiasNode : public Node{
    static NodeRegister<BiasNode> m_reg;
    static std::string m_type;
    shared_ptr<DNNGraphSettings> m_graph; // global settings for graph
    
    shared_ptr<Edge> syncEdge;
    vector<shared_ptr<Edge>> forwardEdges;
    
    int seenCount = 0;
    float value = 1;
public:
    vector<float> weights; 
    BiasNode(shared_ptr<GraphSettings> graphSettings): Node(graphSettings){
        try{
            m_graph = std::static_pointer_cast<DNNGraphSettings>(graphSettings);
        } catch (const std::bad_cast& e) {
            std::cout << e.what() << std::endl;
        }
    }
    virtual ~BiasNode(){}
    string getType() override {return BiasNode::m_type;}
    bool readyToSend() override {
        return seenCount == 1;
    }
    
    void setup() override{
        for(int i = 0; i < outgoingEdges.size(); ++i){
            if(outgoingEdges[i]->dst->getType()=="Sync"){
                syncEdge = outgoingEdges[i];
            } else {
                forwardEdges.push_back(outgoingEdges[i]);
            }
        }
        weights = vector<float>(forwardEdges.size(),1);
    }
    
    void onRecv(shared_ptr<ForwardPropagationMessage> msg) override;
    void onRecv(shared_ptr<BackwardPropagationMessage> msg) override;
    
    bool dispatchMsgs() override;
};


#endif /* BIAS_NODE_H */

