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
    shared_ptr<DNNGraphSettings> m_graphSettings; // global settings for graph
public:
    int seenCount = 0;
    bool sent = false;
    float value = 1;
    vector<float> weights; 
    BiasNode(shared_ptr<GraphSettings> graphSettings): Node(graphSettings){
        // Downcast 
        // This is done so the same map can be used for all nodes.
        try{
            if(m_graphSettings = std::dynamic_pointer_cast<DNNGraphSettings>(graphSettings)){
                
            } else {std::cerr << "Bad cast for " << m_type << " node\n";}
        } catch (exception& e){
            printf("%s does not belong to graph type %s",m_type.c_str(),"TODO");
        }
    }
    virtual ~BiasNode(){}
    string getType() override {return BiasNode::m_type;}
    bool readyToSend() override {
        return !sent;
    }
    
    void setup() override{
        weights = vector<float>(outgoingEdges.size(),1);
    }
    
    bool onSend(shared_ptr<ForwardPropagationMessage> msg) override;
    bool onSend(shared_ptr<BackwardPropagationMessage> msg) override;
    
    void onRecv(shared_ptr<ForwardPropagationMessage> msg) override;
    void onRecv(shared_ptr<BackwardPropagationMessage> msg) override;
};


#endif /* BIAS_NODE_H */

