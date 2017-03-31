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

#ifndef DNN_NODE_H
#define DNN_NODE_H

#include "nodes/node.h"
#include "graphs/graph_settings.h"
#include "misc/node_factory.h"
#include "graphs/dnn_graph_settings.h"

#include <string>
#include <cassert>
#include <cstdio>
#include <exception>
#include <memory>
#include <typeinfo>

using namespace std;

class DNNNode: public Node{
    static NodeRegister<DNNNode> m_reg;
    static std::string m_type;
    
    shared_ptr<DNNGraphSettings> m_graph;
    vector<shared_ptr<Edge>> forwardEdges;
    vector<shared_ptr<Edge>> backwardEdges;
    vector<float> weights;
    
    float value = 0;
    int seenCount = 0;
    int seenCountForward = 0;
    int seenCountBackward = 0;
    shared_ptr<Message> m_msg;
public:
    DNNNode(shared_ptr<GraphSettings> graphSettings): Node(graphSettings){
        try{
            // Downcast 
            // This is done so the same map can be used for all nodes.
            if(m_graph = std::static_pointer_cast<DNNGraphSettings>(graphSettings)){
                
            } else {std::cerr << "Bad cast for " << m_type << " node";}
        } catch (exception& e){
            printf("%s does not belong to graph type %s",m_type.c_str(),"TODO");
        }
    }
    virtual ~DNNNode(){}
    string getType() override {return DNNNode::m_type;}
    bool readyToSend() override {
        return (seenCountForward==forwardEdges.size() ||
                seenCountBackward==backwardEdges.size()) 
                && (m_msg != NULL);
    }
    
    void setup() override{
        for(auto& e: outgoingEdges){
            if(e->dst->getId() > m_id){
                forwardEdges.push_back(e);
            } else {
                backwardEdges.push_back(e);
            }
        }
        
        weights = vector<float>(outgoingEdges.size(),1);
    }

    bool onSend(shared_ptr<ForwardPropagationMessage> msg) override;
    bool onSend(shared_ptr<BackwardPropagationMessage> msg) override;
    
    void onRecv(shared_ptr<ForwardPropagationMessage> msg) override;
    void onRecv(shared_ptr<BackwardPropagationMessage> msg) override;
};

#endif /* DNN_NODE_H */

