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
public:
    int seenCount = 0;
    shared_ptr<Message> m_msg;
    InputNode(shared_ptr<GraphSettings> graphSettings): Node(graphSettings){
        try{
            // Downcast 
            // This is done so the same map can be used for all nodes.
            if(m_graph = std::static_pointer_cast<DNNGraphSettings>(graphSettings)){
                
            } else {std::cerr << "Bad cast for " << m_type << " node";}
        } catch (exception& e){
            printf("%s does not belong to graph type %s",m_type.c_str(),"TODO");
        }
    }
    virtual ~InputNode(){}
    string getType() override {return InputNode::m_type;}
    bool readyToSend() override {}

    bool onSend(shared_ptr<ForwardPropagationMessage> msg) override {}
    bool onSend(shared_ptr<BackwardPropagationMessage> msg) override {}
    
    void onRecv(shared_ptr<ForwardPropagationMessage> msg) override {}
    void onRecv(shared_ptr<BackwardPropagationMessage> msg) override {}
};

#endif /* INPUT_NODE_H */

