/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   BasicNode.h
 * Author: ryan
 *
 * Created on 09 February 2017, 20:05
 * 
 * Basic node that passes on the current received message to the destination
 * nodes.
 */

#ifndef BASICNODE_H
#define BASICNODE_H

#include "nodes/node.h"
#include "misc/node_factory.h"
#include "graphs/graph_settings.h"
#include "graphs/basic_graph_settings.h"

#include <string>
#include <cassert>
#include <cstdio>
#include <exception>
#include <memory>

using namespace std;

class BasicNode : public Node{
    static NodeRegister<BasicNode> m_reg;
    static std::string m_type;
    shared_ptr<BasicGraphSettings> m_graphSettings; // global settings for graph
public:
    int seenCount = 0;
    BasicNode(shared_ptr<GraphSettings> graphSettings): Node(graphSettings){
        // Downcast 
        // This is done so the same map can be used for all nodes.
        try{
            if(m_graphSettings = std::dynamic_pointer_cast<BasicGraphSettings>(graphSettings)){
                
            } else {std::cerr << "Bad cast for " << m_type << " node\n";}
        } catch (exception& e){
            printf("%s does not belong to graph type %s",m_type.c_str(),"TODO");
        }
    }
    virtual ~BasicNode(){}
    void onInit(){}
    bool readyToSend(){}
    
    bool onSend(shared_ptr<ForwardPropagationMessage> msg) override {
        std::cout << "Sending forward msg\n";
    }
    bool onSend(shared_ptr<BackwardPropagationMessage> msg) override {
        std::cout << "Sending backward msg\n";
    }
    
    void onRecv(shared_ptr<ForwardPropagationMessage> msg) override {
        std::cout << "Receiving forward msg\n";
    }
    void onRecv(shared_ptr<BackwardPropagationMessage> msg) override {
        std::cout << "Receiving backward msg\n";
    }
};

#endif /* BASICNODE_H */

