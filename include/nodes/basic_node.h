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
#include "graphs/graph.h"
#include "misc/node_factory.h"
#include "graphs/basic_graph.h"

#include <string>
#include <cassert>
#include <cstdio>
#include <exception>
#include <memory>

class BasicNode : public Node{
    static NodeRegister<BasicNode> m_reg;
    static std::string m_type;
    std::shared_ptr<BasicGraph> m_graph; // global settings for graph
public:
    int seenCount = 0;
    std::shared_ptr<Message> m_msg;
    BasicNode(shared_ptr<Graph> graph): Node(graph){
        // Downcast 
        // This is done so the same map can be used for all nodes.
        try{
            if(m_graph = std::dynamic_pointer_cast<BasicGraph>(graph)){
                std::cout << "cast was fine";
            } else {
                std::cout << "bad cast";
            }
        } catch (exception& e){
            printf("%s does not belong to graph type %s",m_type.c_str(),"TODO");
        }
    }
    virtual ~BasicNode(){}
    void onInit(){}
    bool readyToSend(){
        return (seenCount==incomingEdges.size()) && (m_msg != NULL);
    }
    void onRecv(std::shared_ptr<Message>& msg){
        seenCount++;
        m_msg = msg;
    }
    bool onSend(std::shared_ptr<Message>& msg){
         // check node is ready to send
        assert(readyToSend());
        
        // update state
        seenCount = 0;
        
        // copy message
        msg = m_msg;
        
        return true;       
    }
};

#endif /* BASICNODE_H */

