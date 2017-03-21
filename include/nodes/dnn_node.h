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
#include "graphs/graph.h"
#include "misc/node_factory.h"
#include "graphs/dnn_graph.h"

#include <string>
#include <cassert>
#include <cstdio>
#include <exception>
#include <memory>

class DNNNode: public Node{
    static NodeRegister<DNNNode> m_reg;
    static std::string m_type;
    std::shared_ptr<DNNGraph> m_graph;
public:
    int seenCount = 0;
    std::shared_ptr<Message> m_msg;
    DNNNode(shared_ptr<Graph> graph): Node(graph){
        try{
            // Downcast 
            // This is done so the same map can be used for all nodes.
            m_graph = std::static_pointer_cast<DNNGraph>(graph);
        } catch (exception& e){
            printf("%s does not belong to graph type %s",m_type.c_str(),"TODO");
        }
    }
    virtual ~DNNNode(){}
    void onInit(){}
    bool readyToSend(){
        return (seenCount==incomingEdges.size()) && (m_msg != NULL);
    }
    void onRecv(std::shared_ptr<Message>& msg){
        seenCount++;
        m_msg = msg;
    }
    bool onSend(std::shared_ptr<Message>& msg){
        return true;       
    }
};

#endif /* DNN_NODE_H */

