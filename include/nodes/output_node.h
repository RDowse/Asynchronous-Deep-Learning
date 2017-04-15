/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   output_node.h
 * Author: ryan
 *
 * Created on 23 March 2017, 23:54
 */

#ifndef OUTPUT_NODE_H
#define OUTPUT_NODE_H

#include "nodes/node.h"
#include "graphs/graph_settings.h"
#include "misc/node_factory.h"
#include "graphs/dnn_graph_settings.h"
#include "tools/logging.h"

#include <string>
#include <cassert>
#include <cstdio>
#include <exception>
#include <memory>
#include <typeinfo>

using namespace std;

class OutputNode: public Node{
    static NodeRegister<OutputNode> m_reg;
    static std::string m_type;
    shared_ptr<DNNGraphSettings> m_graph;
    
    shared_ptr<Edge> syncEdge;
    vector<shared_ptr<Edge>> backwardEdges;
public:
    int seenCount = 0;
    float error = 0;
    float target = 0;
    float output = 0;
    float value = 0;
    OutputNode(shared_ptr<GraphSettings> graphSettings): Node(graphSettings){
        try{
            m_graph = std::static_pointer_cast<DNNGraphSettings>(graphSettings);
        } catch (const std::bad_cast& e) {
            std::cout << e.what() << std::endl;
        }
    }
    virtual ~OutputNode(){}
    string getType() override{return OutputNode::m_type;}
    bool readyToSend() override {
        return seenCount == incomingEdges.size()-1;
    }

    void setup() override{
        for(auto& e: outgoingEdges){
            if(e->dst->getType() == "Sync"){
                syncEdge = e;
            } else {
                backwardEdges.push_back(e);
            }
        }
    }

    bool onSend(shared_ptr<ForwardPropagationMessage> msg) override;
    bool onSend(shared_ptr<BackwardPropagationMessage> msg) override;
    
    void onRecv(shared_ptr<ForwardPropagationMessage> msg) override;
    void onRecv(shared_ptr<BackwardPropagationMessage> msg) override;
};


#endif /* OUTPUT_NODE_H */

