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

#include <stack>
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
    
    // edge seen count
    int seenCountForward = 0;
    int seenCountBackward = 0;
    
    // backprop
    stack<pair<int,float>> deltas;  // store received delta values
    map<int,int> idIndexMap;        // map backprop index to the relevant weight
    vector<float> newWeights;       // new weights to update
    
    // fwdprop
    float error = 0;
    float value = 0;
    float output = 0;

public:
    vector<float> weights;
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
        bool ready = false;
        if(m_graph->operation==1){
            ready = (seenCountForward==(incomingEdges.size()-forwardEdges.size())); 
        }else if(m_graph->operation==2) {
            ready = (seenCountBackward==backwardEdges.size());
        }
        return ready;
    }
    
    // Setup to be called after the graph is constructed
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
        assert(forwardEdges.size() == backwardEdges.size());
        weights = vector<float>(forwardEdges.size(),1);
        newWeights = vector<float>(forwardEdges.size(),0);
    }

    bool onSend(shared_ptr<ForwardPropagationMessage> msg) override;
    bool onSend(shared_ptr<BackwardPropagationMessage> msg) override;
    
    void onRecv(shared_ptr<ForwardPropagationMessage> msg) override;
    void onRecv(shared_ptr<BackwardPropagationMessage> msg) override;
};

#endif /* DNN_NODE_H */

