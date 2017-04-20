/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   sync_node.h
 * Author: ryan
 *
 * Created on 13 April 2017, 00:47
 */

#ifndef SYNC_NODE_H
#define SYNC_NODE_H

#include "nodes/node.h"
#include "graphs/graph_settings.h"
#include "misc/node_factory.h"
#include "graphs/dnn_graph_settings.h"

#include "misc/data_wrapper.h"

#include <stack>
#include <string>
#include <cassert>
#include <cstdio>
#include <exception>
#include <memory>
#include <typeinfo>

using namespace std;

class SyncNode: public Node{
    static NodeRegister<SyncNode> m_reg;
    static std::string m_type;
    shared_ptr<DNNGraphSettings> m_graph;
    
    DataWrapper* m_dataset;
    
    vector<shared_ptr<Edge>> inputEdges;
    vector<shared_ptr<Edge>> biasEdges;
    vector<shared_ptr<Edge>> outputEdges;
    
    // current operation
    enum Operation{
        forward, backward
    };
    
    stack<pair<int,float>> output;
    
    // operation flags and counts
    int inputSeenCount = 0;
    int outputSeenCount = 0;
    bool tick = true;           // trigger initial message propagation
    bool validating = false;    // flag for propagating validation set
  
    // training 
    int trainingIndex = 0;
    int batchCount = 0;
    int epochCount = 0;
    float error = 0;
    float validation_error = 0;
public:
    SyncNode(shared_ptr<GraphSettings> graphSettings): Node(graphSettings){
        try{
            m_graph = std::static_pointer_cast<DNNGraphSettings>(graphSettings);
        } catch (const std::bad_cast& e) {
            std::cout << e.what() << std::endl;
        }
    }
    virtual ~SyncNode(){}
    string getType() override {return SyncNode::m_type;}
    bool readyToSend() override {
        if(m_graph->cmd == DNNGraphSettings::Command::predict && tick){
            return true;
        } else if(m_graph->cmd == DNNGraphSettings::Command::train && epochCount < m_graph->maxEpoch) {
            return (outputSeenCount == outputEdges.size()) || (inputSeenCount == inputEdges.size()) || tick;
        } 
        return false;
    }

    void setup() override{
        for(auto e: outgoingEdges){
            if(e->dst->getType() == "Input"){
                inputEdges.push_back(e);
            } else if(e->dst->getType() == "Bias") {
                biasEdges.push_back(e);
            } else if(e->dst->getType() == "Output"){
                outputEdges.push_back(e);
            }
        }
    }
    
    void setDataSet(DataWrapper* dataset){
        m_dataset = dataset;
    }
   
    void onRecv(shared_ptr<ForwardPropagationMessage> msg) override;
    void onRecv(shared_ptr<BackwardPropagationMessage> msg) override;
    
    bool dispatchMsgs() override{
        if(DNNGraphSettings::Operation::forward == m_graph->op){
            dispatchForwardMsgs();
        } else if(DNNGraphSettings::Operation::backward == m_graph->op){
            dispatchBackwardMsgs();
        }
    }
    bool dispatchBackwardMsgs();
    bool dispatchForwardMsgs();
};

#endif /* SYNC_NODE_H */