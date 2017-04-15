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
    
    //MNISTDatasetWrapper* m_dataset;
    MNISTDataset* m_dataset;
    
    vector<shared_ptr<Edge>> inputEdges;
    vector<shared_ptr<Edge>> biasEdges;
    vector<shared_ptr<Edge>> outputEdges;
    
    int seenCount = 0;
    bool tick = true; // when to trigger message propagation
    
    // training 
    int batchCount = 0;
    int epochCount = 0;
    float error = 0;
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
        } else if(m_graph->cmd == DNNGraphSettings::Command::train 
                && (seenCount == incomingEdges.size() || tick)) {
            cout << incomingEdges.size() << endl;
            cout << tick << " " << seenCount << endl;
            return true;
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
        if(inputEdges.size()==0){tick=false;} // check if its the first or last node.
    }
    
    void setDataSet(MNISTDataset* dataset){
        m_dataset = dataset;
    }
    
    bool onSend(shared_ptr<ForwardPropagationMessage> msg) override;
    bool onSend(shared_ptr<BackwardPropagationMessage> msg) override;
    
    void onRecv(shared_ptr<ForwardPropagationMessage> msg) override;
    void onRecv(shared_ptr<BackwardPropagationMessage> msg) override;
};

#endif /* SYNC_NODE_H */

