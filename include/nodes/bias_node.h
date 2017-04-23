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
#include "tools/math.h"

#include <stack>
#include <cassert>

using namespace std;

class BiasNode : public Node{
    static NodeRegister<BiasNode> m_reg;
    static std::string m_type;
    shared_ptr<DNNGraphSettings> m_graph; // global settings for graph
    
    Edge* syncEdge;
    vector<Edge*> forwardEdges;
    
    int forwardSeenCount = 0;
    int backwardSeenCount = 0;
    float value = 1;
    stack< pair<int,float> > deltas;
    map<int,int> idIndexMap;        // map backprop index to the relevant weight
    vector<float> newWeights;
    vector<float> deltaWeights;
public:
    vector<float> weights; 
    BiasNode(shared_ptr<GraphSettings> graphSettings): Node(graphSettings){
        try{
            m_graph = std::static_pointer_cast<DNNGraphSettings>(graphSettings);
        } catch (const std::bad_cast& e) {
            std::cout << e.what() << std::endl;
        }
    }
    virtual ~BiasNode(){}
    string getType() override {return BiasNode::m_type;}
    bool readyToSend() override {
        return forwardSeenCount == 1;
    }
    
    void setup() override{
        int i = 0;
        for(auto& e: outgoingEdges){
            if(e->dst->getType()=="Sync"){
                syncEdge = e;
            } else {
                forwardEdges.push_back(e);
                idIndexMap[e->dst->getId()] = i++;
            }
        }
        weights = vector<float>(forwardEdges.size(),0);
        for(auto& w: weights) w = math::randomFloat(-0.1,0.1);
        newWeights = weights;
        deltaWeights = vector<float>(weights.size(),0);
    }
    
    void onRecv(shared_ptr<ForwardPropagationMessage> msg) override;
    void onRecv(shared_ptr<BackwardPropagationMessage> msg) override;
    
    bool onSend(vector< shared_ptr<Message> >& msgs) override;
};


#endif /* BIAS_NODE_H */

