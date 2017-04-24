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

#include "nodes/neural_node.h"
#include "graphs/graph_settings.h"
#include "misc/node_factory.h"
#include "graphs/dnn_graph_settings.h"
#include "tools/logging.h"
#include "tools/math.h"

#include <stack>
#include <string>
#include <cassert>
#include <cstdio>
#include <exception>
#include <memory>
#include <typeinfo>
#include <random>
#include <functional>

using namespace std;

class InputNode: public NeuralNode{
    static NodeRegister<InputNode> m_reg;
    static std::string m_type;
    
    // backpropagation vars
    stack<pair<int,float>> deltas;  // delta step
    map<int,int> idIndexMap;        // map of weights associated to dst ids
    
    vector<float> deltaWeights;     // delta weights, for momentum
    vector<float> newWeights;       // intermediate updated weights
    vector<float> weights;
    
    // node values
    float input = 0;
    
public:
    InputNode(shared_ptr<GraphSettings> graphSettings): NeuralNode(graphSettings){
        try{
            settings = std::static_pointer_cast<DNNGraphSettings>(graphSettings);
        } catch (const std::bad_cast& e) {
            std::cout << e.what() << std::endl;
        }
    }
    virtual ~InputNode(){}
    string getType() override {return InputNode::m_type;}

    virtual void addEdge(Edge* e) override {
        //int i = 0; // refactor
        // add to original edge sets
        Node::addEdge(e);
        // check edge belongs to this node
        if(e->src->getId() == m_id){
            if(e->dst->getType() == "Sync"){
                outgoingBackwardEdges.push_back(e);
            } else if(e->dst->getType() == "Hidden"
                    || e->dst->getType() == "Output"){
                outgoingForwardEdges.push_back(e);
                //idIndexMap[e->dst->getId()] = i++;
            } else {
                cout << "Unknown type " << e->dst->getType() << endl;
                assert(0);
            }
        } else if(e->dst->getId() == m_id){
            if(e->src->getType() == "Sync"){
                incomingForwardEdges.push_back(e);
            } else if(e->src->getType() == "Hidden"
                    || e->src->getType() == "Output"){
                incomingBackwardEdges.push_back(e);
            } else {
                cout << "Unknown type " << e->dst->getType() << endl;
                assert(0);
            }
        }
        
        // tmp tests
        assert(outgoingBackwardEdges.size() <= 1);        
        assert(incomingForwardEdges.size() <= 1);
    }
    
    void setWeights(const vector<float>& w) override{
//        weights = vector<float>(outgoingForwardEdges.size(),0);
//        float maxW = 1/sqrt(incomingBackwardEdges.size());
//        for(auto& w: weights) w = math::randomFloat(-maxW,maxW);
//        newWeights = weights;
//        deltaWeights = vector<float>(weights.size(),0);
        
        assert(w.size() == outgoingForwardEdges.size());
        weights = w;
    }
        
    void onRecv(shared_ptr<ForwardPropagationMessage> msg) override;
    void onRecv(shared_ptr<BackwardPropagationMessage> msg) override;

    bool sendBackwardMsgs(vector<shared_ptr<Message>>& msgs) override;
    bool sendForwardMsgs(vector<shared_ptr<Message>>& msgs) override;
};

#endif /* INPUT_NODE_H */

