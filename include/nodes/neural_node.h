/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   neural_node.h
 * Author: ryan
 *
 * Created on 24 April 2017, 00:27
 */

#ifndef NEURAL_NODE_H
#define NEURAL_NODE_H

#include "nodes/node.h"
#include "graphs/dnn_graph_settings.h"

#include <vector>
#include <memory>

using namespace std;

class State;
class ForwardPropagationMessage;
class BackwardPropagationMessage;

class NeuralNode: public Node{
public:
    shared_ptr<DNNGraphSettings> settings;
    
    // sorted edges
    vector<Edge*> incomingForwardEdges;
    vector<Edge*> incomingBackwardEdges;
    vector<Edge*> outgoingBackwardEdges;
    vector<Edge*> outgoingForwardEdges;
    
    // seen counts
    int forwardSeenCount = 0;
    int backwardSeenCount = 0;
    
    NeuralNode(shared_ptr<GraphSettings> settings): Node(settings){}
    
    virtual string getType()=0;
    virtual bool readyToSend(){
        settings->state->readyToSend(this);
    }  
    
    // Handle sending of messages and routing for the node
    virtual bool onSend(vector< shared_ptr<Message> >& msgs){
        settings->state->onSend(this, msgs);
    }
    
    virtual void setWeights(const vector<float>& w){
        cout << "setWeights not implemented for this node, " << m_id << endl;
    }
    
    // Handle message receiving
    virtual void onRecv(shared_ptr<ForwardPropagationMessage> msg)=0;
    virtual void onRecv(shared_ptr<BackwardPropagationMessage> msg)=0;
    
    virtual bool sendBackwardMsgs(vector<shared_ptr<Message>>& msgs)=0;
    virtual bool sendForwardMsgs(vector<shared_ptr<Message>>& msgs)=0;
    
    bool readyToSendForward(){
        return (forwardSeenCount == incomingForwardEdges.size()); 
    }
    bool readyToSendBackward(){
        return (backwardSeenCount == incomingBackwardEdges.size());
    }
};

#endif /* NEURAL_NODE_H */

