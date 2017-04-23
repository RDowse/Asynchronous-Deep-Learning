/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   node.h
 * Author: ryan
 *
 * Created on 19 March 2017, 16:48
 * 
 * Description:
 * Abstract Node
 */

#ifndef NODE_H
#define NODE_H

#include "misc/edge.h"
#include "graphs/graph_settings.h"

#include <memory>
#include <vector>
#include <cassert>
#include <iostream>

using namespace std;

class ForwardPropagationMessage;
class BackwardPropagationMessage;

// Acts as a message handler
class Node{
protected:
    static int curr_id;
    int m_id;
    void send(vector<shared_ptr<Message>>& msgs, vector<Edge*>& edges);
public:
    vector<Edge*> incomingEdges;
    vector<Edge*> outgoingEdges; 
    Node(shared_ptr<GraphSettings> graphSettings){m_id = curr_id; curr_id++;}
    int getId() const{
        return m_id;
    }
    virtual string getType()=0;
    virtual bool readyToSend()=0;
    
    // Additional setup after the graph is constructed
    virtual void setup()=0;
    
    // Handle sending of messages and routing for the node
    virtual bool onSend(vector< shared_ptr<Message> >& msgs)=0;
    
    // Handle message receiving
    virtual void onRecv(shared_ptr<ForwardPropagationMessage> msg)=0;
    virtual void onRecv(shared_ptr<BackwardPropagationMessage> msg)=0;
};

// For registration. NodeFactory.h 
template<typename T> Node* createT(shared_ptr<GraphSettings> g) { return new T(g); }
#endif /* NODE_H */