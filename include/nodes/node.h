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
#include "states/state.h"

#include <stdexcept>
#include <memory>
#include <vector>
#include <map>
#include <cassert>
#include <iostream>

using namespace std;

class ForwardPropagationMessage;
class BackwardPropagationMessage;

// Acts as a message handler
class Node{
    static int curr_id;
protected:
    int m_id;
    void send(vector<Message*>& msgs, vector<Edge*>& edges);
public:
//    map<int, Edge*> incomingEdges;  // map of edges indexed by their src
//    map<int, Edge*> outgoingEdges;  // map of edges indexed by their dst
    vector<Edge*> incomingEdges; 
    vector<Edge*> outgoingEdges;  
    
    Node(shared_ptr<GraphSettings> settings){m_id = curr_id++;}
    
    int getId() const{ return m_id;}
    virtual string getType()=0;
    
    // check if the node is ready
    virtual bool readyToSend()=0;
    
    // general implementation of adding edges
    virtual void addEdge(Edge* e);
    
    // handle sending of messages and routing for the node
    virtual bool onSend(vector<Message*>& msgs)=0;
    
    // send messages to the corresponding node id, while check validity
    virtual void send(vector<Message*>& msgs);
    
    // handle message receiving for different message types
    virtual void onRecv(ForwardPropagationMessage* msg)=0;
    virtual void onRecv(BackwardPropagationMessage* msg)=0;
};

// For registration. NodeFactory.h 
template<typename T> Node* createT(shared_ptr<GraphSettings> g) { return new T(g); }
#endif /* NODE_H */