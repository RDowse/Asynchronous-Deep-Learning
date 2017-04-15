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

using namespace std;

class ForwardPropagationMessage;
class BackwardPropagationMessage;

// Acts as a message handler
class Node{
protected:
    static int curr_id;
    int m_id;
public:
    vector<shared_ptr<Edge>> incomingEdges;
    vector<shared_ptr<Edge>> outgoingEdges; // replace with weak ptr for circular dependencies
    Node(shared_ptr<GraphSettings> graphSettings){m_id = curr_id; curr_id++;}
    int getId() const{
        return m_id;
    }
    virtual string getType()=0;
    virtual bool readyToSend()=0;
    
    // Additional setup after the graph is constructed
    virtual void setup()=0;
    
    /*
     *  design is not particularly scalable since all nodes must know about all
     * message types. Opt to change for dynamic casting.
     */
    virtual bool onSend(shared_ptr<ForwardPropagationMessage> msg)=0;
    virtual bool onSend(shared_ptr<BackwardPropagationMessage> msg)=0;
    
    virtual void onRecv(shared_ptr<ForwardPropagationMessage> msg)=0;
    virtual void onRecv(shared_ptr<BackwardPropagationMessage> msg)=0;
};

// For registration. NodeFactory.h 
template<typename T> shared_ptr<Node> createT(shared_ptr<GraphSettings> g) { return make_shared<T>(g); }
#endif /* NODE_H */