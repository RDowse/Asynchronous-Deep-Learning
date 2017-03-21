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
#include "messages/message.h"
#include "graphs/graph.h"

#include <memory>

using namespace std;

class Node{
protected:
    static int curr_id;
    int m_id;
    shared_ptr<Graph> graph;
public:
    vector<shared_ptr<Edge>> incomingEdges;
    vector<shared_ptr<Edge>> outgoingEdges;
    Node(shared_ptr<Graph> graph):graph(graph){m_id = curr_id; curr_id++;}
    int getId() const{
        return m_id;
    }
    virtual void onInit()=0;
    virtual bool onSend(shared_ptr<Message>& msg)=0;
    virtual void onRecv(shared_ptr<Message>& msg)=0;
    virtual bool readyToSend()=0;
};

// For registration. NodeFactory.h 
template<typename T> Node * createT(shared_ptr<Graph> g) { return new T(g); }
#endif /* NODE_H */