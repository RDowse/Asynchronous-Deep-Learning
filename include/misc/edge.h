/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   edge.h
 * Author: ryan
 *
 * Created on 19 March 2017, 16:56
 */

#ifndef EDGE_H
#define EDGE_H

#include <memory>

using namespace std;

class Message;
class Node;

class Edge{ 
    unsigned delay;
public:
    enum MessageStatus{
        empty,
        ready,    
        inflight
    };
    shared_ptr<Message> msg;
    Node* dst;
    Node* src;
    MessageStatus msgStatus;
    
    Edge(Node* src, Node* dst, unsigned delay) :
    dst(dst), src(src), delay(delay), msgStatus(MessageStatus::empty) {
    }

    unsigned getDelay()const {
        return delay;
    }
};
#endif /* EDGE_H */