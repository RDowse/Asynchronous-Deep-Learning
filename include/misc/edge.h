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

    shared_ptr<Node> dst;
    shared_ptr<Node> src;
    MessageStatus msgStatus;
    
    Edge(shared_ptr<Node> dst, shared_ptr<Node> src, unsigned delay) :
    dst(dst), src(src), delay(delay){
    }

    unsigned getDelay()const {
        return delay;
    }
};

namespace edge{
template<typename T> 
shared_ptr<Edge> createT(shared_ptr<Node> dst, shared_ptr<Node> src, unsigned delay)
{ return make_shared<T>(dst,src,delay); }
}
#endif /* EDGE_H */

