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
    typedef int channel_type; 
    unsigned delay;
    channel_type channel;
public:
    enum MessageStatus{
        empty,
        ready,    
        inflight
    };

    shared_ptr<Message> msg;
    shared_ptr<Node> dst;
    shared_ptr<Node> src;
    MessageStatus msgStatus;
    
    Edge(shared_ptr<Node> dst, shared_ptr<Node> src, unsigned delay, channel_type channel) :
    dst(dst), src(src), delay(delay), channel(channel) {
    }

    unsigned getDelay()const {
        return delay;
    }

    unsigned getChannel()const {
        return channel;
    } 
};

#endif /* EDGE_H */

