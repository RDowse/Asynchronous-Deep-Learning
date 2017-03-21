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

#include "messages/message.h"

#include <memory>

class Node;

class Edge{
    typedef std::shared_ptr<Node> NodePtr;
    typedef std::shared_ptr<Message> MsgPtr;
    typedef int channel_type; // TODO update type
    
    unsigned delay;
    channel_type channel;
public:
    enum MessageStatus{
        empty,
        ready,    
        inflight
    };

    MsgPtr msg;
    NodePtr dst;
    NodePtr src;
    MessageStatus msgStatus;
    
    Edge(NodePtr dst, NodePtr src, unsigned delay, channel_type channel) :
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

