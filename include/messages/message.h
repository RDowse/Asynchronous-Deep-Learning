/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   message.h
 * Author: ryan
 *
 * Created on 19 March 2017, 16:53
 */

#ifndef MESSAGE_H
#define MESSAGE_H

#include <memory>

class Node;

using namespace std;

// Abstract message
class Message{
public:
    // send message to node
    virtual bool dispatchTo(shared_ptr<Node> handler)=0;
    // prepare message sent from node
    virtual bool dispatchFrom(shared_ptr<Node> handler)=0;
};

#endif /* MESSAGE_H */