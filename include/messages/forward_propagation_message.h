/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   forward_propagation_message.h
 * Author: ryan
 *
 * Created on 21 March 2017, 17:24
 */

#ifndef FORWARD_PROPAGATION_MESSAGE_H
#define FORWARD_PROPAGATION_MESSAGE_H

#include "messages/message.h"

class ForwardPropagationMessage: public Message,
        public std::enable_shared_from_this<ForwardPropagationMessage>
{
public:
    // send message to node
    bool dispatchTo(shared_ptr<Node> handler) override{
        handler->onRecv(shared_from_this());
    }
    // prepare message sent from node
    bool dispatchFrom(shared_ptr<Node> handler) override{
        handler->onSend(shared_from_this());
    }
}; 

#endif /* FORWARD_PROPAGATION_MESSAGE_H */

