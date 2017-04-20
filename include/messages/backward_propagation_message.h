/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   backward_propagation_message.h
 * Author: ryan
 *
 * Created on 21 March 2017, 17:24
 */

#ifndef BACKWARD_PROPAGATION_MESSAGE_H
#define BACKWARD_PROPAGATION_MESSAGE_H

#include "messages/message.h"

class BackwardPropagationMessage: public Message,
    public std::enable_shared_from_this<BackwardPropagationMessage>
{
public:
    float delta = 0;
    // send message to node
    bool dispatchTo(shared_ptr<Node> handler) override;
};

#endif /* BACKWARD_PROPAGATION_MESSAGE_H */

