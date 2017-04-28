/* 
 * File:   backward_propagation_message.h
 * Author: ryan
 *
 * Created on 21 March 2017, 17:24
 */

#ifndef BACKWARD_PROPAGATION_MESSAGE_H
#define BACKWARD_PROPAGATION_MESSAGE_H

#include "messages/message.h"

class BackwardPropagationMessage: public Message
{
public:
    float delta = 0;
    float target = 0;
    // send message to node
    bool dispatchTo(Node* handler) override;
};

#endif /* BACKWARD_PROPAGATION_MESSAGE_H */

