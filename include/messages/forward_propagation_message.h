/* 
 * File:   forward_propagation_message.h
 * Author: ryan
 *
 * Created on 21 March 2017, 17:24
 */

#ifndef FORWARD_PROPAGATION_MESSAGE_H
#define FORWARD_PROPAGATION_MESSAGE_H

#include "messages/message.h"

class ForwardPropagationMessage: public Message
{
public:
    float value = 0;
    // send message to node
    bool dispatchTo(Node* handler) override;
}; 

#endif /* FORWARD_PROPAGATION_MESSAGE_H */

