/* 
 * File:   forward_propagation_message.h
 * Author: ryan
 *
 * Created on 21 March 2017, 17:24
 */

#ifndef FORWARD_PROPAGATION_MESSAGE_H
#define FORWARD_PROPAGATION_MESSAGE_H

#include "messages/message.h"

#include <Eigen/Dense>

class ForwardPropagationMessage: public Message
{
public:
    // individual node value
    float activation = 0;
    
    // block node value
    Eigen::MatrixXf matActivation;
    
    // send message to node
    bool dispatchTo(Node* handler) override;
}; 

#endif /* FORWARD_PROPAGATION_MESSAGE_H */

