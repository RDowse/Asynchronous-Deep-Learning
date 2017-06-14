/* 
 * File:   backward_propagation_message.h
 * Author: ryan
 *
 * Created on 21 March 2017, 17:24
 */

#ifndef BACKWARD_PROPAGATION_MESSAGE_H
#define BACKWARD_PROPAGATION_MESSAGE_H

#include "messages/message.h"

#include <Eigen/Dense>

class BackwardPropagationMessage: public Message
{
public:
    // numbered batch for debugging
    int batchIndex;
    
    // batch count
    int batchNum = 0;
    
    // individual node values
    Eigen::VectorXf delta;
    Eigen::VectorXf target;
    
    // block node values
    Eigen::MatrixXf matDelta;
    Eigen::MatrixXf matTarget;
    
    // send message to node
    bool dispatchTo(Node* handler) override;
    
    void reset() override;
};

#endif /* BACKWARD_PROPAGATION_MESSAGE_H */

