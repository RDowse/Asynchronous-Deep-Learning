/* 
 * File:   forward_propagation_message.h
 * Author: ryan
 *
 * Created on 21 March 2017, 17:24
 */

#ifndef FORWARD_PROPAGATION_MESSAGE_H
#define FORWARD_PROPAGATION_MESSAGE_H

#include "messages/message.h"
#include "common.h"

#include <Eigen/Dense>

class ForwardPropagationMessage: public Message
{
public:
    // numbered batches for debugging
    int batchIndex;
    
    // training/validation/testing data
    DataSetType dataSetType = DataSetType::training;
    
    // individual node value
    Eigen::VectorXf activation;
    
    // block node value
    Eigen::MatrixXf matActivation;
    
    // send message to node
    bool dispatchTo(Node* handler) override;
}; 

#endif /* FORWARD_PROPAGATION_MESSAGE_H */

