
#include "messages/backward_propagation_message.h"
#include "nodes/node.h"

bool BackwardPropagationMessage::dispatchTo(Node* handler){
    handler->onRecv(this);
}

void BackwardPropagationMessage::reset(){
        src = 0;
        dst = 0;
        batchIndex = 0;
        batchNum = 0;
        delta = Eigen::VectorXf();
        target = Eigen::VectorXf();
        matDelta = Eigen::MatrixXf();
        matTarget = Eigen::MatrixXf();
}