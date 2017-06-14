
#include "messages/forward_propagation_message.h"
#include "nodes/node.h"

bool ForwardPropagationMessage::dispatchTo(Node* handler) {
    handler->onRecv(this);
}

void ForwardPropagationMessage::reset(){
        src = 0;
        dst = 0;
        batchIndex = 0;
        updateNumber = 0;
        batchNum = 0;
        activation = Eigen::VectorXf();
        matActivation = Eigen::MatrixXf();
        dataSetType = DataSetType::training;
}